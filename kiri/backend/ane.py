"""
kiri/backend/ane.py

Apple Neural Engine dispatch via CoreML.

Strategy:
  - Convert Linear layer weight matrices to CoreML .mlpackage at runtime
  - Request MLComputeUnits.cpuAndNeuralEngine — CoreML routes to ANE
    automatically for supported shapes (multiples of 16, FP16)
  - Cache compiled models in memory so we don't recompile each forward pass
  - Fall back to MLX/NumPy silently if ANE dispatch fails

Why CoreML instead of private APIs:
  - Public, stable — won't break on macOS updates
  - Works on all Apple Silicon (M1 → M4)
  - CoreML automatically handles ANE routing for supported ops
  - No reverse engineering required

ANE-friendly shapes:
  - Input/output dimensions should be multiples of 16
  - FP16 preferred (ANE converts FP32 internally anyway)
  - Batch size 1–64 works well; larger batches better on GPU

Usage:
  from kiri.backend.ane import ANELinear, is_available, get_stats
"""

import os
import platform
import tempfile
import threading
import time
import numpy as np
from typing import Optional, Dict, Tuple

# ── availability check ────────────────────────────────────────────────────────

_ANE_AVAILABLE = False
_ct            = None          # coremltools module
_Foundation    = None          # objc Foundation for model loading

def _check_availability() -> bool:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    try:
        import coremltools as ct
        globals()["_ct"] = ct
        return True
    except ImportError:
        return False

_ANE_AVAILABLE = _check_availability()


def is_available() -> bool:
    """Returns True if CoreML ANE dispatch is available (Apple Silicon + coremltools)."""
    return _ANE_AVAILABLE


# ── stats tracking ────────────────────────────────────────────────────────────

class _Stats:
    def __init__(self):
        self.ane_calls     = 0
        self.ane_ms_total  = 0.0
        self.fallback_calls = 0
        self._lock         = threading.Lock()

    def record_ane(self, ms: float):
        with self._lock:
            self.ane_calls    += 1
            self.ane_ms_total += ms

    def record_fallback(self):
        with self._lock:
            self.fallback_calls += 1

    def summary(self) -> dict:
        with self._lock:
            avg = self.ane_ms_total / max(self.ane_calls, 1)
            return {
                "ane_calls":      self.ane_calls,
                "ane_avg_ms":     round(avg, 4),
                "fallback_calls": self.fallback_calls,
            }

_stats = _Stats()


def get_stats() -> dict:
    """Return ANE dispatch statistics."""
    return _stats.summary()


# ── CoreML model builder ──────────────────────────────────────────────────────

def _build_linear_coreml(W: np.ndarray, b: Optional[np.ndarray],
                          batch_size: int = 1) -> object:
    """
    Build a CoreML model for y = x @ W.T + b.

    CoreML expects:
      input  : (batch_size, in_features)  — Float16 for ANE routing
      output : (batch_size, out_features)

    ANE routing is most reliable when:
      - in_features and out_features are multiples of 16
      - batch_size is small (1–64)
      - weights are FP16
    """
    ct = _ct
    import coremltools.proto.FeatureTypes_pb2 as ft

    out_features, in_features = W.shape

    # Build MIL (Model Intermediate Language) program
    @ct.program(
        input_features=[
            ct.TensorType(name="x",
                         shape=(batch_size, in_features),
                         dtype=np.float16)
        ]
    )
    def linear_prog(x):
        # Cast to float16 for ANE
        W_fp16 = W.astype(np.float16)
        w_var  = ct.mb.const(val=W_fp16, name="weight")
        out    = ct.mb.linear(x=x, weight=w_var, name="linear_out")
        if b is not None:
            b_fp16 = b.astype(np.float16)
            b_var  = ct.mb.const(val=b_fp16, name="bias")
            out    = ct.mb.add(x=out, y=b_var, name="output")
        return out

    model = ct.convert(
        linear_prog,
        compute_units=ct.ComputeUnit.CPU_AND_NE,   # request ANE
        minimum_deployment_target=ct.target.macOS13,
    )
    return model


def _build_matmul_coreml(shape_A: Tuple[int, int],
                          shape_B: Tuple[int, int]) -> object:
    """
    Build a CoreML model for C = A @ B.
    Used for embedding lookups and attention score computation.
    """
    ct = _ct
    M, K = shape_A
    K2, N = shape_B
    assert K == K2

    @ct.program(
        input_features=[
            ct.TensorType(name="A", shape=(M, K), dtype=np.float16),
            ct.TensorType(name="B", shape=(K, N), dtype=np.float16),
        ]
    )
    def matmul_prog(A, B):
        return ct.mb.matmul(x=A, y=B, name="output")

    return ct.convert(
        matmul_prog,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS13,
    )


# ── model cache ───────────────────────────────────────────────────────────────

class _ModelCache:
    """
    In-memory cache of compiled CoreML models keyed by (shape, bias_flag).
    Avoids recompiling the same Linear shape repeatedly.
    """
    def __init__(self, maxsize: int = 64):
        self._cache:  Dict[tuple, object] = {}
        self._maxsize = maxsize
        self._lock    = threading.Lock()

    def get(self, key: tuple) -> Optional[object]:
        with self._lock:
            return self._cache.get(key)

    def put(self, key: tuple, model: object):
        with self._lock:
            if len(self._cache) >= self._maxsize:
                # evict oldest
                oldest = next(iter(self._cache))
                del self._cache[oldest]
            self._cache[key] = model

    def __len__(self):
        return len(self._cache)

    def clear(self):
        with self._lock:
            self._cache.clear()


_cache = _ModelCache(maxsize=128)


def cache_stats() -> dict:
    return {"cached_models": len(_cache)}


# ── ANE Linear ────────────────────────────────────────────────────────────────

class ANELinear:
    """
    CoreML-backed Linear layer that dispatches to the Apple Neural Engine.

    Drop-in for the inference path of kiri.nn.Linear on Apple Silicon.
    Falls back to MLX/NumPy automatically for unsupported shapes.

    ANE-friendly shapes (for best routing):
      - in_features  multiple of 16
      - out_features multiple of 16
      - batch_size   1–64

    Usage:
        from kiri.backend.ane import ANELinear, is_available
        if is_available():
            ane_layer = ANELinear.from_kiri_linear(my_linear_layer)
            output = ane_layer(input_array)
    """

    # Shapes that ANE handles well
    _ANE_MIN_DIM = 16

    def __init__(self, W: np.ndarray, b: Optional[np.ndarray] = None):
        self.W   = W.astype(np.float32)   # keep FP32 master copy
        self.b   = b.astype(np.float32) if b is not None else None
        self._compiled: Dict[int, object] = {}   # batch_size → compiled model
        self._ane_supported = self._check_shape_support()

    def _check_shape_support(self) -> bool:
        """ANE works best with dims that are multiples of 16."""
        out_f, in_f = self.W.shape
        return (in_f  % self._ANE_MIN_DIM == 0 and
                out_f % self._ANE_MIN_DIM == 0)

    def _get_or_compile(self, batch_size: int) -> Optional[object]:
        """Return cached CoreML model for this batch size, compiling if needed."""
        if not _ANE_AVAILABLE:
            return None

        cache_key = (self.W.shape, batch_size, self.b is not None)
        model = _cache.get(cache_key)
        if model is not None:
            return model

        try:
            model = _build_linear_coreml(self.W, self.b, batch_size)
            _cache.put(cache_key, model)
            return model
        except Exception as e:
            return None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass. Tries ANE dispatch, falls back to numpy matmul.
        x: (batch, in_features) float32
        """
        if not _ANE_AVAILABLE or not self._ane_supported:
            _stats.record_fallback()
            return self._numpy_forward(x)

        # Only dispatch to ANE for batch sizes we can compile for
        batch = x.shape[0] if x.ndim == 2 else 1
        if batch > 64:
            _stats.record_fallback()
            return self._numpy_forward(x)

        model = self._get_or_compile(batch)
        if model is None:
            _stats.record_fallback()
            return self._numpy_forward(x)

        try:
            t0 = time.perf_counter()
            # CoreML expects dict input, FP16
            result = model.predict({"x": x.astype(np.float16)})
            ms = (time.perf_counter() - t0) * 1000
            _stats.record_ane(ms)
            # Get output — CoreML names it "output" or "linear_out"
            out = result.get("output") or result.get("linear_out") or list(result.values())[0]
            return out.astype(np.float32)
        except Exception:
            _stats.record_fallback()
            return self._numpy_forward(x)

    def _numpy_forward(self, x: np.ndarray) -> np.ndarray:
        from kiri.backend.accelerate import sgemm, is_available as accel_ok
        out = sgemm(x, self.W.T) if accel_ok() else x @ self.W.T
        if self.b is not None:
            out += self.b
        return out

    @classmethod
    def from_kiri_linear(cls, layer) -> "ANELinear":
        """Create ANELinear from a kiri.nn.Linear layer."""
        from kiri.backend.detect import BACKEND
        if BACKEND == "mlx":
            import numpy as np
            W = np.array(layer.weight)
            b = np.array(layer.b) if layer.b is not None else None
        else:
            W = layer.weight.data
            b = layer.b.data if layer.b is not None else None
        return cls(W, b)

    @property
    def shape(self):
        return self.W.shape


# ── model-level ANE conversion ────────────────────────────────────────────────

def ane_optimize(model, verbose: bool = True) -> Tuple[object, dict]:
    """
    Convert all Linear layers in a kiri Model to ANE-backed layers.

    Only converts layers with ANE-friendly shapes (multiples of 16).
    Everything else stays as-is.

    Returns (model, report) where report has conversion stats.

    Usage:
        model.eval()
        model, report = kiri.ane_optimize(model)
        print(report)
        # {'converted': 3, 'skipped': 1, 'reason': [...]}
    """
    if not _ANE_AVAILABLE:
        return model, {"error": "CoreML not available on this platform"}

    from kiri.nn.layers import Linear, Sequential

    converted = 0
    skipped   = 0
    reasons   = []

    def _convert_attr(obj, attr_name, layer):
        nonlocal converted, skipped
        ane = ANELinear.from_kiri_linear(layer)
        if ane._ane_supported:
            setattr(obj, attr_name, ane)
            converted += 1
            if verbose:
                print(f"  ✓ ANE: {attr_name} {layer.weight.shape if hasattr(layer.weight, 'shape') else ''}")
        else:
            skipped += 1
            shape = layer.weight.shape if hasattr(layer.weight, 'shape') else '?'
            reasons.append(f"{attr_name}: shape {shape} not multiple of 16")

    def _scan(obj):
        for name, attr in vars(obj).items():
            if isinstance(attr, Linear):
                _convert_attr(obj, name, attr)
            elif isinstance(attr, Sequential):
                for i, layer in enumerate(attr.layers):
                    if isinstance(layer, Linear):
                        ane = ANELinear.from_kiri_linear(layer)
                        if ane._ane_supported:
                            attr.layers[i] = ane
                            converted += 1
                        else:
                            skipped += 1
            elif hasattr(attr, '__dict__') and hasattr(attr, 'forward'):
                _scan(attr)

    _scan(model)

    report = {
        "converted":  converted,
        "skipped":    skipped,
        "skipped_reasons": reasons,
        "cache_size": len(_cache),
    }
    return model, report
