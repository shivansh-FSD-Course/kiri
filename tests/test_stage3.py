"""
tests/test_stage3.py

Stage 3 tests: ANE dispatch via CoreML.
These tests run on all platforms — ANE-specific paths are skipped
gracefully on non-Apple Silicon.

Run with: pytest tests/test_stage3.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import platform

import kiri.backend.detect as _bd
_bd.BACKEND = "numpy"

from kiri.backend.ane import (
    is_available, ANELinear, ane_optimize,
    get_stats, cache_stats, _cache
)
from kiri.nn.layers  import Linear, Sequential
from kiri.nn.activations import ReLU
from kiri.model      import Model
from kiri.autograd   import Tensor


# ─── availability ─────────────────────────────────────────────────────────────

class TestANEAvailability:
    def test_is_available_returns_bool(self):
        assert isinstance(is_available(), bool)

    def test_unavailable_on_linux(self):
        if platform.system() != "Darwin":
            assert is_available() is False

    def test_get_stats_returns_dict(self):
        s = get_stats()
        assert "ane_calls"      in s
        assert "ane_avg_ms"     in s
        assert "fallback_calls" in s

    def test_cache_stats_returns_dict(self):
        c = cache_stats()
        assert "cached_models" in c


# ─── ANELinear (CPU fallback path) ────────────────────────────────────────────

class TestANELinearFallback:
    """
    ANELinear falls back to NumPy/Accelerate on non-Apple Silicon.
    These tests verify the fallback is correct — same results as np.matmul.
    """

    def test_forward_correctness(self):
        W = np.random.randn(32, 16).astype(np.float32)
        b = np.random.randn(32).astype(np.float32)
        X = np.random.randn(8, 16).astype(np.float32)

        layer    = ANELinear(W, b)
        result   = layer(X)
        expected = X @ W.T + b

        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)

    def test_forward_no_bias(self):
        W = np.random.randn(16, 32).astype(np.float32)
        X = np.random.randn(4, 32).astype(np.float32)

        layer  = ANELinear(W, None)
        result = layer(X)
        np.testing.assert_allclose(result, X @ W.T, rtol=1e-4, atol=1e-4)

    def test_output_shape(self):
        for batch, in_f, out_f in [(1, 64, 32), (8, 128, 256), (32, 512, 512)]:
            W   = np.random.randn(out_f, in_f).astype(np.float32)
            X   = np.random.randn(batch, in_f).astype(np.float32)
            out = ANELinear(W, None)(X)
            assert out.shape == (batch, out_f)

    def test_from_kiri_linear(self):
        fc  = Linear(32, 16)
        ane = ANELinear.from_kiri_linear(fc)
        assert ane.W.shape == (16, 32)
        assert ane.b is not None

    def test_from_kiri_linear_no_bias(self):
        fc  = Linear(32, 16, bias=False)
        ane = ANELinear.from_kiri_linear(fc)
        assert ane.b is None

    def test_ane_friendly_shape_detection(self):
        # Multiples of 16 → ANE-friendly
        W_good = np.zeros((64, 128), dtype=np.float32)
        assert ANELinear(W_good)._ane_supported is True

        # Not multiple of 16 → not ANE-friendly
        W_bad = np.zeros((50, 100), dtype=np.float32)
        assert ANELinear(W_bad)._ane_supported is False

    def test_large_batch_falls_back(self):
        """Batch > 64 should fall back (ANE compile limit)."""
        W     = np.random.randn(64, 64).astype(np.float32)
        X     = np.random.randn(128, 64).astype(np.float32)  # batch=128
        layer = ANELinear(W)
        out   = layer(X)
        assert out.shape == (128, 64)   # still works via fallback

    def test_output_dtype_float32(self):
        """Output should always be float32 regardless of internal FP16 cast."""
        W   = np.random.randn(32, 16).astype(np.float32)
        X   = np.random.randn(4, 16).astype(np.float32)
        out = ANELinear(W)(X)
        assert out.dtype == np.float32


# ─── ane_optimize ─────────────────────────────────────────────────────────────

class TestAneOptimize:
    def _make_model(self):
        class Net(Model):
            def __init__(self):
                super().__init__()
                self.net = Sequential(
                    Linear(128, 256), ReLU(),
                    Linear(256, 128), ReLU(),
                    Linear(128, 10)
                )
            def forward(self, x): return self.net(x)
        return Net()

    def test_optimize_returns_tuple(self):
        model       = self._make_model()
        result      = ane_optimize(model, verbose=False)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_optimize_report_keys(self):
        model       = self._make_model()
        _, report   = ane_optimize(model, verbose=False)
        if "error" in report:
            pytest.skip("ANE not available on this platform")
        assert "converted"  in report
        assert "skipped"    in report
        assert "cache_size" in report

    def test_optimize_output_unchanged(self):
        """ane_optimize must not change model outputs."""
        model = self._make_model()
        X     = np.random.randn(8, 128).astype(np.float32)

        # Get original output
        model.eval()
        x_t      = Tensor(X, requires_grad=False)
        out_orig = model.forward(x_t).data.copy()

        # Optimize
        model, _ = ane_optimize(model, verbose=False)

        # Re-run
        out_ane = model.forward(x_t)
        # ANE uses FP16 internally — allow slightly larger tolerance
        out_ane_data = out_ane.data if isinstance(out_ane, Tensor) else out_ane
        np.testing.assert_allclose(out_orig, out_ane_data, rtol=1e-2, atol=1e-2)

    def test_optimize_ane_unfriendly_model(self):
        """Model with non-multiple-of-16 shapes should have 0 converted."""
        class OddNet(Model):
            def __init__(self):
                super().__init__()
                self.fc = Linear(100, 50)   # 100, 50 — not multiples of 16
            def forward(self, x): return self.fc(x)

        model       = OddNet()
        _, report   = ane_optimize(model, verbose=False)
        # On non-macOS, ANE unavailable → error key
        # On macOS, shapes aren't ANE-friendly → converted=0
        if "error" not in report:
            assert report["converted"] == 0
            assert report["skipped"]   == 1

    def test_optimize_does_not_break_training(self):
        """After ane_optimize, model should still be trainable (falls back to numpy)."""
        model       = self._make_model()
        _, _        = ane_optimize(model, verbose=False)

        X = np.random.randn(32, 128).astype(np.float32)
        y = np.random.randint(0, 10, 32).astype(np.int32)

        # Should not raise
        from kiri.nn.loss import cross_entropy
        x_t  = Tensor(X, requires_grad=False)
        pred = model.forward(x_t)
        # pred might be numpy array (from ANELinear) — wrap if needed
        if not isinstance(pred, Tensor):
            pred = Tensor(pred, requires_grad=False)
        loss = cross_entropy(pred, y)
        assert loss.item() > 0


# ─── model cache ──────────────────────────────────────────────────────────────

class TestModelCache:
    def test_cache_grows(self):
        _cache.clear()
        initial = len(_cache)

        # Each unique shape gets a new cache entry (on macOS only)
        if is_available():
            for in_f in [16, 32, 48]:
                W = np.random.randn(16, in_f).astype(np.float32)
                X = np.random.randn(4, in_f).astype(np.float32)
                ANELinear(W)(X)
            assert len(_cache) > initial

    def test_cache_hit_same_shape(self):
        """Two layers with same shape should reuse the same compiled model."""
        if not is_available():
            pytest.skip("ANE not available")

        _cache.clear()
        W = np.random.randn(32, 32).astype(np.float32)
        X = np.random.randn(8, 32).astype(np.float32)

        ANELinear(W)(X)   # first — compiles
        size1 = len(_cache)
        ANELinear(W)(X)   # second — should hit cache
        size2 = len(_cache)

        assert size1 == size2   # no new entry

    def test_cache_clear(self):
        _cache.clear()
        assert len(_cache) == 0
