"""
kiri/benchmark.py

kiri.benchmark() — measures real performance across ops and compares
to PyTorch MPS and MLX where available.

Usage:
    import kiri
    kiri.benchmark()
    kiri.benchmark(quick=True)   # shorter run
"""

import time
import numpy as np
import platform
from kiri.backend.detect   import BACKEND
from kiri.backend.accelerate import is_available as accel_available, sgemm


# ─── timing helpers ───────────────────────────────────────────────────────────

def _timer(fn, warmup=3, runs=20):
    """Returns (mean_ms, std_ms) over `runs` timed calls after `warmup`."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))


def _sync_mlx():
    if BACKEND == "mlx":
        import mlx.core as mx
        mx.eval(mx.array([0.0]))


# ─── individual op benchmarks ─────────────────────────────────────────────────

def _bench_matmul(M, N, K, runs=20):
    """Compare numpy matmul vs Accelerate sgemm for a given shape."""
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)

    np_ms, np_std  = _timer(lambda: np.matmul(A, B), runs=runs)
    results = {"numpy": (np_ms, np_std)}

    if accel_available():
        ac_ms, ac_std = _timer(lambda: sgemm(A, B), runs=runs)
        results["accelerate"] = (ac_ms, ac_std)

    if BACKEND == "mlx":
        import mlx.core as mx
        A_mx = mx.array(A); B_mx = mx.array(B)
        def mlx_mm(): r = A_mx @ B_mx; mx.eval(r)
        mlx_ms, mlx_std = _timer(mlx_mm, runs=runs)
        results["mlx"] = (mlx_ms, mlx_std)

    return results


def _bench_linear_relu(batch, in_f, out_f, runs=20):
    """Benchmark Linear+ReLU: fused vs unfused vs MLX."""
    from kiri.nn.layers import Linear
    from kiri.autograd import Tensor
    from kiri.backend.fusion import linear_relu_fused
    import numpy as np

    X  = np.random.randn(batch, in_f).astype(np.float32)
    fc = Linear(in_f, out_f)

    # Extract weight/bias as numpy regardless of backend
    if BACKEND == "mlx":
        import mlx.core as mx
        W = np.array(fc.weight)
        b = np.array(fc.b) if fc.b is not None else None
    else:
        W = fc.weight.data
        b = fc.b.data if fc.b is not None else None

    def unfused():
        if BACKEND == "mlx":
            import mlx.core as mx
            out = mx.array(X) @ fc.weight.T
            if fc.b is not None: out = out + fc.b
            out = mx.maximum(out, 0)
            mx.eval(out)
        else:
            x_t = Tensor(X, requires_grad=False)
            out = x_t.matmul(fc.weight.T)
            if fc.b is not None: out = out + fc.b
            out.relu()

    def fused():
        linear_relu_fused(X, W, b)

    uf_ms, uf_std = _timer(unfused, runs=runs)
    fu_ms, fu_std = _timer(fused,   runs=runs)
    results = {"unfused": (uf_ms, uf_std), "fused": (fu_ms, fu_std)}

    if BACKEND == "mlx":
        import mlx.core as mx
        W_mx = fc.weight
        b_mx = fc.b
        X_mx = mx.array(X)
        def mlx_fn():
            out = X_mx @ W_mx.T
            if b_mx is not None: out = out + b_mx
            out = mx.maximum(out, 0)
            mx.eval(out)
        mlx_ms, mlx_std = _timer(mlx_fn, runs=runs)
        results["mlx"] = (mlx_ms, mlx_std)

    return results


def _bench_full_model(runs=10):
    """End-to-end forward pass: 512→1024→512→256→10."""
    import kiri
    import kiri.nn as nn

    class Net(kiri.Model):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(512, 1024), nn.ReLU(),
                nn.Linear(1024, 512), nn.ReLU(),
                nn.Linear(512, 256),  nn.ReLU(),
                nn.Linear(256, 10)
            )
        def forward(self, x): return self.net(x)

    X     = np.random.randn(256, 512).astype(np.float32)
    model = Net()
    model.eval()

    if BACKEND == "mlx":
        import mlx.core as mx
        X_mx = mx.array(X)
        def fn():
            out = model.forward(X_mx)
            mx.eval(out)
    else:
        from kiri.autograd import Tensor
        x_t = Tensor(X, requires_grad=False)
        def fn(): model.forward(x_t)

    ms, std = _timer(fn, warmup=5, runs=runs)

    # Now fuse and re-bench (CPU only — MLX already fuses)
    fused_ms = None
    if BACKEND != "mlx":
        from kiri.backend.fusion import fuse_model
        model.eval()
        fuse_model(model)
        ms_f, std_f = _timer(fn, warmup=5, runs=runs)
        fused_ms = (ms_f, std_f)

    return {"baseline": (ms, std), "fused": fused_ms}


def _bench_training_throughput(runs=5):
    """Training step throughput: samples/sec."""
    import kiri
    import kiri.nn as nn

    class Net(kiri.Model):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(512, 256)
            self.act = nn.ReLU()
            self.fc2 = nn.Linear(256, 10)
        def forward(self, x): return self.fc2(self.act(self.fc1(x)))

    BATCH = 512
    X = np.random.randn(BATCH, 512).astype(np.float32)
    y = np.random.randint(0, 10, BATCH).astype(np.int32)

    model = Net()
    if BACKEND == "mlx":
        import mlx.core as mx
        import mlx.nn as mlx_nn
        from kiri.optim import Adam
        opt = Adam(model, lr=1e-3)
        def step():
            x_t = mx.array(X); y_t = mx.array(y)
            def loss_fn(m, x, yy): return mlx_nn.losses.cross_entropy(m.forward(x), yy).mean()
            loss, grads = mlx_nn.value_and_grad(model, loss_fn)(model, x_t, y_t)
            opt._opt.update(model, grads)
            mx.eval(model.parameters())
    else:
        from kiri.autograd import Tensor
        from kiri.nn.loss import cross_entropy
        from kiri.optim import Adam
        opt = Adam(model.parameters(), lr=1e-3)
        def step():
            opt.zero_grad()
            x_t  = Tensor(X, requires_grad=False)
            pred = model.forward(x_t)
            loss = cross_entropy(pred, y)
            loss.backward()
            opt.step()

    ms, std = _timer(step, warmup=3, runs=runs)
    samples_per_sec = BATCH / (ms / 1000)
    return {"ms_per_step": (ms, std), "samples_per_sec": samples_per_sec}


# ─── main benchmark entry point ───────────────────────────────────────────────

def benchmark(quick=False):
    """
    Run the full Kiri benchmark suite.

    Parameters
    ----------
    quick : bool
        Run fewer iterations for a faster result (useful in CI).
    """
    runs_mm    = 10 if quick else 50
    runs_layer = 10 if quick else 30
    runs_model = 5  if quick else 15
    runs_train = 3  if quick else 8

    sep = "─" * 62
    print("\n" + "=" * 62)
    print("  KIRI BENCHMARK SUITE")
    print(f"  Backend  : {BACKEND}")
    print(f"  Platform : {platform.platform()}")
    print(f"  Accelerate: {'✓ available' if accel_available() else '✗ not available'}")
    print("=" * 62)

    # ── 1. Matmul ─────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  1. Matrix Multiply (ms, lower is better)")
    print(sep)
    print(f"  {'Shape':<22} {'NumPy':>10}  {'Accelerate':>12}  {'MLX':>8}")
    print(f"  {'─'*22} {'─'*10}  {'─'*12}  {'─'*8}")

    for M, N, K in [(64, 64, 64), (256, 256, 256), (512, 1024, 512), (1024, 1024, 1024)]:
        r = _bench_matmul(M, N, K, runs=runs_mm)
        np_s   = f"{r['numpy'][0]:.3f}ms"
        ac_s   = f"{r['accelerate'][0]:.3f}ms" if "accelerate" in r else "N/A"
        mlx_s  = f"{r['mlx'][0]:.3f}ms"         if "mlx" in r        else "N/A"
        speedup = ""
        if "accelerate" in r:
            s = r["numpy"][0] / r["accelerate"][0]
            speedup = f" ({s:.1f}x)"
        print(f"  ({M}×{K}) @ ({K}×{N}){'':<5} {np_s:>10}  {ac_s+speedup:>12}  {mlx_s:>8}")

    # ── 2. Linear + ReLU fusion ───────────────────────────────────────────
    print(f"\n{sep}")
    print("  2. Linear+ReLU: Fused vs Unfused (ms)")
    print(sep)
    print(f"  {'Shape':<22} {'Unfused':>10}  {'Fused':>10}  {'MLX':>8}  {'Speedup':>8}")
    print(f"  {'─'*22} {'─'*10}  {'─'*10}  {'─'*8}  {'─'*8}")

    for batch, in_f, out_f in [(64, 256, 256), (256, 512, 512), (512, 1024, 512)]:
        r  = _bench_linear_relu(batch, in_f, out_f, runs=runs_layer)
        uf = f"{r['unfused'][0]:.3f}ms"
        fu = f"{r['fused'][0]:.3f}ms"
        mx_s = f"{r['mlx'][0]:.3f}ms" if "mlx" in r else "N/A"
        sp = r["unfused"][0] / r["fused"][0]
        print(f"  ({batch}×{in_f})→{out_f:<12} {uf:>10}  {fu:>10}  {mx_s:>8}  {sp:>7.2f}x")

    # ── 3. Full model forward ─────────────────────────────────────────────
    print(f"\n{sep}")
    print("  3. Full Model Forward Pass (512→1024→512→256→10, batch=256)")
    print(sep)
    r = _bench_full_model(runs=runs_model)
    ms, std = r["baseline"]
    print(f"  Baseline : {ms:.3f}ms ± {std:.3f}ms")
    if r["fused"] is not None:
        fms, fstd = r["fused"]
        sp = ms / fms
        print(f"  Fused    : {fms:.3f}ms ± {fstd:.3f}ms  ({sp:.2f}x faster)")
    elif BACKEND == "mlx":
        print(f"  Fused    : N/A (MLX fuses automatically)")

    # ── 4. Training throughput ────────────────────────────────────────────
    print(f"\n{sep}")
    print("  4. Training Throughput (512→256→10, batch=512)")
    print(sep)
    r  = _bench_training_throughput(runs=runs_train)
    ms = r["ms_per_step"][0]
    sp = r["samples_per_sec"]
    print(f"  {ms:.2f}ms/step  │  {sp:,.0f} samples/sec")

    print(f"\n{'='*62}\n")
