"""
kiri/backend/ane_bench.py

Benchmarks ANE dispatch vs MLX vs NumPy for Linear layer inference.
Run this on your Mac to see actual ANE utilization numbers.

Usage:
    python3 -c "from kiri.backend.ane_bench import run; run()"
"""

import time
import numpy as np
import platform

def run():
    print("\n" + "=" * 62)
    print("  ANE DISPATCH BENCHMARK")
    print(f"  Platform : {platform.platform()}")
    print("=" * 62)

    from kiri.backend.ane import is_available, ANELinear, get_stats, cache_stats
    from kiri.backend.accelerate import sgemm, is_available as accel_ok
    from kiri.backend.detect import BACKEND

    if not is_available():
        print("\n  ✗ ANE not available on this platform.")
        print("  Run this on an Apple Silicon Mac with coremltools installed:")
        print("    pip install coremltools")
        return

    print(f"\n  ANE dispatch: ✓ available")
    print(f"  Accelerate:   {'✓' if accel_ok() else '✗'} available")
    print(f"  MLX backend:  {'✓' if BACKEND == 'mlx' else '✗'} active\n")

    def timer(fn, warmup=3, runs=20):
        for _ in range(warmup): fn()
        t = []
        for _ in range(runs):
            t0 = time.perf_counter()
            fn()
            t.append((time.perf_counter() - t0) * 1000)
        return float(np.mean(t)), float(np.std(t))

    sep = "─" * 62
    print(sep)
    print("  Linear layer inference (ms, lower is better)")
    print("  Shape = (batch × in) → out")
    print(sep)
    print(f"  {'Shape':<28} {'NumPy':>8}  {'Accel':>8}  {'ANE':>8}  {'MLX':>8}")
    print(f"  {'─'*28} {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

    # Test shapes — ANE-friendly (multiples of 16) and unfriendly
    shapes = [
        (1,   64,  64,  True),    # tiny — ANE overhead visible
        (8,   128, 128, True),
        (16,  256, 256, True),    # sweet spot starts here
        (32,  512, 512, True),
        (64,  512, 512, True),
        (32,  768, 768, True),    # BERT-like
        (32,  1024, 1024, True),  # GPT-2-like
        (1,   50,  50,  False),   # ANE-unfriendly shape
        (32,  100, 200, False),   # ANE-unfriendly shape
    ]

    for batch, in_f, out_f, ane_friendly in shapes:
        W = np.random.randn(out_f, in_f).astype(np.float32)
        b = np.random.randn(out_f).astype(np.float32)
        X = np.random.randn(batch, in_f).astype(np.float32)

        # NumPy
        np_ms, _ = timer(lambda: X @ W.T + b)

        # Accelerate
        if accel_ok():
            ac_ms, _ = timer(lambda: sgemm(X, W) + b)
        else:
            ac_ms = float("nan")

        # ANE
        ane_layer = ANELinear(W, b)
        ane_ms, _ = timer(lambda: ane_layer(X))

        # MLX
        if BACKEND == "mlx":
            import mlx.core as mx
            W_mx = mx.array(W); b_mx = mx.array(b); X_mx = mx.array(X)
            def mlx_fn():
                r = X_mx @ W_mx.T + b_mx
                mx.eval(r)
            mlx_ms, _ = timer(mlx_fn)
        else:
            mlx_ms = float("nan")

        label  = f"({batch}×{in_f})→{out_f}"
        flag   = " ✓" if ane_friendly else " ✗"
        ac_s   = f"{ac_ms:.3f}" if not np.isnan(ac_ms) else "  N/A"
        mlx_s  = f"{mlx_ms:.3f}" if not np.isnan(mlx_ms) else "  N/A"

        # Highlight ANE win
        ane_s  = f"{ane_ms:.3f}"
        best   = min(v for v in [np_ms, ac_ms if not np.isnan(ac_ms) else 999,
                                  ane_ms, mlx_ms if not np.isnan(mlx_ms) else 999])
        if ane_ms == best:
            ane_s = f"\033[32m{ane_ms:.3f}\033[0m"  # green

        print(f"  {label+flag:<28} {np_ms:>8.3f}  {ac_s:>8}  {ane_s:>8}  {mlx_s:>8}")

    print(f"\n{sep}")
    print("  ANE dispatch statistics")
    print(sep)
    stats = get_stats()
    cache = cache_stats()
    print(f"  ANE calls:      {stats['ane_calls']}")
    print(f"  ANE avg latency:{stats['ane_avg_ms']:.3f}ms")
    print(f"  Fallback calls: {stats['fallback_calls']}")
    print(f"  Cached models:  {cache['cached_models']}")

    print(f"\n{sep}")
    print("  Compilation overhead (first call vs cached)")
    print(sep)
    W2 = np.random.randn(256, 256).astype(np.float32)
    b2 = np.random.randn(256).astype(np.float32)
    X2 = np.random.randn(32, 256).astype(np.float32)

    # First call — compiles
    t0 = time.perf_counter()
    layer2 = ANELinear(W2, b2)
    layer2(X2)  # triggers compile
    first_ms = (time.perf_counter() - t0) * 1000

    # Subsequent calls — cached
    cached_ms, _ = timer(lambda: layer2(X2), warmup=1, runs=10)

    print(f"  First call (compile): {first_ms:.1f}ms")
    print(f"  Cached calls:         {cached_ms:.3f}ms")
    print(f"  Compile overhead:     {first_ms/cached_ms:.0f}x (one-time cost)\n")
    print("=" * 62 + "\n")
