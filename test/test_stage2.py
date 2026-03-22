"""
tests/test_stage2.py

Stage 2 tests: Accelerate BLAS, operator fusion, benchmark smoke test.
Run with: pytest tests/test_stage2.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import platform

import kiri.backend.detect as _bd
_bd.BACKEND = "numpy"

from kiri.backend.accelerate import sgemm, sgemm_batched, is_available
from kiri.backend.fusion     import (
    linear_relu_fused, linear_gelu_fused, linear_sigmoid_fused,
    fuse_model, FusedLinear
)
from kiri.autograd   import Tensor
from kiri.nn.layers  import Linear, Sequential
from kiri.nn.activations import ReLU, GELU, Sigmoid
from kiri.model      import Model


# ─── Accelerate BLAS ─────────────────────────────────────────────────────────

class TestAccelerate:
    def test_sgemm_correctness(self):
        """sgemm must match np.matmul to float32 precision."""
        A = np.random.randn(64, 128).astype(np.float32)
        B = np.random.randn(128, 64).astype(np.float32)
        expected = A @ B
        result   = sgemm(A, B)
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)

    def test_sgemm_shapes(self):
        """Various shapes produce correct output dimensions."""
        for M, K, N in [(1, 8, 1), (32, 64, 16), (256, 512, 128)]:
            A = np.random.randn(M, K).astype(np.float32)
            B = np.random.randn(K, N).astype(np.float32)
            C = sgemm(A, B)
            assert C.shape == (M, N)

    def test_sgemm_fallback_non_float32(self):
        """Non-float32 inputs fall back to np.matmul without error."""
        A = np.random.randn(8, 8).astype(np.float64)
        B = np.random.randn(8, 8).astype(np.float64)
        C = sgemm(A, B)
        np.testing.assert_allclose(C, A @ B, rtol=1e-10)

    def test_sgemm_batched(self):
        A = np.random.randn(4, 16, 32).astype(np.float32)
        B = np.random.randn(4, 32, 8).astype(np.float32)
        C = sgemm_batched(A, B)
        assert C.shape == (4, 16, 8)
        np.testing.assert_allclose(C, np.matmul(A, B), rtol=1e-4, atol=1e-4)

    def test_matmul_uses_accelerate(self):
        """Tensor.matmul should use sgemm when available."""
        A = Tensor(np.random.randn(32, 64).astype(np.float32))
        B = Tensor(np.random.randn(64, 16).astype(np.float32))
        C = A.matmul(B)
        assert C.shape == (32, 16)
        # Gradient still works
        C.sum().backward()
        assert A.grad.shape == (32, 64)
        assert B.grad.shape == (64, 16)

    def test_availability_reported(self):
        """is_available() returns bool without crashing."""
        assert isinstance(is_available(), bool)
        # On macOS it should be True; on Linux False
        if platform.system() == "Darwin":
            assert is_available() is True
        else:
            assert is_available() is False


# ─── Operator fusion ─────────────────────────────────────────────────────────

class TestFusion:
    def _linear_data(self, in_f=32, out_f=16, batch=8):
        W = np.random.randn(out_f, in_f).astype(np.float32)
        b = np.random.randn(out_f).astype(np.float32)
        X = np.random.randn(batch, in_f).astype(np.float32)
        return X, W, b

    def test_linear_relu_correctness(self):
        X, W, b = self._linear_data()
        expected = np.maximum(X @ W.T + b, 0)
        result   = linear_relu_fused(X, W, b)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_linear_relu_no_bias(self):
        X, W, _ = self._linear_data()
        expected = np.maximum(X @ W.T, 0)
        result   = linear_relu_fused(X, W, None)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_linear_gelu_correctness(self):
        import math
        X, W, b = self._linear_data()
        pre      = X @ W.T + b
        t        = np.tanh(np.sqrt(2/math.pi) * (pre + 0.044715 * pre**3))
        expected = 0.5 * pre * (1 + t)
        result   = linear_gelu_fused(X, W, b)
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)

    def test_linear_sigmoid_correctness(self):
        X, W, b = self._linear_data()
        pre      = X @ W.T + b
        expected = 1 / (1 + np.exp(-pre))
        result   = linear_sigmoid_fused(X, W, b)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_fuse_model_detects_linear_relu(self):
        """fuse_model should find and fuse Linear→ReLU in a Sequential."""
        class Net(Model):
            def __init__(self):
                super().__init__()
                self.net = Sequential(
                    Linear(16, 32), ReLU(),
                    Linear(32, 8),  ReLU(),
                    Linear(8, 4)
                )
            def forward(self, x): return self.net(x)

        model = Net()
        model.eval()
        _, n_fused = fuse_model(model)
        assert n_fused == 2  # two Linear→ReLU pairs
        # Layers should now be FusedLinear, FusedLinear, Linear
        assert isinstance(model.net.layers[0], FusedLinear)
        assert isinstance(model.net.layers[1], FusedLinear)
        assert isinstance(model.net.layers[2], Linear)

    def test_fuse_model_output_unchanged(self):
        """Fused model must produce identical output to unfused."""
        class Net(Model):
            def __init__(self):
                super().__init__()
                self.net = Sequential(
                    Linear(8, 16), ReLU(),
                    Linear(16, 4), ReLU(),
                    Linear(4, 2)
                )
            def forward(self, x): return self.net(x)

        X = np.random.randn(4, 8).astype(np.float32)

        # Get unfused output
        model = Net()
        model.eval()
        x_t      = Tensor(X, requires_grad=False)
        out_orig = model.forward(x_t).data.copy()

        # Fuse and re-run
        fuse_model(model)
        out_fused = model.forward(x_t).data

        np.testing.assert_allclose(out_orig, out_fused, rtol=1e-5, atol=1e-5)

    def test_fuse_model_no_gelu_missed(self):
        """Linear→GELU also gets fused."""
        class Net(Model):
            def __init__(self):
                super().__init__()
                self.net = Sequential(Linear(8, 8), GELU())
            def forward(self, x): return self.net(x)

        model = Net()
        model.eval()
        _, n = fuse_model(model)
        assert n == 1
        assert isinstance(model.net.layers[0], FusedLinear)
        assert model.net.layers[0].activation == "gelu"

    def test_fused_parameters_preserved(self):
        """FusedLinear.parameters() still returns all trainable tensors."""
        fc    = Linear(8, 16)
        fused = FusedLinear(fc, "relu")
        params = fused.parameters()
        assert len(params) == 2  # weight + bias


# ─── Benchmark smoke test ────────────────────────────────────────────────────

class TestBenchmark:
    def test_benchmark_runs(self):
        """benchmark(quick=True) should run without error."""
        from kiri.benchmark import benchmark
        # Should not raise
        benchmark(quick=True)

    def test_matmul_bench_returns_results(self):
        from kiri.benchmark import _bench_matmul
        r = _bench_matmul(64, 64, 64, runs=5)
        assert "numpy" in r
        ms, std = r["numpy"]
        assert ms > 0
        assert std >= 0

    def test_linear_relu_bench(self):
        from kiri.benchmark import _bench_linear_relu
        r = _bench_linear_relu(32, 64, 32, runs=5)
        assert "unfused" in r
        assert "fused" in r
        uf_ms = r["unfused"][0]
        fu_ms = r["fused"][0]
        assert uf_ms > 0
        assert fu_ms > 0

    def test_training_throughput_bench(self):
        from kiri.benchmark import _bench_training_throughput
        r = _bench_training_throughput(runs=2)
        assert "ms_per_step" in r
        assert "samples_per_sec" in r
        assert r["samples_per_sec"] > 0
