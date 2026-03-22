"""kiri/nn/pooling.py"""

from kiri.backend.detect import BACKEND
from kiri.nn.layers import Module

if BACKEND == "mlx":
    import mlx.core as mx


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride      = self.kernel_size if stride is None else (
            (stride, stride) if isinstance(stride, int) else stride)
        self.padding     = padding

    def forward(self, x):
        if BACKEND == "mlx":
            xn  = mx.transpose(x, (0, 2, 3, 1))
            out = mx.max_pool2d(xn, kernel_size=self.kernel_size,
                                stride=self.stride, padding=self.padding)
            return mx.transpose(out, (0, 3, 1, 2))
        from kiri.backend.cpu_ops import maxpool2d_forward
        return maxpool2d_forward(x, self.kernel_size, self.stride, self.padding)

    def __repr__(self):
        return f"MaxPool2d(k={self.kernel_size}, stride={self.stride})"


class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride      = self.kernel_size if stride is None else (
            (stride, stride) if isinstance(stride, int) else stride)
        self.padding     = padding

    def forward(self, x):
        if BACKEND == "mlx":
            xn  = mx.transpose(x, (0, 2, 3, 1))
            out = mx.avg_pool2d(xn, kernel_size=self.kernel_size,
                                stride=self.stride, padding=self.padding)
            return mx.transpose(out, (0, 3, 1, 2))
        import numpy as np
        from kiri.autograd import Tensor
        xd = x.data
        kH, kW = self.kernel_size
        sH, sW = self.stride
        N, C, H, W = xd.shape
        H_out = (H - kH) // sH + 1
        W_out = (W - kW) // sW + 1
        out = np.zeros((N, C, H_out, W_out), dtype=np.float32)
        for i in range(H_out):
            for j in range(W_out):
                out[:, :, i, j] = xd[:, :, i*sH:i*sH+kH, j*sW:j*sW+kW].mean(axis=(2,3))
        result = Tensor(out, (x,), "avgpool2d")
        def _bwd():
            dx = np.zeros_like(xd)
            for i in range(H_out):
                for j in range(W_out):
                    dx[:, :, i*sH:i*sH+kH, j*sW:j*sW+kW] += \
                        result.grad[:, :, i:i+1, j:j+1] / (kH * kW)
            x.grad += dx
        result._backward = _bwd
        return result
