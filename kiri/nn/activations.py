"""kiri/nn/activations.py"""

import numpy as np
from kiri.backend.detect import BACKEND
from kiri.autograd import Tensor, _wrap
from kiri.nn.layers import Module

if BACKEND == "mlx":
    import mlx.core as mx
    import mlx.nn as mlx_nn


class ReLU(Module):
    def __init__(self): super().__init__()
    def forward(self, x):
        return mx.maximum(x, 0) if BACKEND == "mlx" else x.relu()


class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        if BACKEND == "mlx":
            return mx.where(x >= 0, x, self.negative_slope * x)
        out = Tensor(np.where(x.data >= 0, x.data, self.negative_slope * x.data), (x,), "leaky_relu")
        def _bwd(): x.grad += np.where(x.data >= 0, 1.0, self.negative_slope) * out.grad
        out._backward = _bwd
        return out


class Sigmoid(Module):
    def __init__(self): super().__init__()
    def forward(self, x):
        return mx.sigmoid(x) if BACKEND == "mlx" else x.sigmoid()


class Tanh(Module):
    def __init__(self): super().__init__()
    def forward(self, x):
        return mx.tanh(x) if BACKEND == "mlx" else x.tanh()


class Softmax(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        if BACKEND == "mlx":
            return mx.softmax(x, axis=self.dim)
        data    = x.data
        shifted = data - data.max(axis=self.dim, keepdims=True)
        exp_x   = np.exp(shifted)
        s       = exp_x / exp_x.sum(axis=self.dim, keepdims=True)
        out     = Tensor(s, (x,), "softmax")
        def _bwd():
            x.grad += s * (out.grad - (out.grad * s).sum(axis=self.dim, keepdims=True))
        out._backward = _bwd
        return out


class GELU(Module):
    def __init__(self): super().__init__()
    def forward(self, x):
        if BACKEND == "mlx":
            return mlx_nn.gelu(x)
        import math
        d  = x.data
        t  = np.tanh(np.sqrt(2/math.pi) * (d + 0.044715 * d**3))
        v  = 0.5 * d * (1 + t)
        out = Tensor(v, (x,), "gelu")
        def _bwd():
            c  = np.sqrt(2/math.pi)
            dt = c * (1 + 3*0.044715*d**2) * (1 - t**2)
            x.grad += out.grad * 0.5 * ((1 + t) + d * dt)
        out._backward = _bwd
        return out
