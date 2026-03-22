"""
kiri/nn/layers.py

All layer definitions. Two rules:
  1. Every __init__ calls super().__init__()
  2. No attribute named 'training' — MLX reserves it.
     We use '_is_training' instead.
"""

import numpy as np
from kiri.backend.detect import BACKEND
from kiri.autograd import Tensor, _wrap

if BACKEND == "mlx":
    import mlx.core as mx
    import mlx.nn as mlx_nn
    _Base = mlx_nn.Module
else:
    _Base = object


# ─── Module ───────────────────────────────────────────────────────────────────

class Module(_Base):
    """Base class for all Kiri layers and models."""

    def __init__(self):
        if BACKEND == "mlx":
            super().__init__()

    def parameters(self):
        """Flat list of all trainable parameters."""
        if BACKEND == "mlx":
            params = []
            def _collect(obj):
                if isinstance(obj, mx.array):
                    params.append(obj)
                elif isinstance(obj, mlx_nn.Module):
                    for v in vars(obj).values():
                        _collect(v)
                elif isinstance(obj, (list, tuple)):
                    for item in obj:
                        _collect(item)
            for v in vars(self).values():
                _collect(v)
            return params
        else:
            params = []
            def _collect(obj):
                if isinstance(obj, Tensor):
                    params.append(obj)
                elif isinstance(obj, Module):
                    params.extend(obj.parameters())
                elif isinstance(obj, (list, tuple)):
                    for item in obj:
                        _collect(item)
            for v in vars(self).values():
                _collect(v)
            return params

    def zero_grad(self):
        if BACKEND != "mlx":
            for p in self.parameters():
                p.grad = np.zeros_like(p.data)

    def train(self):
        """Set all layers to training mode."""
        self._is_training = True
        for v in vars(self).values():
            if isinstance(v, Module):
                v.train()

    def eval(self):
        """Set all layers to inference mode (disables Dropout etc.)."""
        self._is_training = False
        for v in vars(self).values():
            if isinstance(v, Module):
                v.eval()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# ─── Linear ───────────────────────────────────────────────────────────────────

class Linear(Module):
    """y = x @ W.T + b"""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        k = np.sqrt(1.0 / in_features)
        if BACKEND == "mlx":
            self.weight  = mx.random.uniform(low=-k, high=k,
                           shape=(out_features, in_features))
            self.b       = mx.zeros((out_features,)) if bias else None
        else:
            self.weight  = Tensor(np.random.uniform(
                           -k, k, (out_features, in_features)).astype(np.float32))
            self.b       = Tensor(np.zeros(out_features, dtype=np.float32)) if bias else None

    def forward(self, x):
        if BACKEND == "mlx":
            out = x @ self.weight.T
            return out + self.b if self.b is not None else out
        else:
            out = x.matmul(self.weight.T)
            return out + self.b if self.b is not None else out

    def parameters(self):
        return [self.weight] + ([self.b] if self.b is not None else [])

    def __repr__(self):
        return f"Linear({self.in_features} → {self.out_features})"


# ─── Conv2d ───────────────────────────────────────────────────────────────────

class Conv2d(Module):
    """2D convolution. Input: (N, C_in, H, W)"""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride       = stride
        self.padding      = padding
        kH, kW            = self.kernel_size
        k                 = np.sqrt(1.0 / (in_channels * kH * kW))
        if BACKEND == "mlx":
            self.weight = mx.random.uniform(low=-k, high=k,
                          shape=(out_channels, in_channels, kH, kW))
            self.b      = mx.zeros((out_channels,)) if bias else None
        else:
            self.weight = Tensor(np.random.uniform(
                          -k, k, (out_channels, in_channels, kH, kW)).astype(np.float32))
            self.b      = Tensor(np.zeros(out_channels, dtype=np.float32)) if bias else None

    def forward(self, x):
        if BACKEND == "mlx":
            # MLX conv2d expects NHWC
            xn = mx.transpose(x, (0, 2, 3, 1))
            wn = mx.transpose(self.weight, (0, 2, 3, 1))
            out = mx.conv2d(xn, wn, stride=self.stride, padding=self.padding)
            out = mx.transpose(out, (0, 3, 1, 2))
            if self.b is not None:
                out = out + self.b.reshape(1, -1, 1, 1)
            return out
        else:
            from kiri.backend.cpu_ops import conv2d_forward
            return conv2d_forward(x, self.weight, self.b, self.stride, self.padding)

    def parameters(self):
        return [self.weight] + ([self.b] if self.b is not None else [])

    def __repr__(self):
        return f"Conv2d({self.in_channels}, {self.out_channels}, k={self.kernel_size})"


# ─── BatchNorm1d ──────────────────────────────────────────────────────────────

class BatchNorm1d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps          = eps
        self.momentum     = momentum
        self._is_training = True
        if BACKEND == "mlx":
            self.weight = mx.ones((num_features,))
            self.b      = mx.zeros((num_features,))
        else:
            self.weight       = Tensor(np.ones(num_features, dtype=np.float32))
            self.b            = Tensor(np.zeros(num_features, dtype=np.float32))
            self.running_mean = np.zeros(num_features, dtype=np.float32)
            self.running_var  = np.ones(num_features, dtype=np.float32)

    def forward(self, x):
        if BACKEND == "mlx":
            mean  = x.mean(axis=0)
            var   = ((x - mean) ** 2).mean(axis=0)
            x_hat = (x - mean) / mx.sqrt(var + self.eps)
            return self.weight * x_hat + self.b
        else:
            if self._is_training:
                mean = x.data.mean(axis=0)
                var  = x.data.var(axis=0)
                self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*mean
                self.running_var  = (1-self.momentum)*self.running_var  + self.momentum*var
            else:
                mean, var = self.running_mean, self.running_var
            x_hat = Tensor((x.data - mean) / np.sqrt(var + self.eps))
            return self.weight * x_hat + self.b

    def parameters(self):
        return [self.weight, self.b]


# ─── Dropout ──────────────────────────────────────────────────────────────────

class Dropout(Module):
    # NOTE: do NOT use 'training' — MLX.Module reserves that property.
    def __init__(self, p=0.5):
        super().__init__()
        self.p            = p
        self._is_training = True

    def forward(self, x):
        if not self._is_training or self.p == 0.0:
            return x
        if BACKEND == "mlx":
            # shape must be a list, not a tuple, for mx.random.uniform
            mask = mx.random.uniform(shape=list(x.shape)) > self.p
            return x * mask / (1.0 - self.p)
        else:
            mask = (np.random.rand(*x.data.shape) > self.p).astype(np.float32)
            return x * Tensor(mask / (1.0 - self.p), requires_grad=False)


# ─── Flatten ─────────────────────────────────────────────────────────────────

class Flatten(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if BACKEND == "mlx":
            return x.reshape(x.shape[0], -1)
        return x.reshape(x.shape[0], -1)


# ─── Sequential ──────────────────────────────────────────────────────────────

class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def train(self):
        self._is_training = True
        for l in self.layers: l.train()

    def eval(self):
        self._is_training = False
        for l in self.layers: l.eval()

    def __repr__(self):
        inner = "\n  ".join(repr(l) for l in self.layers)
        return f"Sequential(\n  {inner}\n)"
