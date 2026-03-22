"""
kiri/optim/optimizers.py

Key fix: MLX optimizers take the MODULE object, not a list of params.
The module must be an mlx.nn.Module (which our Module base class is).
"""

import numpy as np
from kiri.backend.detect import BACKEND

if BACKEND == "mlx":
    import mlx.core as mx
    import mlx.optimizers as mlx_optim


class _Optimizer:
    def __init__(self, params_or_model, lr):
        self.lr = lr
        # params_or_model can be a list (CPU) or an mlx.nn.Module (MLX)
        if BACKEND == "mlx":
            from kiri.nn.layers import Module
            if isinstance(params_or_model, Module):
                self._model = params_or_model
            else:
                raise TypeError(
                    "On Apple Silicon, pass your model directly to the optimizer: "
                    "optim.Adam(model, lr=1e-3)"
                )
        else:
            self._params = params_or_model

    def zero_grad(self):
        if BACKEND != "mlx":
            for p in self._params:
                p.grad = np.zeros_like(p.data)

    def step(self, grads=None):
        raise NotImplementedError


class SGD(_Optimizer):
    def __init__(self, params_or_model, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params_or_model, lr)
        self.momentum     = momentum
        self.weight_decay = weight_decay
        if BACKEND == "mlx":
            self._opt = mlx_optim.SGD(learning_rate=lr, momentum=momentum)
        else:
            self._v = [np.zeros_like(p.data) for p in self._params]

    def step(self, grads=None):
        if BACKEND == "mlx":
            self._opt.learning_rate = self.lr
            self._opt.update(self._model, grads)
            mx.eval(self._model.parameters())
        else:
            for i, p in enumerate(self._params):
                g = p.grad + self.weight_decay * p.data
                if self.momentum:
                    self._v[i] = self.momentum * self._v[i] + g
                    g = self._v[i]
                p.data -= self.lr * g


class Adam(_Optimizer):
    def __init__(self, params_or_model, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.0):
        super().__init__(params_or_model, lr)
        self.betas        = betas
        self.eps          = eps
        self.weight_decay = weight_decay
        self.t            = 0
        if BACKEND == "mlx":
            self._opt = mlx_optim.Adam(learning_rate=lr, betas=betas, eps=eps)
        else:
            self._m = [np.zeros_like(p.data) for p in self._params]
            self._v = [np.zeros_like(p.data) for p in self._params]

    def step(self, grads=None):
        if BACKEND == "mlx":
            self._opt.learning_rate = self.lr
            self._opt.update(self._model, grads)
            mx.eval(self._model.parameters())
        else:
            self.t += 1
            b1, b2 = self.betas
            for i, p in enumerate(self._params):
                g = p.grad + self.weight_decay * p.data
                self._m[i] = b1 * self._m[i] + (1-b1) * g
                self._v[i] = b2 * self._v[i] + (1-b2) * g**2
                mh = self._m[i] / (1 - b1**self.t)
                vh = self._v[i] / (1 - b2**self.t)
                p.data -= self.lr * mh / (np.sqrt(vh) + self.eps)


class AdamW(Adam):
    """Adam with decoupled weight decay."""
    def step(self, grads=None):
        if BACKEND == "mlx":
            self._opt.learning_rate = self.lr
            mlx_optim.AdamW(
                learning_rate=self.lr, betas=self.betas,
                eps=self.eps, weight_decay=self.weight_decay
            ).update(self._model, grads)
            mx.eval(self._model.parameters())
        else:
            self.t += 1
            b1, b2 = self.betas
            for i, p in enumerate(self._params):
                self._m[i] = b1 * self._m[i] + (1-b1) * p.grad
                self._v[i] = b2 * self._v[i] + (1-b2) * p.grad**2
                mh = self._m[i] / (1 - b1**self.t)
                vh = self._v[i] / (1 - b2**self.t)
                p.data -= self.lr * mh / (np.sqrt(vh) + self.eps)
                p.data -= self.lr * self.weight_decay * p.data
