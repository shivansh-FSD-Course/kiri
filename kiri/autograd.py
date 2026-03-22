"""
kiri/autograd.py

Lightweight autograd engine for the CPU (NumPy) backend.
Supports the operations needed for dense nets, CNNs, RNNs, and embeddings.
"""

import numpy as np


def _unbroadcast(grad, shape):
    """Sum grad back to original shape after broadcasting."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, (gs, ss) in enumerate(zip(grad.shape, shape)):
        if ss == 1 and gs != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    """
    CPU tensor with automatic differentiation.
    Wraps a numpy float32 array and tracks the computation graph.
    """

    __slots__ = ("data", "grad", "requires_grad", "_backward", "_prev", "_op")

    def __init__(self, data, _children=(), _op="", requires_grad=True):
        self.data          = np.asarray(data, dtype=np.float32)
        self.grad          = np.zeros_like(self.data)
        self.requires_grad = requires_grad
        self._backward     = lambda: None
        self._prev         = set(_children)
        self._op           = _op

    # ── arithmetic ────────────────────────────────────────────────────────

    def __add__(self, other):
        other = _wrap(other)
        out   = Tensor(self.data + other.data, (self, other), "+")
        def _bwd():
            self.grad  += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)
        out._backward = _bwd
        return out

    def __radd__(self, other): return self + other

    def __mul__(self, other):
        other = _wrap(other)
        out   = Tensor(self.data * other.data, (self, other), "*")
        def _bwd():
            self.grad  += _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += _unbroadcast(self.data  * out.grad, other.data.shape)
        out._backward = _bwd
        return out

    def __rmul__(self, other): return self * other
    def __neg__(self):         return self * Tensor(np.full(self.data.shape, -1.0))
    def __sub__(self, other):  return self + (-_wrap(other))
    def __rsub__(self, other): return _wrap(other) + (-self)

    def __truediv__(self, other):
        other = _wrap(other)
        return self * other.pow(-1)

    def pow(self, exp):
        out = Tensor(self.data ** exp, (self,), f"**{exp}")
        def _bwd():
            self.grad += _unbroadcast(
                exp * (self.data ** (exp - 1)) * out.grad, self.data.shape
            )
        out._backward = _bwd
        return out

    def matmul(self, other):
        other = _wrap(other)
        out   = Tensor(self.data @ other.data, (self, other), "matmul")
        def _bwd():
            self.grad  += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _bwd
        return out

    # ── activations ───────────────────────────────────────────────────────

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), "relu")
        def _bwd(): self.grad += (self.data > 0) * out.grad
        out._backward = _bwd
        return out

    def sigmoid(self):
        s   = 1 / (1 + np.exp(-self.data.clip(-500, 500)))
        out = Tensor(s, (self,), "sigmoid")
        def _bwd(): self.grad += s * (1 - s) * out.grad
        out._backward = _bwd
        return out

    def tanh(self):
        t   = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")
        def _bwd(): self.grad += (1 - t ** 2) * out.grad
        out._backward = _bwd
        return out

    def exp(self):
        e   = np.exp(self.data)
        out = Tensor(e, (self,), "exp")
        def _bwd(): self.grad += e * out.grad
        out._backward = _bwd
        return out

    def log(self):
        out = Tensor(np.log(self.data + 1e-9), (self,), "log")
        def _bwd(): self.grad += (1 / (self.data + 1e-9)) * out.grad
        out._backward = _bwd
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")
        def _bwd():
            g = out.grad
            if not keepdims and axis is not None:
                g = np.expand_dims(g, axis=axis)
            self.grad += np.broadcast_to(g, self.data.shape).copy()
        out._backward = _bwd
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        orig = self.data.shape
        out  = Tensor(self.data.reshape(shape), (self,), "reshape")
        def _bwd(): self.grad += out.grad.reshape(orig)
        out._backward = _bwd
        return out

    @property
    def T(self):
        out = Tensor(self.data.T, (self,), "T")
        def _bwd(): self.grad += out.grad.T
        out._backward = _bwd
        return out

    # ── backprop ──────────────────────────────────────────────────────────

    def backward(self):
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for c in v._prev:
                    build(c)
                topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    # ── utils ─────────────────────────────────────────────────────────────

    @property
    def shape(self): return self.data.shape

    @property
    def ndim(self): return self.data.ndim

    def numpy(self):   return self.data
    def item(self):    return float(self.data.flat[0])
    def zero_grad(self): self.grad = np.zeros_like(self.data)

    def __getitem__(self, idx):
        out = Tensor(self.data[idx], (self,), "slice")
        def _bwd():
            g = np.zeros_like(self.data)
            np.add.at(g, idx, out.grad)
            self.grad += g
        out._backward = _bwd
        return out

    def __repr__(self):
        return f"Tensor(shape={self.shape}, op='{self._op}')"


def _wrap(x):
    if isinstance(x, Tensor): return x
    return Tensor(np.asarray(x, dtype=np.float32), requires_grad=False)
