"""
kiri/tensor.py

The Tensor class: Kiri's core data structure.
On Apple Silicon → wraps MLX arrays (Metal GPU acceleration).
On CPU          → wraps NumPy arrays with a lightweight autograd engine.
"""

import numpy as np
from kiri.backend.detect import BACKEND

if BACKEND == "mlx":
    import mlx.core as mx


# ─── tiny autograd engine for the CPU backend ────────────────────────────────

class _CPUTensor:
    """
    Minimal scalar/array autograd node for the CPU backend.
    Inspired by micrograd — keeps it readable and hackable.
    """
    def __init__(self, data, _children=(), _op="", requires_grad=True):
        self.data = np.asarray(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # ── arithmetic ──────────────────────────────────────────────────────────

    def __add__(self, other):
        other = other if isinstance(other, _CPUTensor) else _CPUTensor(other)
        out = _CPUTensor(self.data + other.data, (self, other), "+")

        def _backward():
            g = out.grad
            self.grad  += _unbroadcast(g, self.data.shape)
            other.grad += _unbroadcast(g, other.data.shape)
        out._backward = _backward
        return out

    def __radd__(self, other): return self + other

    def __mul__(self, other):
        other = other if isinstance(other, _CPUTensor) else _CPUTensor(other)
        out = _CPUTensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad  += _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += _unbroadcast(self.data  * out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __rmul__(self, other): return self * other

    def __neg__(self):
        return self * _CPUTensor(np.full(self.data.shape, -1.0))

    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return _CPUTensor(other) + (-self)

    def __truediv__(self, other):
        other = other if isinstance(other, _CPUTensor) else _CPUTensor(other)
        return self * other.pow(-1)

    def pow(self, exp):
        out = _CPUTensor(self.data ** exp, (self,), f"**{exp}")

        def _backward():
            self.grad += _unbroadcast(exp * (self.data ** (exp - 1)) * out.grad,
                                      self.data.shape)
        out._backward = _backward
        return out

    def matmul(self, other):
        other = other if isinstance(other, _CPUTensor) else _CPUTensor(other)
        out = _CPUTensor(self.data @ other.data, (self, other), "matmul")

        def _backward():
            self.grad  += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    # ── activations ─────────────────────────────────────────────────────────

    def relu(self):
        out = _CPUTensor(np.maximum(0, self.data), (self,), "relu")

        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1 / (1 + np.exp(-self.data.clip(-500, 500)))
        out = _CPUTensor(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = _CPUTensor(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        e = np.exp(self.data)
        out = _CPUTensor(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = _CPUTensor(np.log(self.data + 1e-9), (self,), "log")

        def _backward():
            self.grad += (1 / (self.data + 1e-9)) * out.grad
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = _CPUTensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            self.grad += np.broadcast_to(
                out.grad if keepdims else np.expand_dims(out.grad, axis=axis)
                    if axis is not None else out.grad,
                self.data.shape
            ).copy()
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # ── reshaping ────────────────────────────────────────────────────────────

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        out = _CPUTensor(self.data.reshape(shape), (self,), "reshape")
        orig = self.data.shape

        def _backward():
            self.grad += out.grad.reshape(orig)
        out._backward = _backward
        return out

    def transpose(self, axes=None):
        out = _CPUTensor(self.data.transpose(axes), (self,), "T")
        if axes is None:
            inv = None
        else:
            inv = np.argsort(axes)

        def _backward():
            self.grad += out.grad.transpose(inv)
        out._backward = _backward
        return out

    @property
    def T(self):
        return self.transpose()

    # ── conv2d (naive but correct — useful for small kernels) ───────────────

    def conv2d(self, weight, bias=None, stride=1, padding=0):
        from kiri.backend.cpu_conv import conv2d_forward_backward
        return conv2d_forward_backward(self, weight, bias, stride, padding)

    # ── backprop ─────────────────────────────────────────────────────────────

    def backward(self):
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    # ── properties / helpers ─────────────────────────────────────────────────

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def numpy(self):
        return self.data

    def item(self):
        return float(self.data.flat[0])

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    def __repr__(self):
        return f"Tensor({self.data}, backend=cpu)"

    def __getitem__(self, idx):
        out = _CPUTensor(self.data[idx], (self,), "slice")

        def _backward():
            g = np.zeros_like(self.data)
            g[idx] += out.grad
            self.grad += g
        out._backward = _backward
        return out


def _unbroadcast(grad, shape):
    """Sum grad back to original shape after broadcasting."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, (gs, ss) in enumerate(zip(grad.shape, shape)):
        if ss == 1 and gs != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


# ─── public Tensor factory ────────────────────────────────────────────────────

class Tensor:
    """
    Public-facing Tensor.
    Dispatches to MLX (Apple Silicon) or _CPUTensor (everything else).
    """

    def __new__(cls, data, requires_grad=True, dtype=None):
        if BACKEND == "mlx":
            return _make_mlx_tensor(data, dtype)
        else:
            arr = np.asarray(data, dtype=np.float32)
            t = _CPUTensor(arr, requires_grad=requires_grad)
            return t

    @staticmethod
    def zeros(*shape, requires_grad=False):
        if BACKEND == "mlx":
            return mx.zeros(shape)
        return _CPUTensor(np.zeros(shape, dtype=np.float32),
                          requires_grad=requires_grad)

    @staticmethod
    def ones(*shape, requires_grad=False):
        if BACKEND == "mlx":
            return mx.ones(shape)
        return _CPUTensor(np.ones(shape, dtype=np.float32),
                          requires_grad=requires_grad)

    @staticmethod
    def randn(*shape, requires_grad=True):
        if BACKEND == "mlx":
            import mlx.core as mx
            return mx.random.normal(shape)
        return _CPUTensor(np.random.randn(*shape).astype(np.float32),
                          requires_grad=requires_grad)

    @staticmethod
    def from_numpy(arr):
        if BACKEND == "mlx":
            return mx.array(arr.astype(np.float32))
        return _CPUTensor(arr.astype(np.float32))


def _make_mlx_tensor(data, dtype=None):
    """Converts data to an MLX array."""
    if isinstance(data, np.ndarray):
        return mx.array(data.astype(np.float32))
    elif isinstance(data, (list, tuple)):
        return mx.array(np.array(data, dtype=np.float32))
    elif isinstance(data, (int, float)):
        return mx.array(np.float32(data))
    return mx.array(data)
