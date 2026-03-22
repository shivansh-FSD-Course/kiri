"""kiri/nn/recurrent.py"""

import numpy as np
from kiri.backend.detect import BACKEND
from kiri.autograd import Tensor
from kiri.nn.layers import Module

if BACKEND == "mlx":
    import mlx.core as mx


class Embedding(Module):
    """Token ID → dense vector lookup table."""

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim  = embedding_dim
        self.padding_idx    = padding_idx
        scale = np.sqrt(2.0 / embedding_dim)
        if BACKEND == "mlx":
            self.weight = mx.random.normal(shape=(num_embeddings, embedding_dim)) * scale
        else:
            w = (np.random.randn(num_embeddings, embedding_dim) * scale).astype(np.float32)
            if padding_idx is not None:
                w[padding_idx] = 0.0
            self.weight = Tensor(w)

    def forward(self, idx):
        """idx: np.ndarray (N, seq_len) int32"""
        if BACKEND == "mlx":
            return self.weight[mx.array(idx)]
        idx_data = idx if isinstance(idx, np.ndarray) else np.asarray(idx, dtype=int)
        out = Tensor(self.weight.data[idx_data], (self.weight,), "embedding")
        def _bwd():
            np.add.at(self.weight.grad, idx_data.ravel(), out.grad.reshape(-1, self.embedding_dim))
            if self.padding_idx is not None:
                self.weight.grad[self.padding_idx] = 0.0
        out._backward = _bwd
        return out

    def parameters(self): return [self.weight]
    def __repr__(self): return f"Embedding({self.num_embeddings}, {self.embedding_dim})"


class LSTM(Module):
    """Single-layer LSTM. Input: (N, T, input_size) → (N, T, hidden_size)"""

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        k = np.sqrt(1.0 / hidden_size)
        def _w(shape): return np.random.uniform(-k, k, shape).astype(np.float32)
        if BACKEND == "mlx":
            self.W_ih = mx.array(_w((4*hidden_size, input_size)))
            self.W_hh = mx.array(_w((4*hidden_size, hidden_size)))
            self.b_ih = mx.zeros((4*hidden_size,)) if bias else None
            self.b_hh = mx.zeros((4*hidden_size,)) if bias else None
        else:
            self.W_ih = Tensor(_w((4*hidden_size, input_size)))
            self.W_hh = Tensor(_w((4*hidden_size, hidden_size)))
            self.b_ih = Tensor(np.zeros(4*hidden_size, dtype=np.float32)) if bias else None
            self.b_hh = Tensor(np.zeros(4*hidden_size, dtype=np.float32)) if bias else None

    def forward(self, x, hc0=None):
        if BACKEND == "mlx":
            return self._forward_mlx(x, hc0)
        return self._forward_cpu(x, hc0)

    def _forward_cpu(self, x, hc0):
        xd = x.data if isinstance(x, Tensor) else np.asarray(x, dtype=np.float32)
        N, T, _ = xd.shape
        H = self.hidden_size
        h = np.zeros((N, H), np.float32) if hc0 is None else hc0[0].data
        c = np.zeros((N, H), np.float32) if hc0 is None else hc0[1].data
        outputs = []
        Wih = self.W_ih.data; Whh = self.W_hh.data
        bih = self.b_ih.data if self.b_ih else 0
        bhh = self.b_hh.data if self.b_hh else 0
        for t in range(T):
            g = xd[:, t, :] @ Wih.T + h @ Whh.T + bih + bhh
            i_g = _sig(g[:, :H]);   f_g = _sig(g[:, H:2*H])
            g_g = np.tanh(g[:, 2*H:3*H]); o_g = _sig(g[:, 3*H:])
            c   = f_g * c + i_g * g_g
            h   = o_g * np.tanh(c)
            outputs.append(h[:, np.newaxis, :])
        out_all = np.concatenate(outputs, axis=1)
        return (Tensor(out_all, requires_grad=False),
                (Tensor(h, requires_grad=False), Tensor(c, requires_grad=False)))

    def _forward_mlx(self, x, hc0):
        if isinstance(x, np.ndarray): x = mx.array(x)
        N, T, _ = x.shape
        H = self.hidden_size
        h = mx.zeros((N, H)) if hc0 is None else hc0[0]
        c = mx.zeros((N, H)) if hc0 is None else hc0[1]
        outputs = []
        for t in range(T):
            g = x[:, t, :] @ self.W_ih.T + h @ self.W_hh.T
            if self.b_ih is not None: g = g + self.b_ih + self.b_hh
            i_g = mx.sigmoid(g[:, :H]);  f_g = mx.sigmoid(g[:, H:2*H])
            g_g = mx.tanh(g[:, 2*H:3*H]); o_g = mx.sigmoid(g[:, 3*H:])
            c = f_g * c + i_g * g_g
            h = o_g * mx.tanh(c)
            outputs.append(h[:, None, :])
        return mx.concatenate(outputs, axis=1), (h, c)

    def parameters(self):
        p = [self.W_ih, self.W_hh]
        if self.b_ih is not None: p += [self.b_ih, self.b_hh]
        return p

    def __repr__(self): return f"LSTM({self.input_size} → {self.hidden_size})"


def _sig(x): return 1 / (1 + np.exp(-x.clip(-500, 500)))
