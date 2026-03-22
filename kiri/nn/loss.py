"""kiri/nn/loss.py"""

import numpy as np
from kiri.backend.detect import BACKEND
from kiri.autograd import Tensor, _wrap

if BACKEND == "mlx":
    import mlx.core as mx
    import mlx.nn as mlx_nn


def cross_entropy(logits, targets):
    """logits: (N, C) raw scores. targets: (N,) int class indices."""
    if BACKEND == "mlx":
        return mlx_nn.losses.cross_entropy(logits, targets).mean()
    N = logits.data.shape[0]
    shifted = logits.data - logits.data.max(axis=1, keepdims=True)
    exp_x   = np.exp(shifted)
    probs   = exp_x / exp_x.sum(axis=1, keepdims=True)
    t       = (targets.data if isinstance(targets, Tensor) else targets).astype(int).ravel()
    loss_val = -np.log(probs[np.arange(N), t] + 1e-9).mean()
    loss     = Tensor(np.float32(loss_val), (logits,), "cross_entropy")
    def _bwd():
        dl = probs.copy()
        dl[np.arange(N), t] -= 1
        logits.grad += dl / N * loss.grad
    loss._backward = _bwd
    return loss


def mse_loss(pred, target):
    """Mean squared error."""
    if BACKEND == "mlx":
        return mx.mean((pred - target) ** 2)
    target = _wrap(target) if not isinstance(target, Tensor) else target
    diff   = pred - target
    return (diff * diff).mean()


def binary_cross_entropy(pred, target):
    """pred: (N,) sigmoid outputs. target: (N,) 0/1 labels."""
    if BACKEND == "mlx":
        return mlx_nn.losses.binary_cross_entropy(pred, target).mean()
    target = _wrap(target) if not isinstance(target, Tensor) else target
    eps    = 1e-9
    p      = pred.data.clip(eps, 1-eps)
    t      = target.data
    val    = -(t * np.log(p) + (1-t) * np.log(1-p)).mean()
    loss   = Tensor(np.float32(val), (pred,), "bce")
    def _bwd():
        pred.grad += (-(t/(p+eps) - (1-t)/(1-p+eps)) / len(p)) * loss.grad
    loss._backward = _bwd
    return loss
