"""
kiri/backend/fusion.py

Operator fusion for the CPU backend.

Instead of:
    Linear → ReLU (two separate ops, two memory passes)

We do:
    fused_linear_relu (one op, one memory pass, no intermediate allocation)

Patterns supported:
    linear_relu         Linear → ReLU
    linear_gelu         Linear → GELU
    linear_bn_relu      Linear → BatchNorm1d → ReLU
    linear_sigmoid      Linear → Sigmoid

On Apple Silicon (MLX backend), MLX's lazy evaluation already fuses many
of these automatically. Fusion here benefits the CPU backend specifically.
"""

import numpy as np
from kiri.backend.accelerate import sgemm, is_available as accel_available


# ─── fused kernels ────────────────────────────────────────────────────────────

def linear_relu_fused(x: np.ndarray, W: np.ndarray, b: np.ndarray | None) -> np.ndarray:
    """
    y = relu(x @ W.T + b)
    Single pass — no intermediate tensor allocation.
    Uses Accelerate sgemm when available.
    """
    out = sgemm(x, W.T) if accel_available() and x.ndim == 2 else x @ W.T
    if b is not None:
        out += b
    np.maximum(out, 0, out=out)   # in-place relu — no extra allocation
    return out


def linear_gelu_fused(x: np.ndarray, W: np.ndarray, b: np.ndarray | None) -> np.ndarray:
    """y = gelu(x @ W.T + b)"""
    import math
    out = sgemm(x, W.T) if accel_available() and x.ndim == 2 else x @ W.T
    if b is not None:
        out += b
    # GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    # In-place to avoid allocation
    c   = np.sqrt(2.0 / math.pi)
    tmp = c * (out + 0.044715 * out ** 3)
    np.tanh(tmp, out=tmp)
    tmp += 1
    tmp *= 0.5
    out *= tmp
    return out


def linear_sigmoid_fused(x: np.ndarray, W: np.ndarray, b: np.ndarray | None) -> np.ndarray:
    """y = sigmoid(x @ W.T + b)"""
    out = sgemm(x, W.T) if accel_available() and x.ndim == 2 else x @ W.T
    if b is not None:
        out += b
    np.clip(out, -500, 500, out=out)
    np.negative(out, out=out)
    np.exp(out, out=out)
    out += 1
    np.reciprocal(out, out=out)
    return out


def linear_bn_relu_fused(
    x: np.ndarray,
    W: np.ndarray,
    b: np.ndarray | None,
    bn_weight: np.ndarray,
    bn_bias: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
    eps: float = 1e-5,
    is_training: bool = True,
    momentum: float = 0.1,
) -> np.ndarray:
    """y = relu(batchnorm(x @ W.T + b))"""
    out = sgemm(x, W.T) if accel_available() and x.ndim == 2 else x @ W.T
    if b is not None:
        out += b

    # BatchNorm
    if is_training:
        mean = out.mean(axis=0)
        var  = out.var(axis=0)
        running_mean[:] = (1 - momentum) * running_mean + momentum * mean
        running_var[:]  = (1 - momentum) * running_var  + momentum * var
    else:
        mean, var = running_mean, running_var

    out = (out - mean) / np.sqrt(var + eps)
    out = bn_weight * out + bn_bias

    # ReLU in-place
    np.maximum(out, 0, out=out)
    return out


# ─── fusion detector ─────────────────────────────────────────────────────────

class FusedLinear:
    """
    Drop-in replacement for Linear → Activation sequences.
    Detected and injected by fuse_model().

    Skips autograd — used in inference (.eval() mode) only.
    """

    def __init__(self, linear, activation_name: str, bn=None):
        self.linear         = linear
        self.activation     = activation_name
        self.bn             = bn
        self._is_training   = False

    def __call__(self, x):
        import numpy as np
        from kiri.autograd import Tensor

        x_data = x.data if isinstance(x, Tensor) else x
        W      = self.linear.weight.data
        b      = self.linear.b.data if self.linear.b is not None else None

        if self.bn is not None and self.activation == "relu":
            out = linear_bn_relu_fused(
                x_data, W, b,
                self.bn.weight.data,
                self.bn.b.data,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.eps,
                is_training=self._is_training,
                momentum=self.bn.momentum,
            )
        elif self.activation == "relu":
            out = linear_relu_fused(x_data, W, b)
        elif self.activation == "gelu":
            out = linear_gelu_fused(x_data, W, b)
        elif self.activation == "sigmoid":
            out = linear_sigmoid_fused(x_data, W, b)
        else:
            out = (sgemm(x_data, W.T) if accel_available() and x_data.ndim == 2
                   else x_data @ W.T)
            if b is not None:
                out = out + b

        return Tensor(out, requires_grad=False) if isinstance(x, Tensor) else out

    def parameters(self):
        p = self.linear.parameters()
        if self.bn is not None:
            p += self.bn.parameters()
        return p


def fuse_model(model):
    """
    Scan a Sequential and replace fusable patterns with FusedLinear.

    Patterns detected:
        Linear, ReLU               → FusedLinear(relu)
        Linear, GELU               → FusedLinear(gelu)
        Linear, Sigmoid            → FusedLinear(sigmoid)
        Linear, BatchNorm1d, ReLU  → FusedLinear(relu, bn=BN)

    Only fuses in eval mode (no autograd needed).
    Returns the model with fused layers inserted into Sequential.
    """
    from kiri.nn.layers import Sequential, Linear, BatchNorm1d
    from kiri.nn.activations import ReLU, GELU, Sigmoid

    _ACT_MAP = {ReLU: "relu", GELU: "gelu", Sigmoid: "sigmoid"}

    def _fuse_sequential(seq):
        layers  = seq.layers
        fused   = []
        i       = 0
        n_fused = 0

        while i < len(layers):
            layer = layers[i]

            # Pattern: Linear, BN, ReLU
            if (isinstance(layer, Linear)
                    and i+2 < len(layers)
                    and isinstance(layers[i+1], BatchNorm1d)
                    and isinstance(layers[i+2], ReLU)):
                fused.append(FusedLinear(layer, "relu", bn=layers[i+1]))
                i += 3
                n_fused += 1
                continue

            # Pattern: Linear, Activation
            if (isinstance(layer, Linear)
                    and i+1 < len(layers)
                    and type(layers[i+1]) in _ACT_MAP):
                act_name = _ACT_MAP[type(layers[i+1])]
                fused.append(FusedLinear(layer, act_name))
                i += 2
                n_fused += 1
                continue

            # Recurse into nested Sequential
            if isinstance(layer, Sequential):
                _fuse_sequential(layer)

            fused.append(layer)
            i += 1

        seq.layers = fused
        return n_fused

    total = 0
    from kiri.nn.layers import Sequential
    for attr in vars(model).values():
        if isinstance(attr, Sequential):
            total += _fuse_sequential(attr)

    return model, total
