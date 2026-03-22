"""
kiri/backend/cpu_ops.py
Numpy implementations of conv2d and pooling for the CPU backend.
"""

import numpy as np
from kiri.autograd import Tensor


def conv2d_forward(x_t, w_t, b_t, stride, padding):
    x = x_t.data
    w = w_t.data
    N, C_in, H, W   = x.shape
    C_out, _, kH, kW = w.shape

    if padding > 0:
        x = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)))

    H_out = (x.shape[2] - kH) // stride + 1
    W_out = (x.shape[3] - kW) // stride + 1
    out   = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)
    w_f   = w.reshape(C_out, -1)

    for n in range(N):
        col = _im2col(x[n:n+1], kH, kW, stride, H_out, W_out)[0]
        out[n] = (w_f @ col).reshape(C_out, H_out, W_out)

    if b_t is not None:
        out += b_t.data.reshape(1, C_out, 1, 1)

    parents = (x_t, w_t) + ((b_t,) if b_t is not None else ())
    result  = Tensor(out, parents, "conv2d")

    def _backward():
        dout  = result.grad
        x_pad = x
        dxp   = np.zeros_like(x_pad)
        dw    = np.zeros_like(w)

        if b_t is not None:
            b_t.grad += dout.sum(axis=(0, 2, 3))

        for n in range(N):
            col_n  = _im2col(x_pad[n:n+1], kH, kW, stride, H_out, W_out)[0]
            dout_n = dout[n].reshape(C_out, -1)
            dw    += (dout_n @ col_n.T).reshape(w.shape)
            dcol   = w_f.T @ dout_n
            dxp[n:n+1] += _col2im(
                dcol[np.newaxis], x_pad.shape[2:], kH, kW, stride, H_out, W_out
            )

        if padding > 0:
            x_t.grad += dxp[:, :, padding:-padding, padding:-padding]
        else:
            x_t.grad += dxp
        w_t.grad += dw

    result._backward = _backward
    return result


def _im2col(x_pad, kH, kW, stride, H_out, W_out):
    N, C, _, _ = x_pad.shape
    col = np.zeros((N, C * kH * kW, H_out * W_out), dtype=np.float32)
    for i in range(kH):
        for j in range(kW):
            s = (i * kW + j) * C
            col[:, s:s+C, :] = x_pad[
                :, :,
                i:i+H_out*stride:stride,
                j:j+W_out*stride:stride
            ].reshape(N, C, -1)
    return col


def _col2im(col, input_shape, kH, kW, stride, H_out, W_out):
    H, W = input_shape
    N    = col.shape[0]
    C    = col.shape[1] // (kH * kW)
    out  = np.zeros((N, C, H, W), dtype=np.float32)
    for i in range(kH):
        for j in range(kW):
            s = (i * kW + j) * C
            out[:, :,
                i:i+H_out*stride:stride,
                j:j+W_out*stride:stride] += col[:, s:s+C, :].reshape(N, C, H_out, W_out)
    return out


def maxpool2d_forward(x_t, kernel_size, stride, padding):
    x         = x_t.data
    kH, kW    = kernel_size
    sH, sW    = stride
    N, C, H, W = x.shape

    if padding > 0:
        x = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)),
                   constant_values=-np.inf)

    H_out = (x.shape[2] - kH) // sH + 1
    W_out = (x.shape[3] - kW) // sW + 1
    out   = np.zeros((N, C, H_out, W_out), dtype=np.float32)

    for i in range(H_out):
        for j in range(W_out):
            out[:, :, i, j] = x[:, :, i*sH:i*sH+kH, j*sW:j*sW+kW].max(axis=(2, 3))

    result = Tensor(out, (x_t,), "maxpool2d")

    def _backward():
        dx = np.zeros_like(x)
        for i in range(H_out):
            for j in range(W_out):
                patch = x[:, :, i*sH:i*sH+kH, j*sW:j*sW+kW]
                idx   = patch.reshape(N, C, -1).argmax(axis=2)
                ri    = idx // kW + i * sH
                ci    = idx %  kW + j * sW
                for n in range(N):
                    for c in range(C):
                        dx[n, c, ri[n,c], ci[n,c]] += result.grad[n, c, i, j]
        if padding > 0:
            x_t.grad += dx[:, :, padding:-padding, padding:-padding]
        else:
            x_t.grad += dx

    result._backward = _backward
    return result
