"""
kiri/backend/accelerate.py

Direct Apple Accelerate framework integration via ctypes.

On Apple Silicon, Accelerate's BLAS (cblas_sgemm) automatically routes
matrix multiplications through the AMX coprocessor — the undocumented
matrix acceleration unit that gives ~2x over NEON SIMD.

NumPy uses Accelerate too, but lazily and with Python overhead.
We call it directly, bypassing that overhead for hot matmul paths.

Falls back to NumPy on non-Apple platforms.
"""

import ctypes
import numpy as np
import platform

# ── try to load Accelerate ────────────────────────────────────────────────────

_accelerate = None
_AVAILABLE  = False

if platform.system() == "Darwin":
    try:
        _accelerate = ctypes.CDLL(
            "/System/Library/Frameworks/Accelerate.framework/Accelerate"
        )

        # void cblas_sgemm(
        #   CBLAS_ORDER order, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
        #   int M, int N, int K,
        #   float alpha,
        #   const float *A, int lda,
        #   const float *B, int ldb,
        #   float beta,
        #   float *C, int ldc
        # )
        _cblas_sgemm = _accelerate.cblas_sgemm
        _cblas_sgemm.restype  = None
        _cblas_sgemm.argtypes = [
            ctypes.c_int,   # order
            ctypes.c_int,   # transA
            ctypes.c_int,   # transB
            ctypes.c_int,   # M
            ctypes.c_int,   # N
            ctypes.c_int,   # K
            ctypes.c_float, # alpha
            ctypes.c_void_p, ctypes.c_int,  # A, lda
            ctypes.c_void_p, ctypes.c_int,  # B, ldb
            ctypes.c_float,                  # beta
            ctypes.c_void_p, ctypes.c_int,  # C, ldc
        ]
        _AVAILABLE = True
    except Exception:
        pass

# CBLAS constants
_CblasRowMajor = 101
_CblasNoTrans  = 111
_CblasTrans    = 112


def is_available() -> bool:
    return _AVAILABLE


def sgemm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    C = A @ B  via Accelerate cblas_sgemm (routes to AMX on Apple Silicon).

    Falls back to np.matmul if Accelerate is unavailable or shapes are wrong.
    Only handles 2D float32 inputs — the common case for Linear layers.
    """
    if (not _AVAILABLE
            or A.ndim != 2 or B.ndim != 2
            or A.dtype != np.float32 or B.dtype != np.float32):
        return np.matmul(A, B)

    # Ensure C-contiguous (row-major) layout — required by cblas_sgemm
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)

    M, K  = A.shape
    K2, N = B.shape
    if K != K2:
        return np.matmul(A, B)

    C = np.empty((M, N), dtype=np.float32)

    _cblas_sgemm(
        _CblasRowMajor, _CblasNoTrans, _CblasNoTrans,
        M, N, K,
        1.0,                            # alpha
        A.ctypes.data_as(ctypes.c_void_p), K,   # A, lda
        B.ctypes.data_as(ctypes.c_void_p), N,   # B, ldb
        0.0,                            # beta
        C.ctypes.data_as(ctypes.c_void_p), N,   # C, ldc
    )
    return C


def sgemm_batched(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Batched matmul: A (batch, M, K) @ B (batch, K, N) → (batch, M, N).
    Loops over batch dimension calling sgemm per sample.
    """
    if A.ndim == 2:
        return sgemm(A, B)
    if not _AVAILABLE or A.dtype != np.float32 or B.dtype != np.float32:
        return np.matmul(A, B)

    batch = A.shape[0]
    M, K  = A.shape[1], A.shape[2]
    N     = B.shape[2]
    C     = np.empty((batch, M, N), dtype=np.float32)
    for i in range(batch):
        C[i] = sgemm(A[i], B[i])
    return C
