"""Numpy reference for the fast Walsh-Hadamard transform (unnormalized x @ H)."""
import numpy as np


def hadamard_matrix(n: int) -> np.ndarray:
    """Natural-order (Sylvester) +/-1 Hadamard matrix of order n (power of two)."""
    H = np.array([[1.0]], dtype=np.float64)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


def ref_hadamard(x: np.ndarray) -> np.ndarray:
    """Unnormalized WHT of each row: y = x @ H. Returns float32."""
    n = x.shape[-1]
    return (x.astype(np.float32) @ hadamard_matrix(n)).astype(np.float32)


def butterfly(x: np.ndarray, log2n: int) -> np.ndarray:
    """Equivalent iterative in-place deinterleave butterfly (== x @ H_natural)."""
    y = x.astype(np.float64).copy()
    ap = y.shape[1] // 2
    for _ in range(log2n):
        even = y[:, 0 : 2 * ap : 2].copy()
        odd = y[:, 1 : 2 * ap : 2].copy()
        y[:, :ap] = even + odd
        y[:, ap : 2 * ap] = even - odd
    return y.astype(np.float32)
