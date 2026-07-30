"""Build + load for ``fast_hadamard_256_a5.cpp`` (Ascend A5, dav-c310 vector core).

The kernel works in tiles of ``ROWS_PER_TILE`` rows, so the returned callable pads
the batch up to a multiple of that and slices the result back -- any batch size,
including non-powers-of-two, works. Same padding convention as ``matmul_swizzle``.
"""

import ctypes
from pathlib import Path

from jit_a5 import BLOCK_DIM, NBUF, PREFETCH, compile_so, entry, stream_ptr

HERE = Path(__file__).resolve().parent
SRC = HERE / "fast_hadamard_256_a5.cpp"
N = 256  # default Walsh-Hadamard block size


def rows_for(n: int = N) -> int:
    """ROWS_PER_TILE for block size ``n``: a 32 KB tile.

    NBUF=4 buffers of that size is 128 KB, inside the 192 KB budget at every
    supported ``n``. Returns 64 at n=256, the historical default.
    """
    return max(8, 16384 // n)


ROWS_PER_TILE = rows_for(N)


def compile_kernel(
    n=N, rows_per_tile=None, nbuf=NBUF, prefetch=PREFETCH, src=None, **kw
) -> Path:
    """Compile the transform. ``kw`` passes ``verbose``/``force`` through."""
    if n < 32 or n > 2048 or n & (n - 1):
        raise ValueError(f"n must be a power of two in [32, 2048], got {n}")
    rows = rows_for(n) if rows_per_tile is None else rows_per_tile
    defs = (
        f"-DHAD_N={n}",
        f"-DROWS_PER_TILE={rows}",
        f"-DNBUF={nbuf}",
        f"-DPREFETCH={prefetch}",
    )
    return compile_so(src or SRC, f"fht_a5_n{n}_r{rows}", defs, **kw)


def load_lib(so_path, block_dim=BLOCK_DIM, n=N, rows_per_tile=None):
    """Return an in-place ``hadamard(x)``.

    Computes the unnormalized ``x @ H`` (H = the +/-1 Hadamard matrix of order
    ``n``); scale by ``1/sqrt(n)`` for the orthonormal WHT.
    """
    import torch  # noqa: F401

    rows = rows_for(n) if rows_per_tile is None else rows_per_tile
    kernel = entry(so_path, "call_hadamard256")

    def hadamard(x, block_dim=block_dim, stream_ptr_=None):
        assert (
            x.dim() == 2 and x.shape[1] == n
        ), f"expected (batch, {n}) fp16, got {tuple(x.shape)}"
        batch = int(x.shape[0])
        padded = -(-batch // rows) * rows
        buf = x
        if padded != batch:
            buf = torch.zeros((padded, n), device=x.device, dtype=x.dtype)
            buf[:batch] = x
        kernel(
            int(block_dim),
            stream_ptr(stream_ptr_),
            ctypes.c_void_p(buf.data_ptr()),
            padded,
        )
        if padded != batch:
            torch.npu.synchronize()
            x.copy_(buf[:batch])
        return x

    hadamard.block_dim = block_dim
    hadamard.raw = kernel
    return hadamard


def build_and_load(block_dim=BLOCK_DIM, n=N, rows_per_tile=None, verbose=True):
    rows = rows_for(n) if rows_per_tile is None else rows_per_tile
    so = compile_kernel(n=n, rows_per_tile=rows, verbose=verbose)
    return load_lib(so, block_dim=block_dim, n=n, rows_per_tile=rows)
