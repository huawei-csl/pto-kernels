"""Build + load for ``copy_ref_256_a5.cpp``, the copy-floor reference.

This is the DMA ceiling ``fast_hadamard_256_a5`` is measured against: a plain
``GM -> UB -> GM`` round trip over the same tiling with no vector-execute work.
Its own translation unit and .so, so the transform stays standalone -- the two
only need to agree on ``ROWS_PER_TILE``, which the caller passes to both.
"""

import ctypes
from pathlib import Path

from jit_a5 import BLOCK_DIM, NBUF, PREFETCH, compile_so, entry, stream_ptr

HERE = Path(__file__).resolve().parent
SRC = HERE / "copy_ref_256_a5.cpp"
N = 256  # elements per row; must match the transform's N
ROWS_PER_TILE = 64  # must match the transform's tiling


def compile_kernel(
    n=N, rows_per_tile=ROWS_PER_TILE, nbuf=NBUF, prefetch=PREFETCH, **kw
):
    defs = (
        f"-DCOPY_N={n}",
        f"-DROWS_PER_TILE={rows_per_tile}",
        f"-DNBUF={nbuf}",
        f"-DPREFETCH={prefetch}",
    )
    return compile_so(SRC, f"copy_a5_n{n}_r{rows_per_tile}", defs, **kw)


def load_lib(so_path, block_dim=BLOCK_DIM, n=N, rows_per_tile=ROWS_PER_TILE):
    """Return ``copy256(x)``, which leaves ``x`` unchanged; it exists to be timed.

    ``batch`` must be a multiple of ``rows_per_tile`` -- the kernel has no tail
    path, and the benchmark only ever calls it with aligned batches.
    """
    kernel = entry(so_path, "call_copy256")

    def copy256(x, block_dim=block_dim, stream_ptr_=None):
        assert (
            x.dim() == 2 and x.shape[1] == n
        ), f"expected (batch, {n}) fp16, got {tuple(x.shape)}"
        batch = int(x.shape[0])
        assert (
            batch % rows_per_tile == 0
        ), f"batch {batch} must be a multiple of {rows_per_tile}"
        kernel(
            int(block_dim),
            stream_ptr(stream_ptr_),
            ctypes.c_void_p(x.data_ptr()),
            batch,
        )
        return x

    copy256.block_dim = block_dim
    copy256.raw = kernel  # for benchmark loops that time the bare call
    return copy256


def build_and_load(block_dim=BLOCK_DIM, rows_per_tile=ROWS_PER_TILE, verbose=True):
    so = compile_kernel(rows_per_tile=rows_per_tile, verbose=verbose)
    return load_lib(so, block_dim=block_dim, rows_per_tile=rows_per_tile)
