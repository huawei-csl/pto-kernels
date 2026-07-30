"""Build + load for the A5 Walsh-Hadamard kernel and its copy-floor reference.

Both are the same plumbing over a different translation unit and launcher, so one
module covers them and a ``Kernel`` descriptor carries what differs. The transform
pads the batch up to a multiple of ``ROWS_PER_TILE`` and slices the result back, so
any batch size works (the ``matmul_swizzle`` convention); the copy reference has no
tail path and requires an aligned batch.
"""

import ctypes
from collections import namedtuple
from pathlib import Path

from jit_a5 import BLOCK_DIM, NBUF, PREFETCH, compile_so, entry, stream_ptr

HERE = Path(__file__).resolve().parent
N = 256  # default block size / row width

Kernel = namedtuple("Kernel", "src launcher macro tag pad")
HADAMARD = Kernel("fast_hadamard_256_a5.cpp", "call_hadamard256", "HAD_N", "fht", True)
COPY = Kernel("copy_ref_256_a5.cpp", "call_copy256", "COPY_N", "copy", False)


def rows_for(n: int = N) -> int:
    """ROWS_PER_TILE for block size ``n``: a 32 KB tile.

    NBUF=4 of those is 128 KB, inside UB at every supported ``n``. 64 at n=256.
    """
    return max(8, 16384 // n)


ROWS_PER_TILE = rows_for(N)


def compile_kernel(
    kind=HADAMARD, n=N, rows_per_tile=None, nbuf=NBUF, prefetch=PREFETCH, **kw
):
    """Compile one kernel. ``kw`` passes ``verbose``/``force`` through."""
    if n < 32 or n > 2048 or n & (n - 1):
        raise ValueError(f"n must be a power of two in [32, 2048], got {n}")
    rows = rows_for(n) if rows_per_tile is None else rows_per_tile
    defs = (
        f"-D{kind.macro}={n}",
        f"-DROWS_PER_TILE={rows}",
        f"-DNBUF={nbuf}",
        f"-DPREFETCH={prefetch}",
    )
    return compile_so(HERE / kind.src, f"{kind.tag}_a5_n{n}_r{rows}", defs, **kw)


def load_lib(so_path, kind=HADAMARD, block_dim=BLOCK_DIM, n=N, rows_per_tile=None):
    """Return an in-place callable over a (batch, n) fp16 tensor.

    For HADAMARD this is the unnormalized ``x @ H`` (H = the +/-1 Hadamard matrix of
    order ``n``); scale by ``1/sqrt(n)`` for the orthonormal WHT. For COPY it leaves
    ``x`` unchanged and exists only to be timed.
    """
    import torch  # noqa: F401

    rows = rows_for(n) if rows_per_tile is None else rows_per_tile
    kernel = entry(so_path, kind.launcher)

    def run(x, block_dim=block_dim, stream=None):
        assert (
            x.dim() == 2 and x.shape[1] == n
        ), f"expected (batch, {n}) fp16, got {tuple(x.shape)}"
        batch = int(x.shape[0])
        if not kind.pad:
            assert batch % rows == 0, f"batch {batch} must be a multiple of {rows}"
        padded = -(-batch // rows) * rows if kind.pad else batch
        buf = x
        if padded != batch:
            buf = torch.zeros((padded, n), device=x.device, dtype=x.dtype)
            buf[:batch] = x
        ptr = ctypes.c_void_p(buf.data_ptr())
        kernel(int(block_dim), stream_ptr(stream), ptr, padded)
        if padded != batch:
            torch.npu.synchronize()
            x.copy_(buf[:batch])
        return x

    run.block_dim = block_dim
    return run


def build_and_load(
    kind=HADAMARD, block_dim=BLOCK_DIM, n=N, rows_per_tile=None, verbose=True
):
    rows = rows_for(n) if rows_per_tile is None else rows_per_tile
    so = compile_kernel(kind, n=n, rows_per_tile=rows, verbose=verbose)
    return load_lib(so, kind, block_dim=block_dim, n=n, rows_per_tile=rows)
