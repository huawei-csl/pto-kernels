"""Build + load for the A5 Walsh-Hadamard kernel.

The transform needs no ``-D``: one .so holds an instantiation per supported ``N``
and its launcher dispatches on ``n``, so every block size shares one build. It pads
the batch up to a multiple of ``ROWS_PER_TILE`` and slices the result back, so any
batch size works (the ``matmul_swizzle`` convention).
"""

import ctypes
from pathlib import Path

from jit_a5 import BLOCK_DIM, KERNEL_ARGS, compile_so, entry, stream_ptr

HERE = Path(__file__).resolve().parent
N = 256  # default block size; must match DEFAULT_N in the kernel
# the dispatching launcher takes n as a fifth argument
DISPATCH_ARGS = KERNEL_ARGS + [ctypes.c_uint32]

SRC = HERE / "fast_hadamard_a5.cpp"


def check_n(n):
    if n < 32 or n > 2048 or n & (n - 1):
        raise ValueError(f"n must be a power of two in [32, 2048], got {n}")
    return n


def rows_for(n: int = N) -> int:
    """ROWS_PER_TILE for block size ``n``: a 16 KB tile, floored at 4 rows.

    The padding wrapper needs this before any .so exists, so it is stated here as
    well as in the kernel's RowsFor<N>. test_rows_for_matches_kernel pins the two
    together rather than trusting them to stay in step.

    16 KB is measured, not assumed: it is the fastest tile at every supported N
    (see README "Block size"). 8192 is 16 KB of fp16.
    """
    return max(4, 8192 // check_n(n))


ROWS_PER_TILE = rows_for(N)


def compile_kernel(n=N, **kw):
    """Compile the transform. Shape lives in the kernel, so there is nothing to pass
    and one .so serves every N. ``kw`` forwards ``verbose``/``force``."""
    check_n(n)
    return compile_so(SRC, "fht_a5", (), **kw)


def load_lib(so_path, block_dim=BLOCK_DIM, n=N, rows_per_tile=None):
    """Return an in-place callable over a (batch, n) fp16 tensor.

    Computes the unnormalized ``x @ H`` (H = the +/-1 Hadamard matrix of order
    ``n``); scale by ``1/sqrt(n)`` for the orthonormal WHT.
    """
    import torch  # noqa: F401

    # Validated here as well as in compile_kernel: the dispatching launcher's
    # default case is a silent no-op, so an unchecked n would hand back unchanged
    # data instead of failing (device-confirmed for n=16/192/4096/0).
    check_n(n)
    rows = rows_for(n) if rows_per_tile is None else rows_per_tile
    kernel = entry(so_path, "call_hadamard", DISPATCH_ARGS)

    def run(x, block_dim=block_dim, stream=None):
        assert (
            x.dim() == 2 and x.shape[1] == n
        ), f"expected (batch, {n}) fp16, got {tuple(x.shape)}"
        # The kernel reads the buffer as half and as one flat run. Neither is
        # checked by the shape assert, and getting either wrong is silent: a
        # wider dtype is reinterpreted, and a strided view is read as if flat.
        # The contiguity case is worse than it looks -- the padding path below
        # copies into a fresh buffer and happens to work, so without this the
        # corruption depends on whether batch divides ROWS_PER_TILE.
        assert x.dtype == torch.float16, f"expected fp16, got {x.dtype}"
        assert x.is_contiguous(), "expected a contiguous tensor; call .contiguous()"
        batch = int(x.shape[0])
        padded = -(-batch // rows) * rows
        buf = x
        if padded != batch:
            buf = torch.zeros((padded, n), device=x.device, dtype=x.dtype)
            buf[:batch] = x
        ptr = ctypes.c_void_p(buf.data_ptr())
        args = (int(block_dim), stream_ptr(stream), ptr, padded)
        kernel(*args, n)
        if padded != batch:
            torch.npu.synchronize()
            x.copy_(buf[:batch])
        return x

    run.block_dim = block_dim
    return run


def kernel_rows_for(so_path):
    """The kernel's own RowsFor<N>, so a test can pin it against rows_for().

    Raises on an unsupported n rather than returning the kernel's 0, which a
    caller computing padding would divide by.
    """
    fn = getattr(ctypes.CDLL(str(so_path)), "hadamard_rows_for")
    fn.argtypes = [ctypes.c_uint32]
    fn.restype = ctypes.c_uint32

    def query(n):
        rows = int(fn(n))
        if rows == 0:
            raise ValueError(f"kernel has no instantiation for n={n}")
        return rows

    return query


def build_and_load(block_dim=BLOCK_DIM, n=N, rows_per_tile=None, verbose=True):
    check_n(n)  # before rows_for divides by it
    rows = rows_for(n) if rows_per_tile is None else rows_per_tile
    so = compile_kernel(n=n, verbose=verbose)
    return load_lib(so, block_dim=block_dim, n=n, rows_per_tile=rows)
