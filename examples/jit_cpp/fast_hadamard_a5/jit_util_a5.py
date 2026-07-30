"""Build + load for the A5 Walsh-Hadamard kernel and its copy-floor reference.

Both are the same plumbing over a different translation unit and launcher, so one
module covers them and a ``Kernel`` descriptor carries what differs.

The transform needs no ``-D``: one .so holds an instantiation per supported ``N``
and its launcher dispatches on ``n``, so every block size shares one build. It pads
the batch up to a multiple of ``ROWS_PER_TILE`` and slices the result back, so any
batch size works (the ``matmul_swizzle`` convention). The copy reference is still
compiled per shape and requires an aligned batch -- it has no tail path.
"""

import ctypes
from collections import namedtuple
from pathlib import Path

from jit_a5 import BLOCK_DIM, KERNEL_ARGS, NBUF, PREFETCH, compile_so, entry, stream_ptr

HERE = Path(__file__).resolve().parent
N = 256  # default block size; must match DEFAULT_N in the kernel
# the dispatching launcher takes n as a fifth argument
DISPATCH_ARGS = KERNEL_ARGS + [ctypes.c_uint32]

Kernel = namedtuple("Kernel", "src launcher tag pad dispatch macro")
HADAMARD = Kernel("fast_hadamard_256_a5.cpp", "call_hadamard", "fht", True, True, None)
COPY = Kernel("copy_ref_256_a5.cpp", "call_copy256", "copy", False, False, "COPY_N")


def check_n(n):
    if n < 32 or n > 2048 or n & (n - 1):
        raise ValueError(f"n must be a power of two in [32, 2048], got {n}")
    return n


def rows_for(n: int = N) -> int:
    """ROWS_PER_TILE for block size ``n``: a 32 KB tile, floored at 8 rows.

    The padding wrapper needs this before any .so exists, so it is stated here as
    well as in the kernel's RowsFor<N>. test_rows_for_matches_kernel pins the two
    together rather than trusting them to stay in step.
    """
    return max(8, 16384 // check_n(n))


ROWS_PER_TILE = rows_for(N)


def compile_kernel(
    kind=HADAMARD, n=N, rows_per_tile=None, nbuf=NBUF, prefetch=PREFETCH, **kw
):
    """Compile one kernel. ``kw`` passes ``verbose``/``force`` through."""
    check_n(n)
    if kind.dispatch:  # shape is fixed inside the kernel; nothing to pass
        return compile_so(HERE / kind.src, f"{kind.tag}_a5", (), **kw)
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

    # Validated here as well as in compile_kernel: the dispatching launcher's
    # default case is a silent no-op, so an unchecked n would hand back unchanged
    # data instead of failing (device-confirmed for n=16/192/4096/0).
    check_n(n)
    rows = rows_for(n) if rows_per_tile is None else rows_per_tile
    kernel = entry(
        so_path, kind.launcher, DISPATCH_ARGS if kind.dispatch else KERNEL_ARGS
    )

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
        args = (int(block_dim), stream_ptr(stream), ptr, padded)
        kernel(*args, n) if kind.dispatch else kernel(*args)
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


def build_and_load(
    kind=HADAMARD, block_dim=BLOCK_DIM, n=N, rows_per_tile=None, verbose=True
):
    check_n(n)  # before rows_for divides by it
    rows = rows_for(n) if rows_per_tile is None else rows_per_tile
    so = compile_kernel(kind, n=n, rows_per_tile=rows, verbose=verbose)
    return load_lib(so, kind, block_dim=block_dim, n=n, rows_per_tile=rows)
