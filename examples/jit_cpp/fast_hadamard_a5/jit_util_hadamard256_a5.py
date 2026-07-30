"""Self-contained JIT build + load for the N=256 fast Walsh-Hadamard kernel
(``fast_hadamard_256_a5.cpp``) on an Ascend 950 / A5 (dav-c310 vector core).

The shared ``examples/jit_cpp`` helper targets dav-c220 (``-DMEMORY_BASE``);
this kernel is A5 (``dav-c310-vec``, ``-DREGISTER_BASE``), so we invoke
``bisheng`` directly here. Only a working ``bisheng`` (set ``ASCEND_HOME_PATH``
/ ``ASCEND_TOOLKIT_HOME``) is needed to build, and ``torch_npu`` at run time.

The device kernel processes the batch in tiles of ``ROWS_PER_TILE`` rows, so the
returned callable pads the batch up to a multiple of ``ROWS_PER_TILE`` and
slices the result back. This lets arbitrary (e.g. non-power-of-2) batch sizes
work transparently, following the padding convention of the ``matmul_swizzle``
example.
"""

import ctypes
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent

N = 256  # Walsh-Hadamard block size this kernel supports
ROWS_PER_TILE = 64  # rows per GM<->UB tile; must match the compiled -DROWS_PER_TILE


def rows_for(n: int = N) -> int:
    """Default ROWS_PER_TILE for block size ``n``: a 32 KB tile.

    NBUF=4 buffers of that size is 128 KB, inside the 192 KB budget at every
    supported ``n``. Returns 64 at n=256, i.e. the historical default.
    """
    return max(8, 16384 // n)


NBUF = 4  # pipeline depth (UB buffers)
PREFETCH = 2  # tiles prefetched ahead


def _ascend_home() -> str:
    for key in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        value = os.environ.get(key)
        if value:
            return value
    raise RuntimeError("set ASCEND_HOME_PATH or ASCEND_TOOLKIT_HOME")


def compile_kernel(
    src: Path | None = None,
    out_dir: Path | None = None,
    n: int = N,
    rows_per_tile: int | None = None,
    nbuf: int = NBUF,
    prefetch: int = PREFETCH,
    verbose: bool = True,
    force: bool = False,
) -> Path:
    """Compile ``fast_hadamard_256_a5.cpp`` to a device ``.so``.

    Skips the ``bisheng`` invocation when an up-to-date ``.so`` already exists
    (pass ``force=True`` to always rebuild).
    """
    src = Path(src) if src else HERE / "fast_hadamard_256_a5.cpp"
    out_dir = Path(out_dir) if out_dir else HERE / "build"
    out_dir.mkdir(parents=True, exist_ok=True)
    home = _ascend_home()
    bisheng = f"{home}/bin/bisheng"
    include = f"{home}/aarch64-linux/include"
    if rows_per_tile is None:
        rows_per_tile = rows_for(n)
    log2n = n.bit_length() - 1
    if 1 << log2n != n:
        raise ValueError(f"block size n must be a power of two, got {n}")
    # artifact name carries every macro that changes codegen
    obj = out_dir / f"fht_a5_n{n}_r{rows_per_tile}.o"
    so = out_dir / f"fht_a5_n{n}_r{rows_per_tile}.so"
    if not force and so.exists() and so.stat().st_mtime >= src.stat().st_mtime:
        if verbose:
            print(f"[compile] up-to-date, reusing {so}")
        return so
    flags = [
        "--cce-aicore-arch=dav-c310-vec",
        "-DREGISTER_BASE",
        f"-DHAD_N={n}",
        f"-DHAD_LOG2N={log2n}",
        f"-DROWS_PER_TILE={rows_per_tile}",
        f"-DNBUF={nbuf}",
        f"-DPREFETCH={prefetch}",
        "-O2",
        "-std=c++17",
        "-fPIC",
        "-Wno-ignored-attributes",
        "-Wno-macro-redefined",
        "-mllvm",
        "-cce-aicore-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-function-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-addr-transform",
        "-mllvm",
        "-cce-aicore-dcci-insert-for-scalar=false",
        "-Xhost-start",
        "-Xhost-end",
        f"-I{include}",
        f"-I{home}/include",
    ]
    if verbose:
        print(f"[compile] n={n} rows_per_tile={rows_per_tile} nbuf={nbuf} -> {so}")
    subprocess.run(
        [bisheng, "-xcce", *flags, "-c", str(src), "-o", str(obj)], check=True
    )
    subprocess.run(
        [
            bisheng,
            "-fPIC",
            "-shared",
            "--cce-fatobj-link",
            f"-Wl,-soname,{so.name}",
            str(obj),
            "-o",
            str(so),
        ],
        check=True,
    )
    return so


def _stream_ptr(stream_ptr):
    if stream_ptr is not None:
        return stream_ptr
    import torch  # noqa: F401

    stream = torch.npu.current_stream()
    ptr = getattr(stream, "_as_parameter_", None)
    if ptr is None:
        raise RuntimeError("could not resolve the current NPU stream pointer")
    return ptr


def load_lib(
    so_path: Path,
    block_dim: int = 64,
    n: int = N,
    rows_per_tile: int | None = None,
):
    """ctypes-load the ``.so`` and return an in-place ``hadamard256(x)`` callable.

    The kernel computes the unnormalized transform ``x @ H`` (H = the +/-1
    Hadamard matrix of order ``n``); scale by ``1/sqrt(n)`` for the orthonormal
    WHT. The callable pads the batch to a multiple of ``rows_per_tile`` so any
    batch size is accepted, and writes the result back in place into ``x``.
    """
    import torch  # noqa: F401

    if rows_per_tile is None:
        rows_per_tile = rows_for(n)
    lib = ctypes.CDLL(str(so_path))
    kernel = lib.call_hadamard256
    kernel.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    kernel.restype = None

    def hadamard256(x, block_dim: int = block_dim, stream_ptr=None):
        assert (
            x.dim() == 2 and x.shape[1] == n
        ), f"expected (batch, {N}) fp16, got {tuple(x.shape)}"
        batch = int(x.shape[0])
        padded = (batch + rows_per_tile - 1) // rows_per_tile * rows_per_tile
        buffer = x
        if padded != batch:
            buffer = torch.zeros((padded, N), device=x.device, dtype=x.dtype)
            buffer[:batch] = x
        kernel(
            int(block_dim),
            _stream_ptr(stream_ptr),
            ctypes.c_void_p(buffer.data_ptr()),
            padded,
        )
        if padded != batch:
            torch.npu.synchronize()
            x.copy_(buffer[:batch])
        return x

    hadamard256.block_dim = block_dim
    return hadamard256


def build_and_load(
    block_dim: int = 64,
    n: int = N,
    rows_per_tile: int | None = None,
    verbose: bool = True,
):
    if rows_per_tile is None:
        rows_per_tile = rows_for(n)
    so = compile_kernel(n=n, rows_per_tile=rows_per_tile, verbose=verbose)
    return load_lib(so, block_dim=block_dim, n=n, rows_per_tile=rows_per_tile)
