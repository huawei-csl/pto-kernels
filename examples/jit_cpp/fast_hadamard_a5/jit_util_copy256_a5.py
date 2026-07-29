"""Self-contained JIT build + load for the copy-floor reference kernel
(``copy_ref_256_a5.cpp``) on an Ascend 950 / A5 (dav-c310 vector core).

This is the DMA ceiling that ``fast_hadamard_256_a5`` is measured against: a
plain ``GM -> UB -> GM`` round trip over the same tiling, with no vector-execute
work. It is built from its own translation unit and loaded through its own
``.so`` so the transform stays standalone -- the two only need to agree on
``ROWS_PER_TILE``, which the caller passes to both.
"""

import ctypes
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent

N = 256  # elements per row; must match the transform's N
ROWS_PER_TILE = 64  # rows per GM<->UB tile; must match the transform's tiling


def _ascend_home() -> str:
    for key in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        value = os.environ.get(key)
        if value:
            return value
    raise RuntimeError("set ASCEND_HOME_PATH or ASCEND_TOOLKIT_HOME")


def compile_kernel(
    src: Path | None = None,
    out_dir: Path | None = None,
    rows_per_tile: int = ROWS_PER_TILE,
    ub_bytes: int | None = None,
    verbose: bool = True,
    force: bool = False,
) -> Path:
    """Compile ``copy_ref_256_a5.cpp`` to a device ``.so``.

    Skips the ``bisheng`` invocation when an up-to-date ``.so`` already exists
    (pass ``force=True`` to always rebuild). The artifact name carries every
    macro that affects codegen, so different tilings never collide.
    """
    src = Path(src) if src else HERE / "copy_ref_256_a5.cpp"
    out_dir = Path(out_dir) if out_dir else HERE / "build"
    out_dir.mkdir(parents=True, exist_ok=True)
    home = _ascend_home()
    bisheng = f"{home}/bin/bisheng"
    include = f"{home}/aarch64-linux/include"
    tag = f"{src.stem}_r{rows_per_tile}" + (f"_ub{ub_bytes}" if ub_bytes else "")
    obj = out_dir / f"{tag}.o"
    so = out_dir / f"{tag}.so"
    if not force and so.exists() and so.stat().st_mtime >= src.stat().st_mtime:
        if verbose:
            print(f"[compile] up-to-date, reusing {so}")
        return so
    flags = [
        "--cce-aicore-arch=dav-c310-vec",
        "-DREGISTER_BASE",
        f"-DROWS_PER_TILE={rows_per_tile}",
        f"-DCOPY_N={N}",
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
    if ub_bytes:
        flags.insert(3, f"-DUB_USABLE_BYTES={ub_bytes}u")
    if verbose:
        print(f"[compile] copy rows_per_tile={rows_per_tile} -> {so}")
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


def load_lib(so_path: Path, block_dim: int = 64, rows_per_tile: int = ROWS_PER_TILE):
    """ctypes-load the ``.so`` and return an in-place ``copy256(x)`` callable.

    The kernel copies each tile GM -> UB -> GM, so ``x`` is unchanged on return;
    it exists purely to be timed. ``batch`` must be a multiple of
    ``rows_per_tile`` (the benchmark only ever calls it with aligned batches).
    """
    lib = ctypes.CDLL(str(so_path))
    kernel = lib.call_copy256
    kernel.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    kernel.restype = None

    def copy256(x, block_dim: int = block_dim, stream_ptr=None):
        assert (
            x.dim() == 2 and x.shape[1] == N
        ), f"expected (batch, {N}) fp16, got {tuple(x.shape)}"
        batch = int(x.shape[0])
        assert (
            batch % rows_per_tile == 0
        ), f"batch {batch} must be a multiple of {rows_per_tile}"
        kernel(
            int(block_dim),
            _stream_ptr(stream_ptr),
            ctypes.c_void_p(x.data_ptr()),
            batch,
        )
        return x

    copy256.block_dim = block_dim
    copy256.raw = kernel  # for benchmark loops that time the bare call
    return copy256


def build_and_load(
    block_dim: int = 64, rows_per_tile: int = ROWS_PER_TILE, verbose: bool = True
):
    so = compile_kernel(rows_per_tile=rows_per_tile, verbose=verbose)
    return load_lib(so, block_dim=block_dim, rows_per_tile=rows_per_tile)
