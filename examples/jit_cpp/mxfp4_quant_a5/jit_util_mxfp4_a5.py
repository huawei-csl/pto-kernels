"""Build + load for the A5 MXFP4 quantization kernel.

batch dynamic; K a compile-time template argument; block size 32 static.
"""

import ctypes
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "mxfp4_quant_a5.cpp"

K = 4096  # default row width when a caller does not choose
MX_BLOCK = 32
VECTOR_CORES = 64  # vector cores on an A5
TILE_ELEMS = 16384  # must match TILE_ELEMS in the kernel
DMA_ALIGN = 32  # a Tile refuses a transfer whose row is under 32 bytes
TILE_GRAIN = DMA_ALIGN * MX_BLOCK  # must match TILE_GRAIN in the kernel
# Widths with an instantiation; mirrors SUPPORTED_K in the kernel.
SUPPORTED_K = tuple(
    int(k)
    for k in (
        "64 96 128 192 256 512 768 896 1024 1152 1280 1408 1536 1664 1792 2048 "
        "2560 2816 3072 3584 4096 5120 6144 7168 8192 14336"
    ).split()
)

# (vector_cores, stream, input, nibbles, scales, batch, k)
KERNEL_ARGS = [
    ctypes.c_uint32,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.c_uint32,
]

# Flags that never vary; one string so black leaves the wrapping alone.
FIXED_FLAGS = (
    "-O2 -std=c++17 -fPIC -Wno-ignored-attributes -Wno-macro-redefined "
    "-mllvm -cce-aicore-stack-size=0x8000 "
    "-mllvm -cce-aicore-function-stack-size=0x8000 "
    "-mllvm -cce-aicore-addr-transform "
    "-mllvm -cce-aicore-dcci-insert-for-scalar=false "
    "-Xhost-start -Xhost-end"
).split()


def ascend_home() -> str:
    for key in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        if os.environ.get(key):
            return os.environ[key]
    raise RuntimeError("set ASCEND_HOME_PATH or ASCEND_TOOLKIT_HOME")


def check_row_width(k: int) -> int:
    """Validate before any arithmetic that could fail on a bad k."""
    if k not in SUPPORTED_K:
        raise ValueError(f"k must be one of {SUPPORTED_K}, got {k}")
    return k


def rows_for(k: int = K) -> int:
    """ROWS_PER_TILE. Must match RowsFor<K>; not simply ``TILE_ELEMS // k``."""
    check_row_width(k)
    cap = max(1, TILE_ELEMS // k)
    for rows in range(cap, 0, -1):
        if (rows * k) % TILE_GRAIN == 0:
            return rows
    raise ValueError(f"no admissible ROWS_PER_TILE for k={k}")


def row_quantum(k: int = K) -> int:
    """Rows the batch must be a multiple of before the wrapper pads.

    A conservative DMA bound, not the tile height: the kernel takes a partial
    last tile.
    """
    return max(1, DMA_ALIGN * MX_BLOCK // check_row_width(k))


def compile_kernel(force: bool = False, verbose: bool = True) -> Path:
    """Compile and link the kernel to a device .so, reusing an up-to-date one."""
    out_dir = HERE / "build"
    out_dir.mkdir(parents=True, exist_ok=True)
    obj, so = out_dir / "mxfp4_a5.o", out_dir / "mxfp4_a5.so"
    if not force and so.exists() and so.stat().st_mtime >= SRC.stat().st_mtime:
        if verbose:
            print(f"[compile] up-to-date, reusing {so.name}")
        return so
    home = ascend_home()
    bisheng = f"{home}/bin/bisheng"
    arch = ["--cce-aicore-arch=dav-c310-vec", "-DREGISTER_BASE"]
    inc = [f"-I{home}/aarch64-linux/include", f"-I{home}/include"]
    cmd = [bisheng, "-xcce", *arch, *FIXED_FLAGS, *inc]
    subprocess.run([*cmd, "-c", str(SRC), "-o", str(obj)], check=True)
    link = f"-fPIC -shared --cce-fatobj-link -Wl,-soname,{so.name}".split()
    subprocess.run([bisheng, *link, str(obj), "-o", str(so)], check=True)
    return so


def bind_launcher(so_path, name, argtypes=None):
    """ctypes-load ``so_path`` and bind the launcher ``name``."""
    fn = getattr(ctypes.CDLL(str(so_path)), name)
    fn.argtypes = list(KERNEL_ARGS if argtypes is None else argtypes)
    fn.restype = None
    return fn


def current_stream_ptr():
    """Raw pointer for the ACTIVE stream, resolved per launch.

    Never cached: a cached pointer sends later launches to whichever stream
    was current first -- a race, not an error. The raw accessor is used instead
    of current_stream() because it is far cheaper and follows a
    `with torch.npu.stream(...)` block identically.
    """
    import torch
    import torch_npu

    # pylint: disable=protected-access  # the public accessor costs 8.9 us
    raw = getattr(torch_npu._C, "_npu_getCurrentRawStream", None)
    if raw is not None:
        return ctypes.c_void_p(raw(torch.npu.current_device()))
    handle = getattr(torch.npu.current_stream(), "npu_stream", None)
    if handle is None:
        raise RuntimeError("could not resolve the NPU stream pointer")
    return ctypes.c_void_p(int(handle))


def kernel_rows_for(so_path):
    """The kernel's own RowsFor<K>, so a test can pin it against rows_for()."""
    fn = getattr(ctypes.CDLL(str(so_path)), "mxfp4_rows_for")
    fn.argtypes = [ctypes.c_uint32]
    fn.restype = ctypes.c_uint32

    def query(k):
        rows = int(fn(k))
        if rows == 0:
            raise ValueError(f"kernel has no instantiation for k={k}")
        return rows

    return query


def load_quantizer(so_path, vector_cores: int = VECTOR_CORES, k: int = K):
    """Return a callable mapping a (batch, k) bf16 tensor to (nibbles, scales)."""
    import torch  # noqa: F401

    # Validated here as well as in compile_kernel: the dispatching launcher's
    # default case is a silent no-op, so an unchecked k would hand back an
    # untouched output buffer rather than failing.
    check_row_width(k)
    quantum = row_quantum(k)
    kernel = bind_launcher(so_path, "call_mxfp4_quant")

    def run(tensor, out=None, stream_ptr=None):
        assert (
            tensor.dim() == 2 and tensor.shape[1] == k
        ), f"expected (batch, {k}) bfloat16, got {tuple(tensor.shape)}"
        # The kernel reads the buffer as bf16 and as one flat run, and can report
        # neither: a wider dtype is reinterpreted and a strided view is read as if
        # contiguous, both silently.
        assert tensor.dtype == torch.bfloat16, f"expected bfloat16, got {tensor.dtype}"
        assert (
            tensor.is_contiguous()
        ), "expected a contiguous tensor; call .contiguous()"

        batch = int(tensor.shape[0])
        padded = -(-batch // quantum) * quantum
        padded_input = tensor
        if padded != batch:
            padded_input = torch.zeros(
                (padded, k), device=tensor.device, dtype=tensor.dtype
            )
            padded_input[:batch] = tensor
            # The ctypes launch is not ordered against torch's copy, so without
            # this the kernel can quantize the still-zero buffer and return all
            # zeros -- which is a plausible-looking result, not an error.
            torch.npu.synchronize()
        if out is None:
            nibbles, scales = (
                torch.empty((padded, cols), device=tensor.device, dtype=torch.uint8)
                for cols in (k // 2, k // MX_BLOCK)
            )
        else:
            nibbles, scales = out
            # RULE: TSTORE needs a 512-byte-aligned destination and fails
            # SILENTLY otherwise -- the DMA just does not land.
            for name, buf in (("nibbles", nibbles), ("scales", scales)):
                assert buf.data_ptr() % 512 == 0, f"{name} not 512-byte aligned"
                assert buf.is_contiguous(), f"{name} output must be contiguous"
        kernel(
            int(vector_cores),
            current_stream_ptr() if stream_ptr is None else stream_ptr,
            ctypes.c_void_p(padded_input.data_ptr()),
            ctypes.c_void_p(nibbles.data_ptr()),
            ctypes.c_void_p(scales.data_ptr()),
            padded,
            k,
        )
        return nibbles[:batch], scales[:batch]

    run.vector_cores = vector_cores
    run.rows_per_tile = rows_for(k)
    run.row_quantum = quantum
    return run


def build_and_load(vector_cores: int = VECTOR_CORES, k: int = K, verbose: bool = True):
    check_row_width(k)  # before rows_for divides by it
    return load_quantizer(
        compile_kernel(verbose=verbose), vector_cores=vector_cores, k=k
    )
