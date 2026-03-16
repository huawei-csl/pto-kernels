import ctypes
import os
import subprocess
from pathlib import Path

import torch

ASCEND_TOOLKIT_HOME = os.environ["ASCEND_TOOLKIT_HOME"]
PTO_LIB_PATH = os.environ.get("PTO_LIB_PATH", ASCEND_TOOLKIT_HOME)
DEFAULT_DEVICE = "npu:0"


def normalize_npu_device(device: str | int) -> str:
    """Normalize NPU device inputs to canonical 'npu:<index>' format."""
    text = str(device).strip().strip('"').strip("'")
    if text.lower().startswith("npu:"):
        index = text.split(":", 1)[1].strip()
    else:
        index = text

    if not index.isdigit():
        raise ValueError(
            f"Invalid NPU device '{device}'. Expected values like 0 or npu:0."
        )
    return f"npu:{int(index)}"


def get_cube_block_dim(device: str | int = DEFAULT_DEVICE) -> int:
    resolved_device = normalize_npu_device(device)
    return int(
        getattr(
            torch.npu.get_device_properties(resolved_device),
            "cube_core_num",
            BLOCK_DIM,
        )
    )


BLOCK_DIM = int(getattr(torch.npu.get_device_properties("npu:0"), "cube_core_num", 20))


def chmod_output_path(path: str | Path) -> None:
    try:
        os.chmod(path, 0o777)
    except OSError:
        pass


def compile_cpp(
    kernel_cpp: str,
    verbose: bool = False,
    timeout: int = 120,
    so_dir: str | None = None,
) -> str:
    kernel_path = Path(kernel_cpp).resolve()
    out_dir = (
        Path(so_dir) if so_dir is not None else kernel_path.parent / "outputs" / "so"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    chmod_output_path(out_dir)
    lib_path = out_dir / f"{kernel_path.stem}_jit.so"

    flags = [
        "-fPIC",
        "-shared",
        "-xcce",
        "-DMEMORY_BASE",
        "-O2",
        "-std=c++17",
        "-Wno-ignored-attributes",
        "--cce-aicore-arch=dav-c220-vec",
        "-isystem",
        f"{PTO_LIB_PATH}/include",
    ]

    command = ["bisheng", *flags, str(kernel_path), "-o", str(lib_path)]
    if verbose:
        print("compile command:", " ".join(command))

    try:
        subprocess.run(command, timeout=timeout, check=True)
    except Exception as e:
        raise RuntimeError(f"Compile failed: {e}") from e

    if verbose:
        print(f"generated {lib_path}")
    chmod_output_path(lib_path)
    return str(lib_path)


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def load_lib(lib_path, block_dim=BLOCK_DIM):
    lib_path = os.path.abspath(lib_path)
    lib = ctypes.CDLL(lib_path)
    resolved_block_dim = max(1, int(block_dim))

    # call_kernel(blockDim, stream, x, batch, n, log2_n)
    lib.call_kernel.argtypes = [
        ctypes.c_uint32,  # blockDim
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # x (in-place)
        ctypes.c_uint32,  # batch
        ctypes.c_uint32,  # n
        ctypes.c_uint32,  # log2_n
    ]
    lib.call_kernel.restype = None

    def hadamard_func(
        x,
        batch,
        n,
        log2_n,
        block_dim=resolved_block_dim,
        stream_ptr=None,
    ):
        if stream_ptr is None:
            stream = torch.npu.current_stream()
            stream_ptr = getattr(stream, "_as_parameter_", None)
        launch_block_dim = resolved_block_dim if block_dim is None else int(block_dim)
        lib.call_kernel(
            launch_block_dim,
            stream_ptr,
            torch_to_ctypes(x),
            batch,
            n,
            log2_n,
        )

    hadamard_func.block_dim = resolved_block_dim

    return hadamard_func


def jit_compile(
    src_path,
    verbose=True,
    clean_up=False,
    so_dir=None,
    device: str | int = DEFAULT_DEVICE,
):
    resolved_device = normalize_npu_device(device)
    block_dim = get_cube_block_dim(resolved_device)
    lib_path = compile_cpp(src_path, verbose=verbose, so_dir=so_dir)
    func = load_lib(lib_path, block_dim=block_dim)
    if clean_up:
        os.remove(lib_path)
    return func
