import ctypes
import os
import subprocess
from pathlib import Path

import torch
from jit_util_hadamard import DEFAULT_DEVICE, get_cube_block_dim, normalize_npu_device

ASCEND_TOOLKIT_HOME = os.environ["ASCEND_TOOLKIT_HOME"]
PTO_LIB_PATH = os.environ.get("PTO_LIB_PATH", ASCEND_TOOLKIT_HOME)


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
    return str(lib_path)


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def load_lib(lib_path, device: str | int = DEFAULT_DEVICE, block_dim=None):
    lib_path = os.path.abspath(lib_path)
    lib = ctypes.CDLL(lib_path)
    resolved_device = normalize_npu_device(device)
    resolved_block_dim = (
        max(1, int(block_dim))
        if block_dim is not None
        else get_cube_block_dim(resolved_device)
    )

    lib.call_quantize_kernel.argtypes = [
        ctypes.c_uint32,  # blockDim
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # x
        ctypes.c_void_p,  # y
        ctypes.c_uint32,  # batch
        ctypes.c_uint32,  # n
        ctypes.c_float,  # scale
    ]
    lib.call_quantize_kernel.restype = None

    def quantize_func(x, y, scale, block_dim=resolved_block_dim, stream_ptr=None):
        if stream_ptr is None:
            stream_ptr = torch.npu.current_stream()._as_parameter_  # noqa
        launch_block_dim = resolved_block_dim if block_dim is None else int(block_dim)
        lib.call_quantize_kernel(
            launch_block_dim,
            stream_ptr,
            torch_to_ctypes(x),
            torch_to_ctypes(y),
            x.shape[0],
            x.shape[1],
            scale,
        )

    quantize_func.block_dim = resolved_block_dim
    return quantize_func


def jit_compile(
    src_path,
    verbose=True,
    clean_up=False,
    so_dir=None,
    device: str | int = DEFAULT_DEVICE,
    block_dim=None,
):
    resolved_device = normalize_npu_device(device)
    lib_path = compile_cpp(src_path, verbose=verbose, so_dir=so_dir)
    func = load_lib(lib_path, device=resolved_device, block_dim=block_dim)
    if clean_up:
        os.remove(lib_path)
    return func
