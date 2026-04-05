import ctypes
import os
import subprocess
from functools import lru_cache

import torch

ASCEND_TOOLKIT_HOME = os.environ["ASCEND_TOOLKIT_HOME"]
PTO_LIB_PATH = os.environ.get("PTO_LIB_PATH", ASCEND_TOOLKIT_HOME)
BLOCK_DIM = int(getattr(torch.npu.get_device_properties("npu:0"), "cube_core_num", 20))


def compile_cpp(
    kernel_cpp: str,
    num_heads: int,
    hidden_size: int,
    chunk_size: int,
    verbose: bool = False,
    timeout: int = 180,
) -> str:
    lib_path = os.path.join(
        os.path.dirname(kernel_cpp),
        f"linear_attention_H{num_heads}_D{hidden_size}_C{chunk_size}_jit.so",
    )

    flags = [
        "-fPIC",
        "-shared",
        "-xcce",
        "-DMEMORY_BASE",
        "-O2",
        "-std=c++17",
        "--cce-aicore-arch=dav-c220",
        f"-I{PTO_LIB_PATH}/include",
        f"-I{ASCEND_TOOLKIT_HOME}/include",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc/runtime",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc/profiling",
        f"-DLINEAR_ATTN_H={num_heads}",
        f"-DLINEAR_ATTN_D={hidden_size}",
        f"-DLINEAR_ATTN_C={chunk_size}",
    ]

    command = ["bisheng", *flags, kernel_cpp, "-o", lib_path]
    if verbose:
        print("compile command:", " ".join(command))

    try:
        subprocess.run(command, timeout=timeout, check=True)
    except Exception as exc:
        raise RuntimeError(f"Compile failed: {exc}") from exc

    if verbose:
        print(f"generated {lib_path}")
    return lib_path


def torch_to_ctypes(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


def load_lib(lib_path: str):
    lib = ctypes.CDLL(os.path.abspath(lib_path))

    lib.call_kernel.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    lib.call_kernel.restype = None

    def linear_attention_func(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        workspace_1: torch.Tensor,
        workspace_2: torch.Tensor,
        o: torch.Tensor,
        block_dim: int | None = None,
        stream_ptr=None,
    ):
        if block_dim is None:
            block_dim = BLOCK_DIM
        if stream_ptr is None:
            stream_ptr = torch.npu.current_stream()._as_parameter_

        lib.call_kernel(
            block_dim,
            stream_ptr,
            torch_to_ctypes(q),
            torch_to_ctypes(k),
            torch_to_ctypes(v),
            torch_to_ctypes(workspace_1),
            torch_to_ctypes(workspace_2),
            torch_to_ctypes(o),
            q.shape[0],
            q.shape[2],
        )

    return linear_attention_func


@lru_cache(maxsize=None)
def jit_compile(
    src_path: str,
    num_heads: int,
    hidden_size: int,
    chunk_size: int,
    verbose: bool = True,
    clean_up: bool = False,
):
    lib_path = compile_cpp(
        src_path,
        num_heads=num_heads,
        hidden_size=hidden_size,
        chunk_size=chunk_size,
        verbose=verbose,
    )
    func = load_lib(lib_path)
    if clean_up:
        os.remove(lib_path)
    return func
