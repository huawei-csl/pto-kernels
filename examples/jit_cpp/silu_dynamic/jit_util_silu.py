import os
import subprocess
import ctypes

import torch

ASCEND_TOOLKIT_HOME = os.environ["ASCEND_TOOLKIT_HOME"]
PTO_LIB_PATH = os.environ["PTO_LIB_PATH"]
BLOCK_DIM = 20  # 910B4, TODO: query platform information


def compile_cpp(kernel_cpp: str, verbose: bool = False, timeout: int = 120) -> str:
    lib_path = os.path.join(os.path.dirname(kernel_cpp), "rilu_jit.so")

    flags = [
        "-fPIC",
        "-shared",
        "-xcce",
        "--npu-arch=dav-2201",
        "-DMEMORY_BASE",  # here hardcoded for A2A3; TODO: expose this option to jit interface
        "-O2",
        "-std=c++17",
        f"-I{PTO_LIB_PATH}/include",
    ]

    command = ["bisheng", *flags, kernel_cpp, "-o", lib_path]
    if verbose:
        print(f"compile {kernel_cpp} with command: \n", command)

    try:
        subprocess.run(command, timeout=timeout, check=True)
    except Exception as e:
        raise RuntimeError(f"Compile failed: {e}") from e

    if verbose:
        print(f"generated {lib_path}")
    return lib_path


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def load_lib(lib_path, check_type=True):
    lib_path = os.path.abspath(lib_path)
    lib = ctypes.CDLL(lib_path)

    if check_type:
        lib.call_kernel.argtypes = [
            ctypes.c_uint32,  # blockDim
            ctypes.c_void_p,  # stream
            ctypes.c_void_p,  # y
            ctypes.c_void_p,  # x
            ctypes.c_int,  # N
        ]
        lib.call_kernel.restype = None

    default_block_dim = BLOCK_DIM

    def silu_func(x, y, block_dim=default_block_dim, stream_ptr=None):
        if stream_ptr is None:
            stream_ptr = torch.npu.current_stream()._as_parameter_  # noqa
        N = x.numel()
        lib.call_kernel(
            block_dim,
            stream_ptr,
            torch_to_ctypes(y),
            torch_to_ctypes(x),
            N,
        )

    return silu_func


def jit_compile(src_path, clean_up=True):
    lib_path = compile_cpp(src_path, verbose=True)
    func = load_lib(lib_path)
    if clean_up:
        os.remove(lib_path)
    return func
