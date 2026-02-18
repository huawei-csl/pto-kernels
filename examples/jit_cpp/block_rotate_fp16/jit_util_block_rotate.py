import ctypes
import os
import subprocess

import torch

ASCEND_TOOLKIT_HOME = os.environ["ASCEND_TOOLKIT_HOME"]
PTO_LIB_PATH = os.environ.get("PTO_LIB_PATH", ASCEND_TOOLKIT_HOME)
BLOCK_DIM = 20  # hard-coded to 910B4 cube core number


def compile_cpp(kernel_cpp: str, verbose: bool = False, timeout: int = 120) -> str:
    lib_path = os.path.join(os.path.dirname(kernel_cpp), "block_rotate_jit.so")

    flags = [
        "-fPIC",
        "-shared",
        "-xcce",
        "-DMEMORY_BASE",
        "-O2",
        "-std=c++17",
        "--cce-soc-version=Ascend910B4",
        "--cce-soc-core-type=CubeCore",
        f"-I{PTO_LIB_PATH}/include",
    ]

    command = ["bisheng", *flags, kernel_cpp, "-o", lib_path]
    if verbose:
        print("compile command:", " ".join(command))

    try:
        subprocess.run(command, timeout=timeout, check=True)
    except Exception as e:
        raise RuntimeError(f"Compile failed: {e}") from e

    if verbose:
        print(f"generated {lib_path}")
    return lib_path


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def load_lib(lib_path):
    lib_path = os.path.abspath(lib_path)
    lib = ctypes.CDLL(lib_path)

    # call_kernel(blockDim, stream, a, b, c, m)
    lib.call_kernel.argtypes = [
        ctypes.c_uint32,  # blockDim
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # a
        ctypes.c_void_p,  # b
        ctypes.c_void_p,  # c
        ctypes.c_uint32,  # m
    ]
    lib.call_kernel.restype = None

    default_block_dim = BLOCK_DIM

    def block_rotate_func(a, b, c, m, block_dim=default_block_dim, stream_ptr=None):
        if stream_ptr is None:
            stream = torch.npu.current_stream()
            stream_ptr = getattr(  # pylint: disable=protected-access
                stream, "_as_parameter_", None
            )
        lib.call_kernel(
            block_dim,
            stream_ptr,
            torch_to_ctypes(a),
            torch_to_ctypes(b),
            torch_to_ctypes(c),
            m,
        )

    return block_rotate_func


def jit_compile(src_path, verbose=True, clean_up=False):
    lib_path = compile_cpp(src_path, verbose=verbose)
    func = load_lib(lib_path)
    if clean_up:
        os.remove(lib_path)
    return func
