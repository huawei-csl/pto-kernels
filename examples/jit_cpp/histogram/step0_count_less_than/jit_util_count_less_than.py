import os
import subprocess
import ctypes

import torch

ASCEND_TOOLKIT_HOME = os.environ["ASCEND_TOOLKIT_HOME"]
PTO_LIB_PATH = os.environ.get("PTO_LIB_PATH", ASCEND_TOOLKIT_HOME)


def compile_cpp(kernel_cpp: str, verbose: bool = False, timeout: int = 120) -> str:
    lib_path = os.path.join(os.path.dirname(kernel_cpp), "count_less_than_jit.so")

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
        subprocess.run(
            command,
            timeout=timeout,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except Exception as e:
        output = e.stdout.decode("utf-8", errors="replace") if e.stdout else ""
        raise RuntimeError(
            f"Compile failed with exit code {e.returncode}:\n{output}"
        ) from e

    if verbose:
        print(f"generated {lib_path}")
    return lib_path


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def load_lib(lib_path, check_type=True):
    lib_path = os.path.abspath(lib_path)
    lib = ctypes.CDLL(lib_path)

    default_block_dim = torch.npu.get_device_properties().vector_core_num

    if check_type:
        lib.count_less_than_fp32.argtypes = [
            ctypes.c_uint32,  # blockDim
            ctypes.c_void_p,  # stream
            ctypes.c_void_p,  # x
            ctypes.c_void_p,  # z
            ctypes.c_uint,  # in_length
            ctypes.c_float,  # pivot
        ]
        lib.count_less_than_fp32.restype = None

    def count_func(x, z, pivot, block_dim=default_block_dim, stream_ptr=None):
        if stream_ptr is None:
            stream_ptr = torch.npu.current_stream()._as_parameter_  # noqa
        N = x.numel()
        lib.count_less_than_fp32(
            block_dim, stream_ptr, torch_to_ctypes(x), torch_to_ctypes(z), N, pivot
        )

    return count_func


def jit_compile(src_path, clean_up=True):
    lib_path = compile_cpp(src_path, verbose=True)
    func = load_lib(lib_path)
    if clean_up:
        os.remove(lib_path)
    return func
