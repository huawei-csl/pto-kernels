import ctypes
import os
import subprocess

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
KERNEL_SRC = os.path.join(REPO_ROOT, "csrc", "kernel", "kernel_abs.cpp")
KERNEL_INC = os.path.join(REPO_ROOT, "csrc", "kernel")

ASCEND_TOOLKIT_HOME = os.environ.get("ASCEND_TOOLKIT_HOME", "")
PTO_LIB_PATH = os.environ.get("PTO_LIB_PATH", ASCEND_TOOLKIT_HOME)

DEFAULT_BLOCK_DIM = 8


def _lib_path() -> str:
    return os.path.join(os.path.dirname(__file__), "libkernel_abs_jit.so")


def compile_cpp(verbose: bool = False, timeout: int = 120) -> str:
    lib_path = _lib_path()

    flags = [
        "bisheng",
        "--cce-aicore-arch=dav-c310",
        "-DREGISTER_BASE",
        "-O2",
        "-std=gnu++17",
        "-xcce",
        "-fPIC",
        "--shared",
        "-mllvm",
        "-cce-aicore-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-function-stack-size=0x8000",
        "-Wno-ignored-attributes",
        f"-I{KERNEL_INC}",
        f"-I{PTO_LIB_PATH}/include",
        f"-I{ASCEND_TOOLKIT_HOME}/include",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc/runtime",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc/profiling",
    ]

    command = [*flags, KERNEL_SRC, "-o", lib_path]
    if verbose:
        print("compile command:", " ".join(command))

    try:
        result = subprocess.run(
            command,
            timeout=timeout,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        output = e.stdout or ""
        if e.stderr:
            output += e.stderr
        raise RuntimeError(
            f"Compile failed with exit code {e.returncode}:\n{output}"
        ) from e

    if verbose and result.stdout:
        print(result.stdout)
    if verbose:
        print(f"generated {lib_path}")
    return lib_path


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def load_lib(lib_path: str):
    lib_path = os.path.abspath(lib_path)
    lib = ctypes.CDLL(lib_path)

    lib.call_vabs_fp16.argtypes = [
        ctypes.c_uint32,  # blockDim
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # x
        ctypes.c_void_p,  # z
        ctypes.c_uint32,  # num_elements
    ]
    lib.call_vabs_fp16.restype = None

    def vabs_fp16(x, z, num_elements, block_dim=DEFAULT_BLOCK_DIM, stream_ptr=None):
        if stream_ptr is None:
            stream_ptr = torch.npu.current_stream()._as_parameter_  # noqa: SLF001
        lib.call_vabs_fp16(
            block_dim,
            stream_ptr,
            torch_to_ctypes(x),
            torch_to_ctypes(z),
            num_elements,
        )

    return vabs_fp16


def jit_compile(verbose: bool = False, force_recompile: bool = False):
    lib_path = _lib_path()
    if force_recompile or not os.path.isfile(lib_path):
        compile_cpp(verbose=verbose)
    return load_lib(lib_path)
