import os
import subprocess
import ctypes
import torch

ASCEND_TOOLKIT_HOME = os.environ.get("ASCEND_TOOLKIT_HOME", "")
PTO_LIB_PATH = os.environ.get("PTO_LIB_PATH", ASCEND_TOOLKIT_HOME)


def _get_lib_path(kernel_cpp: str, tile_size: int) -> str:
    basename = kernel_cpp.replace(".cpp", "")
    return f"{basename}_{tile_size}_jit.so"


def compile_cpp(
    kernel_cpp: str, tile_size=64, verbose: bool = False, timeout: int = 120
) -> str:
    lib_path = _get_lib_path(kernel_cpp, tile_size)

    flags = [
        "-fPIC",
        "-shared",
        "-xcce",
        "--npu-arch=dav-2201",
        "-DMEMORY_BASE",
        "-DSCAN_TILE_SIZE=" + str(tile_size),
        "-O2",
        "-std=c++17",
        f"-I{PTO_LIB_PATH}/include",
        f"-I{ASCEND_TOOLKIT_HOME}/include",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc/runtime",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc/profiling",
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
        output = (
            e.stdout.decode("utf-8", errors="replace")
            if hasattr(e, "stdout") and e.stdout
            else ""
        )
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

    default_block_dim = 1  # single core scan

    if check_type:
        lib.scan_fp32.argtypes = [
            ctypes.c_uint32,  # blockDim
            ctypes.c_void_p,  # stream
            ctypes.c_void_p,  # x
            ctypes.c_void_p,  # y
            ctypes.c_void_p,  # u
            ctypes.c_uint,  # total_len
        ]
        lib.scan_fp32.restype = None

    def scan_func(x, y, u, total_len, block_dim=default_block_dim, stream_ptr=None):
        if stream_ptr is None:
            stream_ptr = torch.npu.current_stream()._as_parameter_  # noqa

        lib.scan_fp32(
            block_dim,
            stream_ptr,
            torch_to_ctypes(x),
            torch_to_ctypes(y),
            torch_to_ctypes(u),
            total_len,
        )

    return scan_func


def jit_compile(src_path, tile_size=64):
    lib_path = compile_cpp(src_path, tile_size=tile_size, verbose=False)
    func = load_lib(lib_path, check_type=True)
    return func


def clean_up(kernel_cpp: str, tile_size: int):
    lib_path = _get_lib_path(kernel_cpp, tile_size)
    if os.path.exists(lib_path):
        os.remove(lib_path)
