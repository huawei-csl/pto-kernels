import ctypes
import os
import subprocess
from functools import lru_cache

import torch

ASCEND_TOOLKIT_HOME = os.environ["ASCEND_TOOLKIT_HOME"]
PTO_LIB_PATH = os.environ.get("PTO_LIB_PATH", ASCEND_TOOLKIT_HOME)
BLOCK_DIM = int(getattr(torch.npu.get_device_properties("npu:0"), "cube_core_num", 20))
AICORE_ARCH = "dav-c220"

STACK_TUNING_FLAGS = [
    "-mllvm",
    "-cce-aicore-stack-size=0x8000",
    "-mllvm",
    "-cce-aicore-function-stack-size=0x8000",
    "-mllvm",
    "-cce-aicore-record-overflow=true",
]

OPTIMIZED_KERNEL_FLAGS = [
    *STACK_TUNING_FLAGS,
    "-mllvm",
    "-cce-aicore-dcci-insert-for-scalar=false",
    "-Wno-macro-redefined",
    "-Wno-ignored-attributes",
]

STEP03_KERNEL_FLAGS = [
    *STACK_TUNING_FLAGS,
    "-mllvm",
    "-cce-aicore-addr-transform",
    "-mllvm",
    "-cce-aicore-dcci-insert-for-scalar=false",
    "-DL2_CACHE_HINT",
    "-Wno-macro-redefined",
    "-Wno-ignored-attributes",
]


def _verify_aicore_predefines(kernel_cpp: str, include_flags: list[str], timeout: int) -> None:
    probe = [
        "bisheng",
        "-dM",
        "-E",
        "-xcce",
        f"--cce-aicore-arch={AICORE_ARCH}",
        *include_flags,
        kernel_cpp,
    ]
    result = subprocess.run(
        probe,
        check=True,
        timeout=min(timeout, 30),
        capture_output=True,
        text=True,
    )
    macros = result.stdout
    expected = ("#define __CCE_AICORE__ 220", "__DAV_C220_CUBE__", "__DAV_C220_VEC__")
    if not all(token in macros for token in expected):
        raise RuntimeError(
            "bisheng did not expose the expected dav-c220 AICORE predefines "
            "for this compile command. That can cause <pto/pto-inst.hpp> to "
            "skip PTO tile/instruction headers in some preprocessing paths."
        )


def compile_cpp(
    kernel_cpp: str,
    *,
    output_name: str,
    std: str,
    defines: list[str] | None = None,
    extra_flags: list[str] | None = None,
    verbose: bool = False,
    timeout: int = 180,
) -> str:
    lib_dir = os.path.join(os.path.dirname(kernel_cpp), "compiled_lib")
    os.makedirs(lib_dir, exist_ok=True)
    lib_path = os.path.join(lib_dir, output_name)
    include_flags = [
        f"-I{PTO_LIB_PATH}/include",
        f"-I{ASCEND_TOOLKIT_HOME}/include",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc/runtime",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc/profiling",
    ]

    flags = [
        "-fPIC",
        "-shared",
        "-xcce",
        "-DMEMORY_BASE",
        "-O2",
        f"-std={std}",
        f"--cce-aicore-arch={AICORE_ARCH}",
        *(extra_flags or []),
        *include_flags,
        *(defines or []),
    ]

    command = ["bisheng", *flags, kernel_cpp, "-o", lib_path]
    if verbose:
        print("compile command:", " ".join(command))

    try:
        _verify_aicore_predefines(kernel_cpp, include_flags, timeout)
        subprocess.run(command, timeout=timeout, check=True)
    except Exception as exc:
        raise RuntimeError(f"Compile failed: {exc}") from exc

    if verbose:
        print(f"generated {lib_path}")
    return lib_path


def torch_to_ctypes(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


@lru_cache(maxsize=None)
def get_causal_mask(chunk_size: int, dtype: torch.dtype, device_index: int):
    vec_num = 2
    if chunk_size % vec_num != 0:
        raise ValueError("chunk_size must be divisible by 2 for the causal mask.")
    half_chunk = chunk_size // vec_num
    mask = torch.zeros(
        (vec_num, half_chunk, chunk_size),
        device=f"npu:{device_index}",
        dtype=dtype,
    )
    for vid in range(vec_num):
        rows = torch.arange(vid * half_chunk, (vid + 1) * half_chunk, device=mask.device)
        cols = torch.arange(chunk_size, device=mask.device)
        mask[vid] = (rows[:, None] >= cols[None, :]).to(dtype)
    return mask.contiguous()


def _load_cdll(lib_path: str):
    return ctypes.CDLL(os.path.abspath(lib_path))


def load_static_nomask_lib(lib_path: str):
    lib = _load_cdll(lib_path)
    lib.call_kernel.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
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
            block_dim = q.shape[0] * q.shape[1]
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
        )

    return linear_attention_func


def load_dynamic_nomask_lib(lib_path: str):
    lib = _load_cdll(lib_path)
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


def load_dynamic_mask_lib(lib_path: str):
    lib = _load_cdll(lib_path)
    lib.call_kernel.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
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
        causal_mask: torch.Tensor,
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
            torch_to_ctypes(causal_mask),
            torch_to_ctypes(o),
            q.shape[0],
            q.shape[2],
        )

    return linear_attention_func
