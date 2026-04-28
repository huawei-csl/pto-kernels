from __future__ import annotations

import ctypes
import os
import subprocess
from functools import lru_cache

import torch

ASCEND_TOOLKIT_HOME = os.environ.get("ASCEND_TOOLKIT_HOME") or os.environ.get(
    "ASCEND_HOME_PATH", ""
)
if not ASCEND_TOOLKIT_HOME:
    raise RuntimeError("Set ASCEND_TOOLKIT_HOME or ASCEND_HOME_PATH")

PTO_LIB_PATH = os.environ.get("PTO_LIB_PATH", ASCEND_TOOLKIT_HOME)
_pto_inc = os.path.join(PTO_LIB_PATH, "include")
if not os.path.isdir(_pto_inc):
    raise RuntimeError(f"PTO include directory missing: {_pto_inc!r}")

_HERE = os.path.dirname(os.path.abspath(__file__))
INCLUDE_DIR = os.path.join(_HERE, "include")
COMPILED_DIR = os.path.join(_HERE, "compiled_lib")
_DRIVER_INC = "/usr/local/Ascend/driver/kernel/inc"
_npu_dev = os.environ.get("GDN_NPU_DEVICE", "npu:0")
try:
    BLOCK_DIM = int(
        getattr(torch.npu.get_device_properties(_npu_dev), "cube_core_num", 20)
    )
except RuntimeError:
    BLOCK_DIM = 24


def torch_to_ctypes(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


def optional_torch_to_ctypes(tensor: torch.Tensor | None) -> ctypes.c_void_p:
    if tensor is None:
        return ctypes.c_void_p()
    return torch_to_ctypes(tensor)


@lru_cache(maxsize=None)
def compile_pto_kernel(
    kernel_cpp_basename: str,
    so_basename: str,
    *,
    num_heads: int,
    hidden_size: int = 128,
    chunk_size: int = 128,
    cpp_mtime_ns: int = 0,
) -> str:
    os.makedirs(COMPILED_DIR, exist_ok=True)
    cpp_path = os.path.join(_HERE, kernel_cpp_basename)
    stem = os.path.splitext(so_basename)[0]
    lib_path = os.path.join(
        COMPILED_DIR,
        f"{stem}_H{num_heads}_D{hidden_size}_C{chunk_size}.so",
    )
    extra = os.environ.get("PTO_DYNAMIC_EXTRA_FLAGS", "").split()
    flags = [
        "-fPIC",
        "-shared",
        "-xcce",
        "-DMEMORY_BASE",
        "-O2",
        "-std=gnu++17",
        "--cce-aicore-arch=dav-c220",
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-record-overflow=true",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-Wno-macro-redefined",
        "-Wno-ignored-attributes",
        f"-I{INCLUDE_DIR}",
        f"-I{_pto_inc}",
        f"-I{ASCEND_TOOLKIT_HOME}/include",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc/runtime",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc/profiling",
        f"-DGDN_H={num_heads}",
        f"-DGDN_D={hidden_size}",
        f"-DGDN_C={chunk_size}",
    ]
    if os.path.isdir(_DRIVER_INC):
        flags.append(f"-I{_DRIVER_INC}")
    flags.extend(extra)
    cmd = ["bisheng", *flags, cpp_path, "-o", lib_path]
    if os.environ.get("VERBOSE_COMPILE"):
        print("compile:", " ".join(cmd))
    subprocess.run(cmd, check=True, timeout=300)
    return lib_path
