"""JIT compile + load utilities for stream_c2v and stream_v2c kernels."""

from __future__ import annotations

import ctypes
import os
import subprocess
from functools import lru_cache

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))

ASCEND_TOOLKIT_HOME = os.environ.get("ASCEND_TOOLKIT_HOME") or os.environ.get(
    "ASCEND_HOME_PATH", ""
)
if not ASCEND_TOOLKIT_HOME:
    raise RuntimeError("Set ASCEND_TOOLKIT_HOME or ASCEND_HOME_PATH")

PTO_LIB_PATH = os.environ.get("PTO_LIB_PATH", ASCEND_TOOLKIT_HOME)
_PTO_INC = os.path.join(PTO_LIB_PATH, "include")
_DRIVER_INC = "/usr/local/Ascend/driver/kernel/inc"

_NPU_DEVICE = os.environ.get("NPU_DEVICE", "npu:7")
try:
    BLOCK_DIM = int(
        getattr(torch.npu.get_device_properties(_NPU_DEVICE), "cube_core_num", 20)
    )
except (RuntimeError, AssertionError):
    BLOCK_DIM = 24


def _compile(cpp_basename: str, so_basename: str, verbose: bool = True) -> str:
    cpp_path = os.path.join(_HERE, cpp_basename)
    lib_path = os.path.join(_HERE, so_basename)
    flags = [
        "-fPIC",
        "-shared",
        "-xcce",
        "-DMEMORY_BASE",
        "-O2",
        "-std=gnu++17",
        "--cce-aicore-arch=dav-c220",
        "-mllvm",
        "-cce-aicore-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-function-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-record-overflow=true",
        "-mllvm",
        "-cce-aicore-dcci-insert-for-scalar=false",
        "-Wno-macro-redefined",
        "-Wno-ignored-attributes",
        f"-I{_PTO_INC}",
        f"-I{ASCEND_TOOLKIT_HOME}/include",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc/runtime",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc/profiling",
    ]
    if os.path.isdir(_DRIVER_INC):
        flags.append(f"-I{_DRIVER_INC}")
    cmd = ["bisheng", *flags, cpp_path, "-o", lib_path]
    if verbose:
        print("Compiling:", " ".join(cmd))
    subprocess.run(cmd, check=True, timeout=300)
    if verbose:
        print(f"Compiled → {lib_path}")
    return lib_path


@lru_cache(maxsize=1)
def load_stream_c2v(verbose: bool = True) -> "StreamC2VKernel":
    lib_path = _compile("stream_c2v.cpp", "stream_c2v.so", verbose=verbose)
    lib = ctypes.CDLL(os.path.abspath(lib_path))
    # void call_stream_c2v(uint32_t block_dim, void *stream,
    #                       uint8_t *A, uint8_t *B, uint8_t *workspace, int32_t num_iters)
    lib.call_stream_c2v.argtypes = [
        ctypes.c_uint32,  # block_dim
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # A
        ctypes.c_void_p,  # B
        ctypes.c_void_p,  # workspace
        ctypes.c_int32,  # num_iters
    ]
    lib.call_stream_c2v.restype = None
    return StreamC2VKernel(lib, BLOCK_DIM)


@lru_cache(maxsize=1)
def load_stream_v2c(verbose: bool = True) -> "StreamV2CKernel":
    lib_path = _compile("stream_v2c.cpp", "stream_v2c.so", verbose=verbose)
    lib = ctypes.CDLL(os.path.abspath(lib_path))
    # void call_stream_v2c(uint32_t block_dim, void *stream,
    #                       uint8_t *A, uint8_t *D,
    #                       uint8_t *workspace, int32_t num_iters)
    lib.call_stream_v2c.argtypes = [
        ctypes.c_uint32,  # block_dim
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # A
        ctypes.c_void_p,  # D
        ctypes.c_void_p,  # workspace
        ctypes.c_int32,  # num_iters
    ]
    lib.call_stream_v2c.restype = None
    return StreamV2CKernel(lib, BLOCK_DIM)


class StreamC2VKernel:
    def __init__(self, lib: ctypes.CDLL, block_dim: int) -> None:
        self._lib = lib
        self._block_dim = block_dim

    def __call__(
        self, A: torch.Tensor, B: torch.Tensor, workspace: torch.Tensor, num_iters: int
    ) -> None:
        """A: [num_cores*T, T], B: [T, T], workspace: [num_cores*T, T]."""
        stream_ptr = ctypes.c_void_p(torch.npu.current_stream().npu_stream)
        self._lib.call_stream_c2v(
            self._block_dim,
            stream_ptr,
            ctypes.c_void_p(A.data_ptr()),
            ctypes.c_void_p(B.data_ptr()),
            ctypes.c_void_p(workspace.data_ptr()),
            ctypes.c_int32(num_iters),
        )


class StreamV2CKernel:
    def __init__(self, lib: ctypes.CDLL, block_dim: int) -> None:
        self._lib = lib
        self._block_dim = block_dim

    def __call__(
        self, A: torch.Tensor, D: torch.Tensor, workspace: torch.Tensor, num_iters: int
    ) -> None:
        """A, D: [num_iters*num_cores*T, T], workspace: [num_cores*T, T]."""
        stream_ptr = ctypes.c_void_p(torch.npu.current_stream().npu_stream)
        self._lib.call_stream_v2c(
            self._block_dim,
            stream_ptr,
            ctypes.c_void_p(A.data_ptr()),
            ctypes.c_void_p(D.data_ptr()),
            ctypes.c_void_p(workspace.data_ptr()),
            ctypes.c_int32(num_iters),
        )
