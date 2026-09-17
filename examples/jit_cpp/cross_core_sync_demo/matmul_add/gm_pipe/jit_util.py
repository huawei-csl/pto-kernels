"""JIT utilities for matmul_add/gm_pipe kernels.

Uses pto-isa-master headers for GlobalData TPOP/TALLOC/TFREE APIs.
All tensors float16; fifo_mem float16 (half slot, same slot size as raw_flag).
"""

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

_PTO_NEW_INC = "/workdir/pto-isa-master/include"
_DRIVER_INC = "/usr/local/Ascend/driver/kernel/inc"

_NPU_DEVICE = os.environ.get("NPU_DEVICE", "npu:7")
try:
    BLOCK_DIM = int(
        getattr(torch.npu.get_device_properties(_NPU_DEVICE), "cube_core_num", 20)
    )
except (RuntimeError, AssertionError):
    BLOCK_DIM = 24

TILE_SIZE = 128
FIFO_DEPTH = 2
FIFO_ELEMS_PER_CORE = FIFO_DEPTH * TILE_SIZE * TILE_SIZE  # float16 elements


def _compile(cpp_basename: str, so_basename: str, verbose: bool = True) -> str:
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
        f"-I{_PTO_NEW_INC}",
        f"-I{ASCEND_TOOLKIT_HOME}/include",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc/runtime",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc/profiling",
    ]
    if os.path.isdir(_DRIVER_INC):
        flags.append(f"-I{_DRIVER_INC}")
    cpp = os.path.join(_HERE, cpp_basename)
    so = os.path.join(_HERE, so_basename)
    cmd = ["bisheng", *flags, cpp, "-o", so]
    if verbose:
        print("Compiling (with pto-isa-master headers):", " ".join(cmd))
    subprocess.run(cmd, check=True, timeout=300)
    if verbose:
        print(f"Compiled → {so}")
    return so


def _make_lib(so_path: str) -> ctypes.CDLL:
    lib = ctypes.CDLL(os.path.abspath(so_path))
    lib.call.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,  # block_dim, stream
        ctypes.c_void_p,
        ctypes.c_void_p,  # A, B
        ctypes.c_void_p,
        ctypes.c_void_p,  # C, D
        ctypes.c_void_p,
        ctypes.c_int64,  # fifo_mem, batch
    ]
    lib.call.restype = None
    return lib


@lru_cache(maxsize=1)
def load_matmul_add_c2v(verbose: bool = True) -> "MatmulKernel":
    so = _compile("matmul_add_c2v.cpp", "matmul_add_c2v.so", verbose=verbose)
    return MatmulKernel(_make_lib(so), BLOCK_DIM)


@lru_cache(maxsize=1)
def load_add_matmul_v2c(verbose: bool = True) -> "MatmulKernel":
    so = _compile("add_matmul_v2c.cpp", "add_matmul_v2c.so", verbose=verbose)
    return MatmulKernel(_make_lib(so), BLOCK_DIM)


class MatmulKernel:
    def __init__(self, lib: ctypes.CDLL, block_dim: int) -> None:
        self._lib = lib
        self._block_dim = block_dim

    def __call__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        fifo_mem: torch.Tensor,
        batch: int | None = None,
    ) -> None:
        if batch is None:
            batch = A.shape[0]
        stream_ptr = ctypes.c_void_p(torch.npu.current_stream().npu_stream)
        self._lib.call(
            self._block_dim,
            stream_ptr,
            ctypes.c_void_p(A.data_ptr()),
            ctypes.c_void_p(B.data_ptr()),
            ctypes.c_void_p(C.data_ptr()),
            ctypes.c_void_p(D.data_ptr()),
            ctypes.c_void_p(fifo_mem.data_ptr()),
            ctypes.c_int64(batch),
        )
