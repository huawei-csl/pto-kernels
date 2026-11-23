"""JIT utilities for matmul_add/pushpop kernels.

matmul_add_c2v:  C = A @ B + D   (pushpop, float slot)
  - D input: float32  (must match VecTileFloat dtype from TPOP)
  - C output: float16
  - fifo_mem: float32

add_matmul_v2c:  C = (A + B) @ D   (pushpop, half slot)
  - all float16; identical types to raw_flag
  - fifo_mem: float16
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

PTO_LIB_PATH = os.environ.get("PTO_LIB_PATH", ASCEND_TOOLKIT_HOME)
_PTO_INC  = os.path.join(PTO_LIB_PATH, "include")
_DRIVER_INC = "/usr/local/Ascend/driver/kernel/inc"

_NPU_DEVICE = os.environ.get("NPU_DEVICE", "npu:7")
try:
    BLOCK_DIM = int(
        getattr(torch.npu.get_device_properties(_NPU_DEVICE), "cube_core_num", 20)
    )
except (RuntimeError, AssertionError):
    BLOCK_DIM = 24

TILE_SIZE  = 128
# C2V: FIFO_DEPTH=1 workaround — see PTO_API_BUGS.md Bug 1
# With FIFO_DEPTH=2 and TILE_UP_DOWN, the tileIndex desync breaks multi-round.
# FIFO_DEPTH=1 forces SyncPeriod=1 (strict alternation), fixing multi-round for C2V.
C2V_FIFO_DEPTH = 1
C2V_FIFO_ELEMS_PER_CORE = C2V_FIFO_DEPTH * TILE_SIZE * TILE_SIZE   # float32 elements

# V2C: FIFO_DEPTH=2 needed — with FIFO_DEPTH=1, only 1 free signal is seeded.
# Both Vec sub-blocks call allocate() independently; sub-block 1 deadlocks if
# the single free signal is already consumed by sub-block 0.
# V2C is therefore scoped to num_rounds=1 (single-round correctness).
V2C_FIFO_DEPTH = 2
V2C_FIFO_ELEMS_PER_CORE = V2C_FIFO_DEPTH * TILE_SIZE * TILE_SIZE   # float16 elements


def _compile(cpp_basename: str, so_basename: str, verbose: bool = True) -> str:
    flags = [
        "-fPIC", "-shared", "-xcce", "-DMEMORY_BASE", "-O2", "-std=gnu++17",
        "--cce-aicore-arch=dav-c220",
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-record-overflow=true",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-Wno-macro-redefined", "-Wno-ignored-attributes",
        f"-I{_PTO_INC}",
        f"-I{ASCEND_TOOLKIT_HOME}/include",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc/runtime",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc/profiling",
    ]
    if os.path.isdir(_DRIVER_INC):
        flags.append(f"-I{_DRIVER_INC}")
    cpp = os.path.join(_HERE, cpp_basename)
    so  = os.path.join(_HERE, so_basename)
    cmd = ["bisheng", *flags, cpp, "-o", so]
    if verbose:
        print("Compiling:", " ".join(cmd))
    subprocess.run(cmd, check=True, timeout=300)
    if verbose:
        print(f"Compiled → {so}")
    return so


@lru_cache(maxsize=1)
def load_matmul_add_c2v(verbose: bool = True) -> "MatmulAddC2VKernel":
    so  = _compile("matmul_add_c2v.cpp", "matmul_add_c2v.so", verbose=verbose)
    lib = ctypes.CDLL(os.path.abspath(so))
    lib.call.argtypes = [
        ctypes.c_uint32,  # block_dim
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # A  (fp16)
        ctypes.c_void_p,  # B  (fp16)
        ctypes.c_void_p,  # C  (fp32 — float FIFO slot, no implicit fp32→fp16)
        ctypes.c_void_p,  # D  (fp32)
        ctypes.c_void_p,  # fifo_mem (fp32)
        ctypes.c_int64,   # batch
    ]
    lib.call.restype = None
    return MatmulAddC2VKernel(lib, BLOCK_DIM)


@lru_cache(maxsize=1)
def load_add_matmul_v2c(verbose: bool = True) -> "AddMatmulV2CKernel":
    so  = _compile("add_matmul_v2c.cpp", "add_matmul_v2c.so", verbose=verbose)
    lib = ctypes.CDLL(os.path.abspath(so))
    lib.call.argtypes = [
        ctypes.c_uint32,  # block_dim
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # A  (fp16)
        ctypes.c_void_p,  # B  (fp16)
        ctypes.c_void_p,  # C  (fp16 output)
        ctypes.c_void_p,  # D  (fp16)
        ctypes.c_void_p,  # fifo_mem (fp16)
        ctypes.c_int64,   # batch
    ]
    lib.call.restype = None
    return AddMatmulV2CKernel(lib, BLOCK_DIM)


class MatmulAddC2VKernel:
    def __init__(self, lib: ctypes.CDLL, block_dim: int) -> None:
        self._lib = lib
        self._block_dim = block_dim

    def __call__(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                 D: torch.Tensor, fifo_mem: torch.Tensor,
                 batch: int | None = None) -> None:
        """D must be float32; C is float32; A,B fp16; fifo_mem float32."""
        if batch is None:
            batch = A.shape[0]
        stream_ptr = ctypes.c_void_p(torch.npu.current_stream().npu_stream)
        self._lib.call(
            self._block_dim, stream_ptr,
            ctypes.c_void_p(A.data_ptr()),
            ctypes.c_void_p(B.data_ptr()),
            ctypes.c_void_p(C.data_ptr()),
            ctypes.c_void_p(D.data_ptr()),
            ctypes.c_void_p(fifo_mem.data_ptr()),
            ctypes.c_int64(batch),
        )


class AddMatmulV2CKernel:
    def __init__(self, lib: ctypes.CDLL, block_dim: int) -> None:
        self._lib = lib
        self._block_dim = block_dim

    def __call__(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                 D: torch.Tensor, fifo_mem: torch.Tensor,
                 batch: int | None = None) -> None:
        """All tensors fp16; fifo_mem fp16."""
        if batch is None:
            batch = A.shape[0]
        stream_ptr = ctypes.c_void_p(torch.npu.current_stream().npu_stream)
        self._lib.call(
            self._block_dim, stream_ptr,
            ctypes.c_void_p(A.data_ptr()),
            ctypes.c_void_p(B.data_ptr()),
            ctypes.c_void_p(C.data_ptr()),
            ctypes.c_void_p(D.data_ptr()),
            ctypes.c_void_p(fifo_mem.data_ptr()),
            ctypes.c_int64(batch),
        )
