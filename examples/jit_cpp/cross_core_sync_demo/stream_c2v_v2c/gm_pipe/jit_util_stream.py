"""JIT utilities for stream_c2v_v2c/gm_pipe kernels.

The gm_pipe variant uses the GlobalData TPOP/TALLOC/TFREE APIs which are
available in pto-isa-master but not in the default installed library.
This utility compiles against /workdir/pto-isa-master/include.

C2V:  TALLOC + TSTORE(slot_half, c_l0) + TPUSH  →  half slot (32 KB/core/slot)
      explicit fp32→fp16 via hardware FIX unit, same slot size as raw_flag
V2C:  TALLOC + TSTORE(slot_half, a_ub) + TPUSH  →  half slot (32 KB/core/slot)
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

# gm_pipe uses the newer pto-isa-master headers for GlobalData TPOP/TALLOC/TFREE.
_PTO_NEW_INC = "/workdir/pto-isa-master/include"
_DRIVER_INC  = "/usr/local/Ascend/driver/kernel/inc"

_NPU_DEVICE = os.environ.get("NPU_DEVICE", "npu:7")
try:
    BLOCK_DIM = int(
        getattr(torch.npu.get_device_properties(_NPU_DEVICE), "cube_core_num", 20)
    )
except (RuntimeError, AssertionError):
    BLOCK_DIM = 24

TILE_SIZE  = 128
FIFO_DEPTH = 2

# Both C2V and V2C use half slots in gm_pipe.
C2V_FIFO_ELEMS_PER_CORE = FIFO_DEPTH * TILE_SIZE * TILE_SIZE   # float16 elements
V2C_FIFO_ELEMS_PER_CORE = FIFO_DEPTH * TILE_SIZE * TILE_SIZE   # float16 elements


def _compile(cpp_basename: str, so_basename: str, verbose: bool = True) -> str:
    flags = [
        "-fPIC", "-shared", "-xcce", "-DMEMORY_BASE", "-O2", "-std=gnu++17",
        "--cce-aicore-arch=dav-c220",
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-record-overflow=true",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-Wno-macro-redefined", "-Wno-ignored-attributes",
        # pto-isa-master FIRST (provides GlobalData TPOP/TALLOC/TFREE APIs)
        f"-I{_PTO_NEW_INC}",
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
        print("Compiling (with pto-isa-master headers):", " ".join(cmd))
    subprocess.run(cmd, check=True, timeout=300)
    if verbose:
        print(f"Compiled → {so}")
    return so


@lru_cache(maxsize=1)
def load_stream_c2v(verbose: bool = True) -> "StreamC2VKernel":
    so = _compile("stream_c2v.cpp", "stream_c2v.so", verbose=verbose)
    lib = ctypes.CDLL(os.path.abspath(so))
    lib.call_stream_c2v.argtypes = [
        ctypes.c_uint32,  # block_dim
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # A
        ctypes.c_void_p,  # B
        ctypes.c_void_p,  # fifo_mem  (float16, half slot)
        ctypes.c_int32,   # num_iters
    ]
    lib.call_stream_c2v.restype = None
    return StreamC2VKernel(lib, BLOCK_DIM)


@lru_cache(maxsize=1)
def load_stream_v2c(verbose: bool = True) -> "StreamV2CKernel":
    so = _compile("stream_v2c.cpp", "stream_v2c.so", verbose=verbose)
    lib = ctypes.CDLL(os.path.abspath(so))
    lib.call_stream_v2c.argtypes = [
        ctypes.c_uint32,  # block_dim
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # A
        ctypes.c_void_p,  # D
        ctypes.c_void_p,  # fifo_mem  (float16, half slot)
        ctypes.c_int32,   # num_iters
    ]
    lib.call_stream_v2c.restype = None
    return StreamV2CKernel(lib, BLOCK_DIM)


class StreamC2VKernel:
    def __init__(self, lib: ctypes.CDLL, block_dim: int) -> None:
        self._lib = lib
        self._block_dim = block_dim

    def __call__(self, A: torch.Tensor, B: torch.Tensor,
                 fifo_mem: torch.Tensor, num_iters: int) -> None:
        """A: [BLOCK_DIM*T, T] fp16; B: [T, T] fp16;
        fifo_mem: [BLOCK_DIM * C2V_FIFO_ELEMS_PER_CORE] float16 (half slot)."""
        stream_ptr = ctypes.c_void_p(torch.npu.current_stream().npu_stream)
        self._lib.call_stream_c2v(
            self._block_dim, stream_ptr,
            ctypes.c_void_p(A.data_ptr()),
            ctypes.c_void_p(B.data_ptr()),
            ctypes.c_void_p(fifo_mem.data_ptr()),
            ctypes.c_int32(num_iters),
        )


class StreamV2CKernel:
    def __init__(self, lib: ctypes.CDLL, block_dim: int) -> None:
        self._lib = lib
        self._block_dim = block_dim

    def __call__(self, A: torch.Tensor, D: torch.Tensor,
                 fifo_mem: torch.Tensor, num_iters: int) -> None:
        """A, D: [num_iters*BLOCK_DIM*T, T] fp16;
        fifo_mem: [BLOCK_DIM * V2C_FIFO_ELEMS_PER_CORE] fp16."""
        stream_ptr = ctypes.c_void_p(torch.npu.current_stream().npu_stream)
        self._lib.call_stream_v2c(
            self._block_dim, stream_ptr,
            ctypes.c_void_p(A.data_ptr()),
            ctypes.c_void_p(D.data_ptr()),
            ctypes.c_void_p(fifo_mem.data_ptr()),
            ctypes.c_int32(num_iters),
        )
