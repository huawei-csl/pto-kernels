"""JIT utilities for naive_separate kernels.

Both kernels (matmul_add_c2v and add_matmul_v2c) live in a single
naive_separate.cpp and expose two entry points:
    call_matmul_add_c2v  and  call_add_matmul_v2c

Workspace sizing:
    Unlike the pipelined variants that use a small FIFO buffer, the naive
    baseline needs one slot per round (no double-buffering).  Slot size is
    TILE_SIZE × TILE_SIZE fp16, and there are num_rounds = batch/(BLOCK_DIM*T)
    rounds per core.  So:
        workspace[batch, TILE_SIZE]  fp16   (== same shape as A, C, D)
"""
from __future__ import annotations

import ctypes
import os
import subprocess
from functools import lru_cache

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_CPP  = os.path.join(_HERE, "naive_separate.cpp")
_SO   = os.path.join(_HERE, "naive_separate.so")

ASCEND_TOOLKIT_HOME = os.environ.get("ASCEND_TOOLKIT_HOME") or os.environ.get(
    "ASCEND_HOME_PATH", ""
)
if not ASCEND_TOOLKIT_HOME:
    raise RuntimeError("Set ASCEND_TOOLKIT_HOME or ASCEND_HOME_PATH")

_PTO_INC    = os.path.join(ASCEND_TOOLKIT_HOME, "include")
_DRIVER_INC = "/usr/local/Ascend/driver/kernel/inc"

_NPU_DEVICE = os.environ.get("NPU_DEVICE", "npu:7")
try:
    BLOCK_DIM = int(
        getattr(torch.npu.get_device_properties(_NPU_DEVICE), "cube_core_num", 20)
    )
except (RuntimeError, AssertionError):
    BLOCK_DIM = 24

TILE_SIZE = 128


# ── Compilation ───────────────────────────────────────────────────────────────

def _compile(verbose: bool = True) -> str:
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

    cmd = ["bisheng", *flags, _CPP, "-o", _SO]
    if verbose:
        print("Compiling:", " ".join(cmd))
    subprocess.run(cmd, check=True, timeout=300)
    if verbose:
        print(f"Compiled → {_SO}")
    return _SO


def _bind(lib: ctypes.CDLL, fn_name: str) -> None:
    """Bind the call signature for a kernel entry point."""
    fn = getattr(lib, fn_name)
    fn.argtypes = [
        ctypes.c_uint32,   # block_dim
        ctypes.c_void_p,   # stream
        ctypes.c_void_p,   # A
        ctypes.c_void_p,   # B
        ctypes.c_void_p,   # C
        ctypes.c_void_p,   # D
        ctypes.c_void_p,   # workspace
        ctypes.c_int64,    # batch
    ]
    fn.restype = None


@lru_cache(maxsize=1)
def _load_so(verbose: bool = True) -> ctypes.CDLL:
    so = _compile(verbose=verbose)
    lib = ctypes.CDLL(os.path.abspath(so))
    _bind(lib, "call_matmul_add_c2v")
    _bind(lib, "call_add_matmul_v2c")
    return lib


# ── Kernel wrapper ────────────────────────────────────────────────────────────

class NaiveKernel:
    """Callable wrapper for a naive_separate entry point.

    Parameters
    ----------
    lib       : loaded shared library
    fn_name   : symbol name ("call_matmul_add_c2v" or "call_add_matmul_v2c")
    block_dim : number of Cube cores
    """

    def __init__(self, lib: ctypes.CDLL, fn_name: str, block_dim: int) -> None:
        self._fn = getattr(lib, fn_name)
        self._block_dim = block_dim

    def __call__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        workspace: torch.Tensor,
        batch: int | None = None,
    ) -> None:
        """Launch the kernel in-place (result written to C).

        All tensors must be on the same NPU device, contiguous, and fp16.
        workspace must be at least [batch, TILE_SIZE] fp16.
        """
        if batch is None:
            batch = A.shape[0]
        stream_ptr = ctypes.c_void_p(torch.npu.current_stream().npu_stream)
        self._fn(
            self._block_dim,
            stream_ptr,
            ctypes.c_void_p(A.data_ptr()),
            ctypes.c_void_p(B.data_ptr()),
            ctypes.c_void_p(C.data_ptr()),
            ctypes.c_void_p(D.data_ptr()),
            ctypes.c_void_p(workspace.data_ptr()),
            ctypes.c_int64(batch),
        )


def load_matmul_add_c2v(verbose: bool = True) -> NaiveKernel:
    lib = _load_so(verbose=verbose)
    return NaiveKernel(lib, "call_matmul_add_c2v", BLOCK_DIM)


def load_add_matmul_v2c(verbose: bool = True) -> NaiveKernel:
    lib = _load_so(verbose=verbose)
    return NaiveKernel(lib, "call_add_matmul_v2c", BLOCK_DIM)
