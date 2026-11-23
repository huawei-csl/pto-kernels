"""JIT compile + load utility for matmul_add_c2v kernel.

Compiles matmul_add_c2v.cpp with bisheng and returns a Python callable that
invokes the kernel via ctypes.

Usage:
    from jit_util_matmul_add_c2v import compile_and_load, BLOCK_DIM
    kernel = compile_and_load()
    kernel(A, B, C, D, workspace, batch)
"""
from __future__ import annotations

import ctypes
import os
import subprocess
from functools import lru_cache

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_CPP_FILE = os.path.join(_HERE, "matmul_add_c2v.cpp")
_LIB_FILE = os.path.join(_HERE, "matmul_add_c2v.so")

ASCEND_TOOLKIT_HOME = os.environ.get("ASCEND_TOOLKIT_HOME") or os.environ.get(
    "ASCEND_HOME_PATH", ""
)
if not ASCEND_TOOLKIT_HOME:
    raise RuntimeError("Set ASCEND_TOOLKIT_HOME or ASCEND_HOME_PATH")

PTO_LIB_PATH = os.environ.get("PTO_LIB_PATH", ASCEND_TOOLKIT_HOME)
_PTO_INC = os.path.join(PTO_LIB_PATH, "include")
_DRIVER_INC = "/usr/local/Ascend/driver/kernel/inc"

# Determine the number of Cube cores on the current NPU.
_NPU_DEVICE = os.environ.get("NPU_DEVICE", "npu:7")
try:
    BLOCK_DIM = int(
        getattr(torch.npu.get_device_properties(_NPU_DEVICE), "cube_core_num", 20)
    )
except (RuntimeError, AssertionError):
    BLOCK_DIM = 24  # 910B2 default


def _compile(verbose: bool = True) -> str:
    """Compile the kernel and return the .so path."""
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
        f"-I{_PTO_INC}",
        f"-I{ASCEND_TOOLKIT_HOME}/include",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc/runtime",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc/profiling",
    ]
    if os.path.isdir(_DRIVER_INC):
        flags.append(f"-I{_DRIVER_INC}")

    cmd = ["bisheng", *flags, _CPP_FILE, "-o", _LIB_FILE]
    if verbose:
        print("Compiling:", " ".join(cmd))
    subprocess.run(cmd, check=True, timeout=300)
    if verbose:
        print(f"Compiled → {_LIB_FILE}")
    return _LIB_FILE


def _load_lib(lib_path: str) -> ctypes.CDLL:
    """Load the .so and bind the `call` symbol."""
    lib = ctypes.CDLL(os.path.abspath(lib_path))
    # void call(uint32_t block_dim, void *stream,
    #            uint8_t *A, uint8_t *B, uint8_t *C,
    #            uint8_t *D, uint8_t *workspace, int64_t batch)
    lib.call.argtypes = [
        ctypes.c_uint32,   # block_dim
        ctypes.c_void_p,   # stream
        ctypes.c_void_p,   # A
        ctypes.c_void_p,   # B
        ctypes.c_void_p,   # C
        ctypes.c_void_p,   # D
        ctypes.c_void_p,   # workspace
        ctypes.c_int64,    # batch
    ]
    lib.call.restype = None
    return lib


@lru_cache(maxsize=1)
def compile_and_load(verbose: bool = True) -> "MatmulAddC2VKernel":
    """Compile (if needed) and return a callable kernel wrapper."""
    lib_path = _compile(verbose=verbose)
    lib = _load_lib(lib_path)
    return MatmulAddC2VKernel(lib, BLOCK_DIM)


class MatmulAddC2VKernel:
    """Callable wrapper around the matmul_add_c2v ctypes kernel.

    Parameters
    ----------
    lib       : ctypes.CDLL loaded from the compiled .so
    block_dim : number of Cube cores to launch (== num physical cores)
    """

    def __init__(self, lib: ctypes.CDLL, block_dim: int) -> None:
        self._lib = lib
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
        workspace must be at least [block_dim * TILE_SIZE, TILE_SIZE] fp16.
        """
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
            ctypes.c_void_p(workspace.data_ptr()),
            ctypes.c_int64(batch),
        )
