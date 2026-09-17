"""ctypes loaders for the A5 Cube/Vector sync demo."""

from __future__ import annotations

import ctypes
import os
from functools import lru_cache
from pathlib import Path

import torch

from common_build import build

_DEVICE = os.environ.get("NPU_DEVICE", "npu:0")


def _block_dim() -> int:
    try:
        return int(getattr(torch.npu.get_device_properties(_DEVICE), "cube_core_num", 20))
    except (RuntimeError, AssertionError):
        return 32


BLOCK_DIM = _block_dim()
TILE_SIZE = 128


def _stream_ptr() -> ctypes.c_void_p:
    return ctypes.c_void_p(torch.npu.current_stream().npu_stream)


@lru_cache(maxsize=1)
def load_lib(verbose: bool = True) -> ctypes.CDLL:
    lib_path = build()
    if verbose:
        print(f"Loaded {lib_path}")
    lib = ctypes.CDLL(str(Path(lib_path).resolve()))

    lib.cv_stream_c2v.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_void_p,
    ]
    lib.cv_stream_c2v.restype = None

    lib.cv_stream_v2c.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_void_p,
    ]
    lib.cv_stream_v2c.restype = None

    for name in ("cv_matmul_add_c2v", "cv_add_matmul_v2c"):
        fn = getattr(lib, name)
        fn.argtypes = [
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_void_p,
        ]
        fn.restype = None
    return lib


class CvSyncKernels:
    def __init__(self, verbose: bool = True) -> None:
        self.lib = load_lib(verbose=verbose)
        self.block_dim = BLOCK_DIM

    def stream_c2v(self, A: torch.Tensor, B: torch.Tensor, num_iters: int) -> None:
        self.lib.cv_stream_c2v(
            self.block_dim,
            ctypes.c_void_p(A.data_ptr()),
            ctypes.c_void_p(B.data_ptr()),
            ctypes.c_int32(num_iters),
            _stream_ptr(),
        )

    def stream_v2c(self, A: torch.Tensor, D: torch.Tensor, num_iters: int) -> None:
        self.lib.cv_stream_v2c(
            self.block_dim,
            ctypes.c_void_p(A.data_ptr()),
            ctypes.c_void_p(D.data_ptr()),
            ctypes.c_int32(num_iters),
            _stream_ptr(),
        )

    def matmul_add_c2v(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor) -> None:
        self.lib.cv_matmul_add_c2v(
            self.block_dim,
            ctypes.c_void_p(A.data_ptr()),
            ctypes.c_void_p(B.data_ptr()),
            ctypes.c_void_p(C.data_ptr()),
            ctypes.c_void_p(D.data_ptr()),
            ctypes.c_int64(A.shape[0]),
            _stream_ptr(),
        )

    def add_matmul_v2c(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor) -> None:
        self.lib.cv_add_matmul_v2c(
            self.block_dim,
            ctypes.c_void_p(A.data_ptr()),
            ctypes.c_void_p(B.data_ptr()),
            ctypes.c_void_p(C.data_ptr()),
            ctypes.c_void_p(D.data_ptr()),
            ctypes.c_int64(A.shape[0]),
            _stream_ptr(),
        )

