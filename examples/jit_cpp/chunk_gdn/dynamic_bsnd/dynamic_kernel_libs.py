from __future__ import annotations

import ctypes
import os
from functools import lru_cache

import torch

from pto_dynamic_common import (
    BLOCK_DIM,
    compile_pto_kernel,
    optional_torch_to_ctypes,
    torch_to_ctypes,
)


@lru_cache(maxsize=None)
def chunk_cumsum_kernel(num_heads: int, chunk_size: int):
    lib_path = compile_pto_kernel(
        "chunk_cumsum_kernel.cpp",
        "chunk_cumsum_dynamic_bsnd.so",
        num_heads=num_heads,
        chunk_size=chunk_size,
    )
    lib = ctypes.CDLL(os.path.abspath(lib_path))
    lib.call_kernel.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    lib.call_kernel.restype = None
    return lib


def run_chunk_cumsum_kernel(
    g: torch.Tensor,
    out: torch.Tensor,
    *,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None = None,
    batch_size_override: int | None = None,
    block_dim: int | None = None,
):
    if g.ndim != 3:
        raise ValueError("g must be [B,S,H]")
    if g.dtype != torch.float32:
        raise TypeError("g must be float32")
    if cu_seqlens is not None:
        if cu_seqlens.dtype != torch.int32:
            raise TypeError("cu_seqlens must be int32")
        if not cu_seqlens.is_contiguous():
            cu_seqlens = cu_seqlens.contiguous()
    num_heads = g.shape[2]
    batch_size = g.shape[0] if batch_size_override is None else batch_size_override
    if block_dim is None:
        block_dim = BLOCK_DIM
    lib = chunk_cumsum_kernel(num_heads, chunk_size)
    stream = torch.npu.current_stream()._as_parameter_
    g_c = g.contiguous()
    lib.call_kernel(
        block_dim,
        stream,
        torch_to_ctypes(g_c),
        torch_to_ctypes(out),
        optional_torch_to_ctypes(cu_seqlens),
        batch_size,
        g.shape[1],
    )
