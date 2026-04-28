from __future__ import annotations

import ctypes
import os
from functools import lru_cache

import torch

from pto_dynamic_common import (
    BLOCK_DIM,
    compile_pto_kernel,
    optional_torch_to_ctypes,
)

_HERE = os.path.dirname(os.path.abspath(__file__))


def _cpp_mtime(name: str) -> int:
    return os.stat(os.path.join(_HERE, name)).st_mtime_ns


@lru_cache(maxsize=None)
def _compile_and_load(
    cpp_name: str,
    so_stem: str,
    *,
    num_heads: int,
    hidden_size: int = 128,
    chunk_size: int = 128,
    key_heads: int | None = None,
    cpp_mtime_ns: int = 0,
):
    lib_path = compile_pto_kernel(
        cpp_name,
        f"{so_stem}.so",
        num_heads=num_heads,
        hidden_size=hidden_size,
        chunk_size=chunk_size,
        key_heads=key_heads,
        cpp_mtime_ns=cpp_mtime_ns,
    )
    return ctypes.CDLL(os.path.abspath(lib_path))


def _load(cpp_name, so_stem, *, num_heads, hidden_size=128, chunk_size=128,
          key_heads=None):
    return _compile_and_load(
        cpp_name,
        so_stem,
        num_heads=num_heads,
        hidden_size=hidden_size,
        chunk_size=chunk_size,
        key_heads=key_heads,
        cpp_mtime_ns=_cpp_mtime(cpp_name),
    )


def _vp(t):
    return ctypes.c_void_p(t.data_ptr()) if t is not None else ctypes.c_void_p()


def _transpose_g(g_sum):
    return g_sum.squeeze(0).t().contiguous()


def total_chunks(batch_size, seq_len, chunk_size, cu_seqlens=None):
    if cu_seqlens is None:
        return batch_size * ((seq_len + chunk_size - 1) // chunk_size)
    cu = cu_seqlens.cpu().tolist()
    return sum((cu[i + 1] - cu[i] + chunk_size - 1) // chunk_size
               for i in range(len(cu) - 1))


def load_chunk_h(
    num_heads: int,
    hidden_size: int = 128,
    chunk_size: int = 128,
    *,
    key_heads: int | None = None,
):
    lib = _load(
        "chunk_h_kernel.cpp",
        "chunk_h_bsnd_groupvalue",
        num_heads=num_heads,
        hidden_size=hidden_size,
        chunk_size=chunk_size,
        key_heads=key_heads,
    )
    lib.call_kernel.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
    ] + [ctypes.c_void_p] * 9 + [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
    lib.call_kernel.restype = None
    return lib


def run_chunk_h(
    k,
    w,
    u,
    g_sum,
    s_out,
    v_out,
    fs_out,
    *,
    stream,
    g_t,
    chunk_size=128,
    cu_seqlens=None,
    batch_size_override=None,
    block_dim=None,
    key_heads: int | None = None,
):
    """
    ``k``: [B, T, Hg, D]; ``w``, ``u``: [B, T, H, D] with ``H % Hg == 0``.
    Gates ``g_sum`` / ``g_t`` are per **value** head (H), same as Triton FLA.
    """
    assert k.ndim == 4
    hg = k.shape[2]
    kh = key_heads if key_heads is not None else hg
    assert hg == kh, f"k head dim {hg} must match key_heads {kh}"
    H = w.shape[2]
    assert H % kh == 0, f"H={H} must be divisible by Hg={kh}"
    D = k.shape[3]
    batch = k.shape[0] if batch_size_override is None else batch_size_override
    bd = block_dim or BLOCK_DIM
    lib = load_chunk_h(H, D, chunk_size, key_heads=kh)
    if cu_seqlens is not None and cu_seqlens.dtype != torch.int32:
        cu_seqlens = cu_seqlens.to(torch.int32)
    workspace = torch.zeros((bd * 4, D, D), device=k.device, dtype=torch.float16)
    T = g_sum.shape[1]
    lib.call_kernel(
        bd,
        stream,
        _vp(k),
        _vp(w),
        _vp(u),
        _vp(g_t),
        _vp(s_out),
        _vp(v_out),
        _vp(fs_out),
        _vp(workspace),
        _vp(cu_seqlens),
        batch,
        k.shape[1],
        T,
    )
