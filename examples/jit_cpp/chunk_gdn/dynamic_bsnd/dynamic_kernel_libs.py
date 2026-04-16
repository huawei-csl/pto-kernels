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

_HERE = os.path.dirname(os.path.abspath(__file__))


def _cpp_mtime(name: str) -> int:
    return os.stat(os.path.join(_HERE, name)).st_mtime_ns


@lru_cache(maxsize=None)
def _compile_and_load(cpp_name: str, so_stem: str, *, num_heads: int,
                      hidden_size: int = 128, chunk_size: int = 128,
                      cpp_mtime_ns: int = 0):
    lib_path = compile_pto_kernel(
        cpp_name, f"{so_stem}.so",
        num_heads=num_heads, hidden_size=hidden_size, chunk_size=chunk_size,
        cpp_mtime_ns=cpp_mtime_ns,
    )
    return ctypes.CDLL(os.path.abspath(lib_path))


def _load(cpp_name, so_stem, *, num_heads, hidden_size=128, chunk_size=128):
    return _compile_and_load(
        cpp_name, so_stem,
        num_heads=num_heads, hidden_size=hidden_size, chunk_size=chunk_size,
        cpp_mtime_ns=_cpp_mtime(cpp_name),
    )


def _vp(t):
    return ctypes.c_void_p(t.data_ptr()) if t is not None else ctypes.c_void_p()


# ---------- chunk_cumsum ----------
def load_chunk_cumsum(num_heads: int, chunk_size: int = 128):
    lib = _load("chunk_cumsum_kernel.cpp", "chunk_cumsum_bsnd",
                num_heads=num_heads, chunk_size=chunk_size)
    lib.call_kernel.argtypes = [
        ctypes.c_uint32, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64,
    ]
    lib.call_kernel.restype = None
    return lib


def run_chunk_cumsum(g, g_sum, *, chunk_size=128, cu_seqlens=None,
                     batch_size_override=None, block_dim=None):
    assert g.ndim == 3 and g.dtype == torch.float32
    H = g.shape[2]
    batch = g.shape[0] if batch_size_override is None else batch_size_override
    bd = block_dim or BLOCK_DIM
    lib = load_chunk_cumsum(H, chunk_size)
    stream = torch.npu.current_stream()._as_parameter_
    if cu_seqlens is not None and cu_seqlens.dtype != torch.int32:
        cu_seqlens = cu_seqlens.to(torch.int32)
    lib.call_kernel(bd, stream, _vp(g), _vp(g_sum), _vp(cu_seqlens), batch, g.shape[1])


# ---------- scaled_dot_kkt ----------
def load_scaled_dot_kkt(num_heads: int, hidden_size: int = 128, chunk_size: int = 128):
    lib = _load("scaled_dot_kkt_kernel.cpp", "scaled_dot_kkt_bsnd",
                num_heads=num_heads, hidden_size=hidden_size, chunk_size=chunk_size)
    lib.call_kernel.argtypes = [
        ctypes.c_uint32, ctypes.c_void_p,
    ] + [ctypes.c_void_p] * 7 + [ctypes.c_int64, ctypes.c_int64]
    lib.call_kernel.restype = None
    return lib


def run_scaled_dot_kkt(k, beta, g_sum, mask, workspace, A_out, *,
                       chunk_size=128, cu_seqlens=None,
                       batch_size_override=None, block_dim=None):
    assert k.ndim == 4
    H, D = k.shape[2], k.shape[3]
    batch = k.shape[0] if batch_size_override is None else batch_size_override
    bd = block_dim or BLOCK_DIM
    lib = load_scaled_dot_kkt(H, D, chunk_size)
    stream = torch.npu.current_stream()._as_parameter_
    if cu_seqlens is not None and cu_seqlens.dtype != torch.int32:
        cu_seqlens = cu_seqlens.to(torch.int32)
    g_t = g_sum.reshape(-1, g_sum.shape[-1]).permute(1, 0).contiguous()
    beta_t = beta.reshape(-1, beta.shape[-1]).permute(1, 0).contiguous()
    workspace = torch.zeros((bd * 2, chunk_size, chunk_size),
                            device=k.device, dtype=torch.float16)
    torch.npu.current_stream().synchronize()
    lib.call_kernel(bd, stream,
                    _vp(k), _vp(beta_t), _vp(g_t), _vp(mask),
                    _vp(workspace), _vp(A_out), _vp(cu_seqlens),
                    batch, k.shape[1])


# ---------- wy_fast ----------
def load_wy_fast(num_heads: int, hidden_size: int = 128, chunk_size: int = 128):
    lib = _load("wy_fast_kernel.cpp", "wy_fast_bsnd",
                num_heads=num_heads, hidden_size=hidden_size, chunk_size=chunk_size)
    lib.call_kernel.argtypes = [
        ctypes.c_uint32, ctypes.c_void_p,
    ] + [ctypes.c_void_p] * 10 + [ctypes.c_int64, ctypes.c_int64]
    lib.call_kernel.restype = None
    return lib


def run_wy_fast(k, v, beta, g_sum, A, w_out, u_out, *,
                chunk_size=128, cu_seqlens=None,
                batch_size_override=None, block_dim=None):
    assert k.ndim == 4
    H, D, C = k.shape[2], k.shape[3], chunk_size
    batch = k.shape[0] if batch_size_override is None else batch_size_override
    bd = block_dim or BLOCK_DIM
    lib = load_wy_fast(H, D, C)
    stream = torch.npu.current_stream()._as_parameter_
    if cu_seqlens is not None and cu_seqlens.dtype != torch.int32:
        cu_seqlens = cu_seqlens.to(torch.int32)
    workspace_a1 = torch.zeros((bd, C, C), device=k.device, dtype=torch.float16)
    workspace_a2 = torch.zeros_like(workspace_a1)
    torch.npu.current_stream().synchronize()
    lib.call_kernel(bd, stream,
                    _vp(k), _vp(v), _vp(beta), _vp(g_sum), _vp(A),
                    _vp(workspace_a1), _vp(workspace_a2),
                    _vp(w_out), _vp(u_out), _vp(cu_seqlens),
                    batch, k.shape[1])


# ---------- chunk_h ----------
def load_chunk_h(num_heads: int, hidden_size: int = 128, chunk_size: int = 128):
    lib = _load("chunk_h_kernel.cpp", "chunk_h_bsnd",
                num_heads=num_heads, hidden_size=hidden_size, chunk_size=chunk_size)
    lib.call_kernel.argtypes = [
        ctypes.c_uint32, ctypes.c_void_p,
    ] + [ctypes.c_void_p] * 9 + [ctypes.c_int64, ctypes.c_int64]
    lib.call_kernel.restype = None
    return lib


def run_chunk_h(k, w, u, g_sum, s_out, v_out, fs_out, *,
                chunk_size=128, cu_seqlens=None,
                batch_size_override=None, block_dim=None):
    assert k.ndim == 4
    H, D = k.shape[2], k.shape[3]
    batch = k.shape[0] if batch_size_override is None else batch_size_override
    bd = block_dim or BLOCK_DIM
    lib = load_chunk_h(H, D, chunk_size)
    stream = torch.npu.current_stream()._as_parameter_
    if cu_seqlens is not None and cu_seqlens.dtype != torch.int32:
        cu_seqlens = cu_seqlens.to(torch.int32)
    workspace = torch.zeros((bd * 4, D, D), device=k.device, dtype=torch.float16)
    torch.npu.current_stream().synchronize()
    lib.call_kernel(bd, stream,
                    _vp(k), _vp(w), _vp(u), _vp(g_sum),
                    _vp(s_out), _vp(v_out), _vp(fs_out),
                    _vp(workspace), _vp(cu_seqlens),
                    batch, k.shape[1])


# ---------- chunk_o ----------
def load_chunk_o(num_heads: int, hidden_size: int = 128, chunk_size: int = 128):
    lib = _load("chunk_o_kernel.cpp", "chunk_o_bsnd",
                num_heads=num_heads, hidden_size=hidden_size, chunk_size=chunk_size)
    lib.call_kernel.argtypes = [
        ctypes.c_uint32, ctypes.c_void_p,
    ] + [ctypes.c_void_p] * 11 + [ctypes.c_int64, ctypes.c_int64]
    lib.call_kernel.restype = None
    return lib


def run_chunk_o(q, k, v, s, g_sum, mask, o_out, *,
                chunk_size=128, cu_seqlens=None,
                batch_size_override=None, block_dim=None):
    assert q.ndim == 4
    H, D, C = q.shape[2], q.shape[3], chunk_size
    batch = q.shape[0] if batch_size_override is None else batch_size_override
    bd = block_dim or BLOCK_DIM
    lib = load_chunk_o(H, D, C)
    stream = torch.npu.current_stream()._as_parameter_
    if cu_seqlens is not None and cu_seqlens.dtype != torch.int32:
        cu_seqlens = cu_seqlens.to(torch.int32)
    g_t = g_sum.reshape(-1, g_sum.shape[-1]).permute(1, 0).contiguous()
    workspace_qk = torch.zeros((bd, C, C), device=q.device, dtype=torch.float16)
    workspace_qs_qkv = torch.zeros((bd, C, D), device=q.device, dtype=torch.float16)
    workspace_qk_gated = torch.zeros((bd, C, C), device=q.device, dtype=torch.float16)
    torch.npu.current_stream().synchronize()
    lib.call_kernel(bd, stream,
                    _vp(q), _vp(k), _vp(v), _vp(s), _vp(g_t), _vp(mask),
                    _vp(workspace_qk), _vp(workspace_qs_qkv), _vp(workspace_qk_gated),
                    _vp(o_out), _vp(cu_seqlens),
                    batch, q.shape[1])


def total_chunks(batch_size, seq_len, chunk_size, cu_seqlens=None):
    if cu_seqlens is None:
        return batch_size * ((seq_len + chunk_size - 1) // chunk_size)
    cu = cu_seqlens.cpu().tolist()
    return sum((cu[i + 1] - cu[i] + chunk_size - 1) // chunk_size
               for i in range(len(cu) - 1))
