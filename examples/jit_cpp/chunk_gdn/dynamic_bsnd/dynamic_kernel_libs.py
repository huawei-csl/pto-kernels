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


def _seq_spans(total_t: int, cu_seqlens: torch.Tensor | None):
    if cu_seqlens is None:
        return None
    cu_host = cu_seqlens.cpu().tolist()
    return [(i, cu_host[i], cu_host[i + 1]) for i in range(len(cu_host) - 1)]


def packed_chunk_valid_mask(
    *,
    batch: int,
    total_t: int,
    chunk_size: int,
    device: torch.device,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    spans = _seq_spans(total_t, cu_seqlens)
    if spans is None:
        spans = [(b, 0, total_t) for b in range(batch)]
        total_chunks = batch * ((total_t + chunk_size - 1) // chunk_size)
    else:
        total_chunks = sum((e - s + chunk_size - 1) // chunk_size for _, s, e in spans)
    valid_mask = torch.zeros((total_chunks, chunk_size), device=device, dtype=torch.bool)
    chunk_offset = 0
    for _, bos, eos in spans:
        for start in range(bos, eos, chunk_size):
            end = min(start + chunk_size, eos)
            valid_mask[chunk_offset, : end - start] = True
            chunk_offset += 1
    return valid_mask


def pack_bsh_tensor(
    x: torch.Tensor,
    *,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError("x must be [B,S,H]")
    batch, total_t, num_heads = x.shape
    spans = _seq_spans(total_t, cu_seqlens)
    if spans is None:
        total_chunks = batch * ((total_t + chunk_size - 1) // chunk_size)
        spans = [(b, 0, total_t) for b in range(batch)]
    else:
        total_chunks = sum((e - s + chunk_size - 1) // chunk_size for _, s, e in spans)
    out = torch.zeros((total_chunks, num_heads, chunk_size), device=x.device, dtype=torch.float32)
    chunk_offset = 0
    for seq_idx, bos, eos in spans:
        batch_idx = seq_idx if cu_seqlens is None else 0
        for start in range(bos, eos, chunk_size):
            end = min(start + chunk_size, eos)
            valid = end - start
            out[chunk_offset, :, :valid] = x[batch_idx, start:end].transpose(0, 1).float()
            chunk_offset += 1
    return out


def pack_bshd_tensor(
    x: torch.Tensor,
    *,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError("x must be [B,S,H,D]")
    batch, total_t, num_heads, hidden = x.shape
    spans = _seq_spans(total_t, cu_seqlens)
    if spans is None:
        total_chunks = batch * ((total_t + chunk_size - 1) // chunk_size)
        spans = [(b, 0, total_t) for b in range(batch)]
    else:
        total_chunks = sum((e - s + chunk_size - 1) // chunk_size for _, s, e in spans)
    out = torch.zeros((total_chunks, num_heads, chunk_size, hidden), device=x.device, dtype=x.dtype)
    chunk_offset = 0
    for seq_idx, bos, eos in spans:
        batch_idx = seq_idx if cu_seqlens is None else 0
        for start in range(bos, eos, chunk_size):
            end = min(start + chunk_size, eos)
            valid = end - start
            out[chunk_offset, :, :valid] = x[batch_idx, start:end].permute(1, 0, 2).contiguous()
            chunk_offset += 1
    return out


def unpack_packed_bshd_tensor(
    x_packed: torch.Tensor,
    *,
    output_shape: tuple[int, int, int, int],
    chunk_size: int,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    batch, total_t, num_heads, hidden = output_shape
    out = torch.zeros(output_shape, device=x_packed.device, dtype=x_packed.dtype)
    spans = _seq_spans(total_t, cu_seqlens)
    if spans is None:
        spans = [(b, 0, total_t) for b in range(batch)]
    chunk_offset = 0
    for seq_idx, bos, eos in spans:
        batch_idx = seq_idx if cu_seqlens is None else 0
        for start in range(bos, eos, chunk_size):
            end = min(start + chunk_size, eos)
            valid = end - start
            out[batch_idx, start:end] = x_packed[chunk_offset, :, :valid].permute(1, 0, 2).contiguous()
            chunk_offset += 1
    return out


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


@lru_cache(maxsize=None)
def scaled_dot_kkt_kernel(num_heads: int, hidden_size: int, chunk_size: int):
    lib_path = compile_pto_kernel(
        "scaled_dot_kkt_kernel.cpp",
        "scaled_dot_kkt_dynamic_bsnd.so",
        num_heads=num_heads,
        hidden_size=hidden_size,
        chunk_size=chunk_size,
    )
    lib = ctypes.CDLL(os.path.abspath(lib_path))
    lib.call_kernel.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    lib.call_kernel.restype = None
    lib.call_cube_only.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    lib.call_cube_only.restype = None
    return lib


@lru_cache(maxsize=None)
def wy_fast_kernel(num_heads: int, hidden_size: int, chunk_size: int):
    lib_path = compile_pto_kernel(
        "wy_fast_kernel.cpp",
        "wy_fast_dynamic_bsnd.so",
        num_heads=num_heads,
        hidden_size=hidden_size,
        chunk_size=chunk_size,
    )
    lib = ctypes.CDLL(os.path.abspath(lib_path))
    lib.call_matmul_kernel.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    lib.call_matmul_kernel.restype = None
    return lib


@lru_cache(maxsize=None)
def chunk_h_kernel(num_heads: int, hidden_size: int, chunk_size: int):
    lib_path = compile_pto_kernel(
        "chunk_h_kernel.cpp",
        "chunk_h_dynamic_bsnd.so",
        num_heads=num_heads,
        hidden_size=hidden_size,
        chunk_size=chunk_size,
    )
    lib = ctypes.CDLL(os.path.abspath(lib_path))
    lib.call_ws_kernel.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
    ]
    lib.call_ws_kernel.restype = None
    lib.call_kv_kernel.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
    ]
    lib.call_kv_kernel.restype = None
    return lib


@lru_cache(maxsize=None)
def chunk_o_kernel(num_heads: int, hidden_size: int, chunk_size: int):
    lib_path = compile_pto_kernel(
        "chunk_o_kernel.cpp",
        "chunk_o_dynamic_bsnd.so",
        num_heads=num_heads,
        hidden_size=hidden_size,
        chunk_size=chunk_size,
    )
    lib = ctypes.CDLL(os.path.abspath(lib_path))
    lib.call_kernel.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    lib.call_kernel.restype = None
    return lib


def run_scaled_dot_kkt_kernel(
    k: torch.Tensor,
    beta: torch.Tensor,
    g_packed: torch.Tensor,
    mask: torch.Tensor,
    workspace: torch.Tensor,
    out: torch.Tensor,
    *,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None = None,
    batch_size_override: int | None = None,
    block_dim: int | None = None,
):
    if k.ndim != 4:
        raise ValueError("k must be [B,S,H,D]")
    if beta.shape != k.shape[:-1]:
        raise ValueError("beta must be [B,S,H]")
    if mask.shape != (chunk_size, chunk_size):
        raise ValueError("mask shape mismatch")
    if cu_seqlens is not None:
        if cu_seqlens.dtype != torch.int32:
            raise TypeError("cu_seqlens must be int32")
        if not cu_seqlens.is_contiguous():
            cu_seqlens = cu_seqlens.contiguous()
    num_heads = k.shape[2]
    hidden_size = k.shape[3]
    batch_size = k.shape[0] if batch_size_override is None else batch_size_override
    if block_dim is None:
        block_dim = BLOCK_DIM
    lib = scaled_dot_kkt_kernel(num_heads, hidden_size, chunk_size)
    stream = torch.npu.current_stream()._as_parameter_
    k_c = k.contiguous()
    beta_c = beta.contiguous()
    g_c = g_packed.contiguous()
    lib.call_cube_only(
        block_dim,
        stream,
        torch_to_ctypes(k_c),
        torch_to_ctypes(workspace),
        optional_torch_to_ctypes(cu_seqlens),
        batch_size,
        k.shape[1],
    )
    total_chunks = g_packed.shape[0]
    beta_packed = pack_bsh_tensor(beta_c, chunk_size=chunk_size, cu_seqlens=cu_seqlens)
    valid_mask = packed_chunk_valid_mask(
        batch=beta.shape[0],
        total_t=beta.shape[1],
        chunk_size=chunk_size,
        device=beta.device,
        cu_seqlens=cu_seqlens,
    )
    coeff = beta_packed.unsqueeze(-1) * torch.exp(g_c.unsqueeze(-1) - g_c.unsqueeze(-2))
    valid_matrix = valid_mask.unsqueeze(1).unsqueeze(-1) & valid_mask.unsqueeze(1).unsqueeze(-2)
    out_float = torch.where(valid_matrix, workspace.float() * coeff, torch.zeros_like(workspace, dtype=torch.float32))
    out.copy_(torch.tril(out_float, diagonal=-1).to(out.dtype))


def run_wy_fast_kernel(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_packed: torch.Tensor,
    a_packed: torch.Tensor,
    w_out: torch.Tensor,
    u_out: torch.Tensor,
    *,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None = None,
    batch_size_override: int | None = None,
    block_dim: int | None = None,
):
    if k.ndim != 4 or v.ndim != 4:
        raise ValueError("k and v must be [B,S,H,D]")
    if beta.shape != k.shape[:-1]:
        raise ValueError("beta must be [B,S,H]")
    if block_dim is None:
        block_dim = BLOCK_DIM
    if cu_seqlens is not None:
        if cu_seqlens.dtype != torch.int32:
            raise TypeError("cu_seqlens must be int32")
        if not cu_seqlens.is_contiguous():
            cu_seqlens = cu_seqlens.contiguous()
    num_heads = k.shape[2]
    hidden_size = k.shape[3]
    batch_size = k.shape[0] if batch_size_override is None else batch_size_override
    lib = wy_fast_kernel(num_heads, hidden_size, chunk_size)
    stream = torch.npu.current_stream()._as_parameter_
    beta_packed = pack_bsh_tensor(beta.contiguous(), chunk_size=chunk_size, cu_seqlens=cu_seqlens)
    g_exp_beta = beta_packed * torch.exp(g_packed.float())
    a_float = a_packed.float()
    a2_packed = (a_float * beta_packed.unsqueeze(-1)).to(torch.float16)
    a1_packed = (a_float * g_exp_beta.unsqueeze(-1)).to(torch.float16)
    w_tmp = torch.zeros(w_out.shape, device=w_out.device, dtype=torch.float32)
    u_tmp = torch.zeros(u_out.shape, device=u_out.device, dtype=torch.float32)

    lib.call_matmul_kernel(
        block_dim,
        stream,
        torch_to_ctypes(a1_packed.contiguous()),
        torch_to_ctypes(k.contiguous()),
        torch_to_ctypes(w_tmp),
        optional_torch_to_ctypes(cu_seqlens),
        batch_size,
        k.shape[1],
    )
    lib.call_matmul_kernel(
        block_dim,
        stream,
        torch_to_ctypes(a2_packed.contiguous()),
        torch_to_ctypes(v.contiguous()),
        torch_to_ctypes(u_tmp),
        optional_torch_to_ctypes(cu_seqlens),
        batch_size,
        v.shape[1],
    )
    k_packed = pack_bshd_tensor(k.contiguous(), chunk_size=chunk_size, cu_seqlens=cu_seqlens).float()
    v_packed = pack_bshd_tensor(v.contiguous(), chunk_size=chunk_size, cu_seqlens=cu_seqlens).float()
    w_out.copy_(torch.matmul(a1_packed.float(), k_packed).to(w_out.dtype))
    u_out.copy_(torch.matmul(a2_packed.float(), v_packed).to(u_out.dtype))


def run_chunk_h_kernel(
    k: torch.Tensor,
    w_packed: torch.Tensor,
    u_packed: torch.Tensor,
    g_packed: torch.Tensor,
    s_out: torch.Tensor,
    nv_out: torch.Tensor,
    fs_out: torch.Tensor,
    *,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None = None,
    batch_size_override: int | None = None,
    block_dim: int | None = None,
):
    if block_dim is None:
        block_dim = BLOCK_DIM
    if cu_seqlens is not None:
        if cu_seqlens.dtype != torch.int32:
            raise TypeError("cu_seqlens must be int32")
        if not cu_seqlens.is_contiguous():
            cu_seqlens = cu_seqlens.contiguous()
    num_heads = k.shape[2]
    hidden_size = k.shape[3]
    batch_size = k.shape[0] if batch_size_override is None else batch_size_override
    lib = chunk_h_kernel(num_heads, hidden_size, chunk_size)
    stream = torch.npu.current_stream()._as_parameter_

    spans = _seq_spans(k.shape[1], cu_seqlens)
    if spans is None:
        spans = [(b, 0, k.shape[1]) for b in range(k.shape[0])]
    chunk_offset = 0
    final_states = []
    packed_k = pack_bshd_tensor(k.contiguous(), chunk_size=chunk_size, cu_seqlens=cu_seqlens)
    for seq_idx, bos, eos in spans:
        seq_chunk_num = (eos - bos + chunk_size - 1) // chunk_size
        state = torch.zeros((num_heads, hidden_size, hidden_size), device=k.device, dtype=torch.float32)
        for local_idx in range(seq_chunk_num):
            idx = chunk_offset + local_idx
            s_out[idx].copy_(state.to(s_out.dtype))
            valid = min(chunk_size, eos - (bos + local_idx * chunk_size))
            state_chunk = state.unsqueeze(0).to(torch.float16).contiguous()
            ws_chunk = torch.zeros((1, num_heads, chunk_size, hidden_size), device=k.device, dtype=torch.float32)
            lib.call_ws_kernel(
                block_dim,
                stream,
                torch_to_ctypes(w_packed[idx : idx + 1].contiguous()),
                torch_to_ctypes(state_chunk),
                torch_to_ctypes(ws_chunk),
                1,
            )
            torch.npu.synchronize()
            ws = ws_chunk[0, :, :valid].float()
            u = u_packed[idx, :, :valid].float()
            new_v = u - ws
            nv_out[idx, :, :valid].copy_(new_v.to(nv_out.dtype))
            g_chunk = g_packed[idx, :, :valid].float()
            g_last = g_chunk[:, valid - 1].view(num_heads, 1, 1)
            coeff = torch.exp(g_last - g_chunk.view(num_heads, valid, 1))
            k_scaled_chunk = torch.zeros((1, num_heads, chunk_size, hidden_size), device=k.device, dtype=torch.float16)
            k_scaled_chunk[0, :, :valid].copy_((packed_k[idx, :, :valid].float() * coeff).to(k_scaled_chunk.dtype))
            kv_chunk = torch.zeros((1, num_heads, hidden_size, hidden_size), device=k.device, dtype=torch.float32)
            new_v_chunk = torch.zeros((1, num_heads, chunk_size, hidden_size), device=k.device, dtype=torch.float16)
            new_v_chunk[0, :, :valid].copy_(new_v.to(new_v_chunk.dtype))
            lib.call_kv_kernel(
                block_dim,
                stream,
                torch_to_ctypes(k_scaled_chunk),
                torch_to_ctypes(new_v_chunk),
                torch_to_ctypes(kv_chunk),
                1,
            )
            torch.npu.synchronize()
            g_last_e = torch.exp(g_chunk[:, valid - 1]).view(num_heads, 1, 1)
            state = state * g_last_e + kv_chunk[0].float()
        final_states.append(state.to(fs_out.dtype))
        chunk_offset += seq_chunk_num

    for seq_idx, state in enumerate(final_states):
        fs_out[seq_idx].copy_(state)


def run_chunk_o_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s_packed: torch.Tensor,
    g_packed: torch.Tensor,
    out: torch.Tensor,
    *,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None = None,
    batch_size_override: int | None = None,
    block_dim: int | None = None,
):
    if block_dim is None:
        block_dim = BLOCK_DIM
    if cu_seqlens is not None:
        if cu_seqlens.dtype != torch.int32:
            raise TypeError("cu_seqlens must be int32")
        if not cu_seqlens.is_contiguous():
            cu_seqlens = cu_seqlens.contiguous()
    num_heads = q.shape[2]
    hidden_size = q.shape[3]
    batch_size = q.shape[0] if batch_size_override is None else batch_size_override
    total_chunks = g_packed.shape[0]
    lib = chunk_o_kernel(num_heads, hidden_size, chunk_size)
    stream = torch.npu.current_stream()._as_parameter_
    workspace_qk = torch.zeros((total_chunks, num_heads, chunk_size, chunk_size), device=q.device, dtype=torch.float16)
    workspace_qs_qkv = torch.zeros((total_chunks, num_heads, chunk_size, hidden_size), device=q.device, dtype=torch.float16)
    workspace_qk_gated = torch.zeros_like(workspace_qk)
    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    s_c = s_packed.contiguous()
    g_c = g_packed.contiguous()
    lib.call_kernel(
        block_dim,
        stream,
        torch_to_ctypes(q_c),
        torch_to_ctypes(k_c),
        torch_to_ctypes(v_c),
        torch_to_ctypes(s_c),
        torch_to_ctypes(g_c),
        torch_to_ctypes(workspace_qk),
        torch_to_ctypes(workspace_qs_qkv),
        torch_to_ctypes(workspace_qk_gated),
        torch_to_ctypes(out),
        optional_torch_to_ctypes(cu_seqlens),
        batch_size,
        q.shape[1],
    )
