"""
Educational emulation of ``chunk_o_kernel.cpp``.

Mathematics (per chunk)
-----------------------
Three Cube GEMMs (``q_l1``, ``k_l1``, ``s_l1``, ``qk_gated_l1``, ``v_l1``) plus Vec gating.

Memory / PTO mapping (``chunk_o_kernel.cpp``)
---------------------------------------------
**Cube**

1. ``TLOAD`` ``Q``, ``K`` → ``q_l1``, ``k_l1`` ``[C×D]``; ``TFILLPAD`` tail rows.
2. ``TMATMUL`` ``QK = Q @ K^T`` → ``qk_l0`` ``[C×C]`` fp32; **Cube** ``TSTORE`` → GM ``workspace_qk_raw`` fp16.
3. ``TLOAD`` ``S`` ``[D×D]`` → ``s_l1``.
4. ``TMATMUL`` ``QS = Q @ S`` → ``qs_l0`` ``[C×D]`` (stays in L0C / UB for Vec blend; not the ``QK`` workspace path).
5. **Vec** ``TLOAD`` raw ``QK`` GM → UB fp32 ``qk_vec_ub``; gate + mask in UB; ``TSTORE`` gated tile → GM
   ``workspace_qk_gated``; **Cube** ``TLOAD`` → ``qk_gated_l1``.
6. ``TLOAD`` ``V`` → ``v_l1`` (``QK_gated`` already in L1 from workspace).
7. ``TMATMUL`` ``QKV = QK_gated @ V`` → ``qkv_l0`` ``[C×D]``.

**Vec** applies ``exp(min(Δg,0))`` gate and causal mask (PTO recipe).

SRAM **L1 / L0** tiles are pre-allocated once at the start of ``chunk_o_fwd`` / ``chunk_o_fwd_fla``
and reused for every sequence, head, and chunk; data movement uses ``_memory`` helpers
(``TLOAD``/``TFILLPAD``/``tmov_*``/``gemm_v0``).

**GM workspace (Cube ↔ Vec)** — two fp16 **``[C×C]``** tiles: ``workspace_qk_raw`` (Cube→Vec raw ``QK``) and
``workspace_qk_gated`` (Vec→Cube after gate+mask). Each: ``2·C²`` B → **C²/512** KiB (e.g. **32 KiB** @ C=128);
**total** **C²/256** KiB for both (e.g. **64 KiB** @ C=128).

Global tensors
--------------
``q``, ``k``, ``v``: ``[B, T, H, D]``; ``h_states``: ``[num_chunks, H, D, D]``; ``g_cumsum``: ``[B, T, H]``.

**Index conventions** — same packed-time / chunk tiling as ``chunk_h_fwd`` (see ``_common.seq_ranges``):
``(bos, eos)`` per sequence; ``n_chunks_this_seq = ceil_div(eos - bos, C)``; ``s``, ``e``, ``vlen`` for
the current chunk; ``global_chunk_base`` indexes ``h_states`` and advances after each sequence.
"""

from __future__ import annotations

import torch

from ._common import seq_ranges
from ._memory import (
    alloc_l0_stripes_gemm_v0,
    alloc_l1_cd,
    gemm_v0_accum_fp16,
    tfillpad_k_l1_tail_rows,
    tload,
    tload_bsnd_rows,
    tload_gm_fp32_dd_to_l1_half,
    tstore,
    tstore_l0c_flat,
)


def _qk_gate_pto(gc: torch.Tensor) -> torch.Tensor:
    """PTO Vec: ``exp(min(Δg, 0))`` — ``verify_dynamic_bsnd._qk_gate_pto``."""
    d = gc[:, None] - gc[None, :]
    return torch.exp(torch.minimum(d, torch.zeros_like(d)))


def _vec_apply_qk_gate_chunk_o(
    workspace_qk_gated: torch.Tensor,
    workspace_qk_raw: torch.Tensor,
    qk_vec_ub_fp32: torch.Tensor,
    gate: torch.Tensor,
    mask: torch.Tensor,
    *,
    vlen: int,
) -> None:
    """
    ``chunk_o`` only — Vec path with explicit ``tload`` / ``tstore`` (no direct GM tensor indexing).

    1. ``TLOAD`` — ``workspace_qk_raw`` (GM fp16) → ``qk_vec_ub_fp32`` (UB fp32) top ``[vlen×vlen]``.
    2. Vec multiply — gate + causal mask in UB.
    3. ``TSTORE`` — gated UB tile → ``workspace_qk_gated`` (GM fp16) top ``[vlen×vlen]``.
    """
    tload(
        qk_vec_ub_fp32,
        workspace_qk_raw,
        direction="gm_to_ub",
        nrows=vlen,
        ncols=vlen,
    )
    sub = qk_vec_ub_fp32[:vlen, :vlen]
    sub.mul_(gate.to(dtype=sub.dtype))
    sub.mul_(mask.to(dtype=sub.dtype))
    tstore(
        workspace_qk_gated,
        qk_vec_ub_fp32,
        direction="ub_to_gm",
        nrows=vlen,
        ncols=vlen,
    )


def chunk_o_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h_states: torch.Tensor,
    g_cumsum: torch.Tensor,
    chunk_size: int,
    cu_seqlens: torch.LongTensor | None = None,
) -> torch.Tensor:
    """
    Parameters
    ----------
    h_states :
        ``[num_chunks, H, D, D]`` — pre-chunk snapshots (row ``chunk_idx`` is ``S`` **before** that chunk).
    """
    b, t, hd, d = q.shape
    assert b == 1
    device = q.device
    o = torch.zeros_like(q, dtype=torch.float32)
    qf, kf, vf, gf = q.float(), k.float(), v.float(), g_cumsum.float()
    ranges = seq_ranges(t, cu_seqlens)
    global_chunk_base = 0  # row into h_states for the first chunk of the current sequence
    k_tile = 128
    mx = max(chunk_size, d)

    # L1 fp16 ``q_l1`` / ``k_l1`` / ``v_l1`` [C×D] each — ``2·C·D`` B → **C·D/512** KiB each (e.g. **32 KiB** @ C=D=128)
    q_l1 = alloc_l1_cd(chunk_size, d, device=device, dtype=torch.float16)
    k_l1 = alloc_l1_cd(chunk_size, d, device=device, dtype=torch.float16)
    # L1 fp16 ``s_l1`` [D×D] — ``2·D²`` B → **D²/256** KiB (e.g. **32 KiB** @ D=128)
    s_l1 = torch.empty((d, d), device=device, dtype=torch.float16)
    # L1 fp16 ``qk_gated_l1`` [C×C] — ``2·C²`` B → **C²/256** KiB (e.g. **32 KiB** @ C=128)
    qk_gated_l1 = torch.empty(
        (chunk_size, chunk_size), device=device, dtype=torch.float16
    )
    v_l1 = alloc_l1_cd(chunk_size, d, device=device, dtype=torch.float16)
    # L0C fp32 ``qk_l0`` [C×C] — ``4·C²`` B → **C²/128** KiB (e.g. **64 KiB** @ C=128)
    l0c_qk = torch.zeros(chunk_size, chunk_size, device=device, dtype=torch.float32)
    # L0C fp32 ``qs_l0`` / ``qkv_l0`` [C×D] (time-shared) — ``4·C·D`` B → **C·D/256** KiB (e.g. **64 KiB** @ C=D=128)
    l0c_qs_qkv = torch.zeros(chunk_size, d, device=device, dtype=torch.float32)
    # L0A/L0B fp16 stripes — **mx·K_tile/512** KiB each (e.g. **32 KiB** @ mx=K_tile=128)
    l0a_buf, l0b_buf = alloc_l0_stripes_gemm_v0(
        mx, mx, k_tile, device=device, dtype=torch.float16
    )
    # GM ``workspace`` fp16 [C×C] each — **C²/512** KiB per buffer (Cube↔Vec ``QK``; ``chunk_o_kernel``)
    workspace_qk_raw = torch.empty(
        chunk_size, chunk_size, device=device, dtype=torch.float16
    )
    workspace_qk_gated = torch.empty(
        chunk_size, chunk_size, device=device, dtype=torch.float16
    )
    # Vec UB fp32 ``[C×C]`` — ``TLOAD`` raw ``QK`` from GM before gate+mask; **C²/256** KiB @ fp32
    qk_vec_ub_fp32 = torch.zeros(
        chunk_size, chunk_size, device=device, dtype=torch.float32
    )

    for bos, eos in ranges:
        n_tokens = eos - bos
        n_chunks_this_seq = (n_tokens + chunk_size - 1) // chunk_size
        for h in range(hd):
            for chunk_idx in range(n_chunks_this_seq):
                s = bos + chunk_idx * chunk_size
                e = min(bos + (chunk_idx + 1) * chunk_size, eos)
                vlen = e - s  # valid Q/K/V rows; causal mask is vlen×vlen
                gc = gf[0, s:e, h]

                tload_bsnd_rows(
                    q_l1,
                    qf[0],
                    token_start=s,
                    valid_rows=vlen,
                    head_idx=h,
                    hidden_size=d,
                )
                tload_bsnd_rows(
                    k_l1,
                    kf[0],
                    token_start=s,
                    valid_rows=vlen,
                    head_idx=h,
                    hidden_size=d,
                )
                tfillpad_k_l1_tail_rows(q_l1, valid_rows=vlen, chunk_size=chunk_size)
                tfillpad_k_l1_tail_rows(k_l1, valid_rows=vlen, chunk_size=chunk_size)

                # GEMM 1: ``Q @ K^T``
                qk_l0 = gemm_v0_accum_fp16(
                    q_l1,
                    k_l1,
                    transpose_b=True,
                    k_tile=k_tile,
                    l0c_out=l0c_qk,
                    l0a_buf=l0a_buf,
                    l0b_buf=l0b_buf,
                )

                S = h_states[global_chunk_base + chunk_idx, h]
                tload_gm_fp32_dd_to_l1_half(s_l1, S)
                qs_l0 = gemm_v0_accum_fp16(
                    q_l1,
                    s_l1,
                    k_tile=k_tile,
                    l0c_out=l0c_qs_qkv,
                    l0a_buf=l0a_buf,
                    l0b_buf=l0b_buf,
                )
                inter = qs_l0[:vlen, :] * torch.exp(gc)[:, None]

                gate = _qk_gate_pto(gc)
                mask = torch.arange(vlen, device=device)[:, None] >= torch.arange(
                    vlen, device=device
                )[None, :]
                # Cube→Vec: ``TSTORE`` ``QK`` L0C → ``workspace_qk_raw``; Vec gate+mask → ``workspace_qk_gated``; Cube ``TLOAD`` → L1
                tstore_l0c_flat(
                    workspace_qk_raw,
                    qk_l0,
                    chunk_square=chunk_size * chunk_size,
                )
                _vec_apply_qk_gate_chunk_o(
                    workspace_qk_gated,
                    workspace_qk_raw,
                    qk_vec_ub_fp32,
                    gate,
                    mask,
                    vlen=vlen,
                )
                tload(
                    qk_gated_l1,
                    workspace_qk_gated,
                    direction="gm_to_l1",
                    nrows=vlen,
                    ncols=vlen,
                )

                tload_bsnd_rows(
                    v_l1,
                    vf[0],
                    token_start=s,
                    valid_rows=vlen,
                    head_idx=h,
                    hidden_size=d,
                )
                tfillpad_k_l1_tail_rows(v_l1, valid_rows=vlen, chunk_size=chunk_size)

                qkv_l0 = gemm_v0_accum_fp16(
                    qk_gated_l1,
                    v_l1,
                    k_tile=k_tile,
                    l0c_out=l0c_qs_qkv,
                    l0a_buf=l0a_buf,
                    l0b_buf=l0b_buf,
                )
                o[0, s:e, h, :] = inter[:vlen, :] + qkv_l0[:vlen, :]
        global_chunk_base += n_chunks_this_seq
    return o.to(dtype=q.dtype)


def chunk_o_fwd_explained(*args, **kwargs):
    return chunk_o_fwd(*args, **kwargs)


def chunk_o_fwd_fla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h_states: torch.Tensor,
    g_cumsum: torch.Tensor,
    chunk_size: int,
    cu_seqlens: torch.LongTensor | None = None,
) -> torch.Tensor:
    """
    Optional: Triton / FLA-style ``safe_exp`` on the QK gate (see ``ref_chunk_o_fla``).
    """
    from ._common import safe_exp_torch

    b, t, hd, d = q.shape
    o = torch.zeros_like(q, dtype=torch.float32)
    qf, kf, vf, gf = q.float(), k.float(), v.float(), g_cumsum.float()
    ranges = seq_ranges(t, cu_seqlens)
    global_chunk_base = 0  # same indexing as ``chunk_o_fwd``
    k_tile = 128
    mx = max(chunk_size, d)
    dev = q.device

    # L1 fp16 ``q_l1`` / ``k_l1`` / ``v_l1`` [C×D] each — **C·D/512** KiB each (e.g. **32 KiB** @ C=D=128)
    q_l1 = alloc_l1_cd(chunk_size, d, device=dev, dtype=torch.float16)
    k_l1 = alloc_l1_cd(chunk_size, d, device=dev, dtype=torch.float16)
    # L1 fp16 ``s_l1`` [D×D] — **D²/256** KiB (e.g. **32 KiB** @ D=128)
    s_l1 = torch.empty((d, d), device=dev, dtype=torch.float16)
    # L1 fp16 ``qk_gated_l1`` [C×C] — **C²/256** KiB (e.g. **32 KiB** @ C=128)
    qk_gated_l1 = torch.empty((chunk_size, chunk_size), device=dev, dtype=torch.float16)
    v_l1 = alloc_l1_cd(chunk_size, d, device=dev, dtype=torch.float16)
    # L0C fp32 [C×C] — **C²/128** KiB (e.g. **64 KiB** @ C=128)
    l0c_qk = torch.zeros(chunk_size, chunk_size, device=dev, dtype=torch.float32)
    # L0C fp32 [C×D] (QS / QKV time-shared) — **C·D/256** KiB (e.g. **64 KiB** @ C=D=128)
    l0c_qs_qkv = torch.zeros(chunk_size, d, device=dev, dtype=torch.float32)
    # L0A/L0B fp16 stripes — **mx·K_tile/512** KiB each (e.g. **32 KiB** @ mx=K_tile=128)
    l0a_buf, l0b_buf = alloc_l0_stripes_gemm_v0(
        mx, mx, k_tile, device=dev, dtype=torch.float16
    )
    # GM ``workspace`` fp16 [C×C] each — **C²/512** KiB per buffer (same as ``chunk_o_fwd``)
    workspace_qk_raw = torch.empty(
        chunk_size, chunk_size, device=dev, dtype=torch.float16
    )
    workspace_qk_gated = torch.empty(
        chunk_size, chunk_size, device=dev, dtype=torch.float16
    )
    qk_vec_ub_fp32 = torch.zeros(chunk_size, chunk_size, device=dev, dtype=torch.float32)

    for bos, eos in ranges:
        n_tokens = eos - bos
        n_chunks_this_seq = (n_tokens + chunk_size - 1) // chunk_size
        for h in range(hd):
            for chunk_idx in range(n_chunks_this_seq):
                s = bos + chunk_idx * chunk_size
                e = min(bos + (chunk_idx + 1) * chunk_size, eos)
                vlen = e - s
                gc = gf[0, s:e, h]

                tload_bsnd_rows(
                    q_l1,
                    qf[0],
                    token_start=s,
                    valid_rows=vlen,
                    head_idx=h,
                    hidden_size=d,
                )
                tload_bsnd_rows(
                    k_l1,
                    kf[0],
                    token_start=s,
                    valid_rows=vlen,
                    head_idx=h,
                    hidden_size=d,
                )
                tfillpad_k_l1_tail_rows(q_l1, valid_rows=vlen, chunk_size=chunk_size)
                tfillpad_k_l1_tail_rows(k_l1, valid_rows=vlen, chunk_size=chunk_size)

                qk_l0 = gemm_v0_accum_fp16(
                    q_l1,
                    k_l1,
                    transpose_b=True,
                    k_tile=k_tile,
                    l0c_out=l0c_qk,
                    l0a_buf=l0a_buf,
                    l0b_buf=l0b_buf,
                )

                S = h_states[global_chunk_base + chunk_idx, h]
                tload_gm_fp32_dd_to_l1_half(s_l1, S)
                qs_l0 = gemm_v0_accum_fp16(
                    q_l1,
                    s_l1,
                    k_tile=k_tile,
                    l0c_out=l0c_qs_qkv,
                    l0a_buf=l0a_buf,
                    l0b_buf=l0b_buf,
                )
                inter = qs_l0[:vlen, :] * torch.exp(gc)[:, None]

                gate = safe_exp_torch(gc[:, None] - gc[None, :])
                mask = torch.arange(vlen, device=q.device)[:, None] >= torch.arange(
                    vlen, device=q.device
                )[None, :]
                tstore_l0c_flat(
                    workspace_qk_raw,
                    qk_l0,
                    chunk_square=chunk_size * chunk_size,
                )
                _vec_apply_qk_gate_chunk_o(
                    workspace_qk_gated,
                    workspace_qk_raw,
                    qk_vec_ub_fp32,
                    gate,
                    mask,
                    vlen=vlen,
                )
                tload(
                    qk_gated_l1,
                    workspace_qk_gated,
                    direction="gm_to_l1",
                    nrows=vlen,
                    ncols=vlen,
                )

                tload_bsnd_rows(
                    v_l1,
                    vf[0],
                    token_start=s,
                    valid_rows=vlen,
                    head_idx=h,
                    hidden_size=d,
                )
                tfillpad_k_l1_tail_rows(v_l1, valid_rows=vlen, chunk_size=chunk_size)

                qkv_l0 = gemm_v0_accum_fp16(
                    qk_gated_l1,
                    v_l1,
                    k_tile=k_tile,
                    l0c_out=l0c_qs_qkv,
                    l0a_buf=l0a_buf,
                    l0b_buf=l0b_buf,
                )
                o[0, s:e, h, :] = inter[:vlen, :] + qkv_l0[:vlen, :]
        global_chunk_base += n_chunks_this_seq
    return o.to(dtype=q.dtype)
