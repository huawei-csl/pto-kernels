"""
Educational emulation of ``chunk_h_kernel.cpp``.

Mathematics (per sequence, head)
--------------------------------
Same as the C++ header (``WS = W@S``, gated ``K``, ``KV = K̃^T @ V_new``, state update).

Memory / PTO mapping (``chunk_h_kernel.cpp``)
----------------------------------------------
**Cube** tiles (``TileMatL1`` / ``TileAcc``):

- ``s_l1`` ``[D×D]`` — ``TLOAD`` current state from GM workspace / ``FS``.
- ``w_l1`` ``[C×D]`` — ``W`` chunk (``TLOAD`` from BSND).
- ``ws_l0`` ``[C×D]`` fp32 — ``gemm_v0(W, S)``: ``TEXTRACT`` stripes from ``w_l1``/``s_l1`` → L0A/L0B.
- ``k_l1`` ``[D×C]`` — Vec-prepared **scaled** keys (``D×valid`` active columns).
- ``v_l1`` ``[C×D]`` — ``V_new`` chunk.
- ``kv_l0`` ``[D×D]`` fp32 — ``gemm_v0`` with ``transpose_A`` (``K^T @ V`` path).

**Vec** (omitted as fine-grained sync): ``TLOAD`` gates, ``TROWEXPAND``, ``TSUB`` for ``V_new``.

**GM ``workspace`` (Cube ↔ Vec)** — same role as ``chunk_h_kernel`` ``WS_WS`` / ``WS_K`` / ``WS_KV``.
Buffer sizes (fp16 on GM unless noted; ``C`` = chunk size, ``D`` = hidden):

- ``workspace_ws`` **``[C×D]``** fp16 — ``2·C·D`` B → **C·D/512** KiB (Cube→Vec ``WS``).
- ``workspace_k`` **``[D×C]``** fp16 — same numel as ``[C×D]`` → **C·D/512** KiB (Vec→Cube ``K̃``).
- ``workspace_kv`` **``[D×D]``** fp16 — ``2·D²`` B → **D²/512** KiB (Cube→Vec ``KV``).
- Vec UB fp32 staging: ``ws_ub_fp32`` **``[C×D]``** — **C·D/256** KiB; ``kv_ub_fp32`` **``[D×D]``** — **D²/256** KiB (after ``TLOAD`` from workspace).

SRAM tiles are **pre-allocated once at the start of** ``chunk_h_fwd`` and reused for every
sequence, head, and chunk; GM state ``S`` is a single ``[D×D]`` buffer reset with ``zero_()`` per
head. Data paths use helpers in ``_memory.py`` (``TLOAD``/``TFILLPAD``/``TMOV``/``gemm_v0``).

**Index conventions (loops below)** — See ``_common.seq_ranges`` and the "Chunk iteration" section
in ``_common.py``. Here: ``C`` = ``chunk_size``; ``bos``/``eos`` bound one sequence in packed ``T``;
``n_chunks_this_seq = ceil_div(eos - bos, C)``; ``s``/``e`` are the chunk's token span; ``valid`` =
``e - s`` (``< C`` on the last chunk only). ``global_chunk_base`` indexes the leading dimension of
``h_out`` (cumulative chunk count over prior sequences).

Outputs match ``verify_dynamic_bsnd.ref_chunk_h``.
"""

from __future__ import annotations

import torch

from ._common import seq_ranges, total_chunks
from ._memory import (
    alloc_l0_stripes_gemm_v0,
    alloc_l1_cd,
    gemm_v0_accum_fp16,
    tfillpad_k_l1_tail_rows,
    tload_bsnd_chunk_rows_to_l1,
    tload_gm_fp32_dd_to_l1_half,
    tload_workspace_cd_half_to_fp32_ub,
    tload_workspace_dc_half_to_k_l1,
    tload_workspace_dd_half_to_fp32,
    tmov_l1_half_rows,
    tstore_l0c_fp32_to_workspace_cd_half,
    tstore_l0c_fp32_to_workspace_dd_half,
    tstore_vec_ktilde_to_workspace_dc_half,
)


def chunk_h_fwd(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g_cumsum: torch.Tensor,
    chunk_size: int,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns ``(h_states, v_new, final_state)`` as float32 tensors (caller may cast).
    """
    b, t, hd, d = k.shape
    assert b == 1
    device = k.device
    kf, wf, uf, gf = k.float(), w.float(), u.float(), g_cumsum.float()
    ranges = seq_ranges(t, cu_seqlens)
    n_seq = len(ranges)  # number of sequences in the packed batch (1 if no cu_seqlens)
    tc = total_chunks(n_seq, t, chunk_size, cu_seqlens)  # total kernel chunks = h_out.shape[0]
    h_out = torch.zeros(tc, hd, d, d, device=device, dtype=torch.float32)
    v_new = torch.zeros_like(uf)
    final = torch.zeros(n_seq, hd, d, d, device=device, dtype=torch.float32)

    k_tile = 128
    mx = max(chunk_size, d)

    # L1 / L0 tiles — single PTO-style buffer set for the whole forward (overwritten each step)
    # L1 fp16 ``w_l1`` [C×D] — ``2·C·D`` B → **C·D/512** KiB (e.g. **32 KiB** @ C=D=128)
    w_l1 = alloc_l1_cd(chunk_size, d, device=device, dtype=torch.float16)
    # L1 fp16 ``s_l1`` [D×D] — ``2·D²`` B → **D²/256** KiB (e.g. **32 KiB** @ D=128)
    s_l1 = torch.empty((d, d), device=device, dtype=torch.float16)
    # L1 fp16 ``k_l1`` [D×C] — same numel as ``[C×D]`` → **C·D/512** KiB @ fp16
    k_l1 = torch.empty((d, chunk_size), device=device, dtype=torch.float16)
    # L1 fp16 ``v_l1`` [C×D] — **C·D/512** KiB (e.g. **32 KiB** @ C=D=128)
    v_l1 = alloc_l1_cd(chunk_size, d, device=device, dtype=torch.float16)
    # L0C fp32 ``ws_l0`` scratch [C×D] — ``4·C·D`` B → **C·D/256** KiB (e.g. **64 KiB** @ C=D=128)
    l0c_ws = torch.zeros(chunk_size, d, device=device, dtype=torch.float32)
    # L0C fp32 ``kv_l0`` scratch [D×D] — ``4·D²`` B → **D²/128** KiB (e.g. **64 KiB** @ D=128)
    l0c_kv = torch.zeros(d, d, device=device, dtype=torch.float32)
    # L0A/L0B fp16 stripes (``[mx×K_tile]``, ``[K_tile×mx]``) — **mx·K_tile/512** KiB each (e.g. **32 KiB** @ mx=K_tile=128)
    l0a_buf, l0b_buf = alloc_l0_stripes_gemm_v0(
        mx, mx, k_tile, device=device, dtype=torch.float16
    )
    # GM ``S`` fp32 [D×D] — ``4·D²`` B → **D²/128** KiB (e.g. **64 KiB** @ D=128); recurrent state (``zero_()`` per head)
    S = torch.zeros(d, d, device=device, dtype=torch.float32)
    # GM workspace fp16 — Cube ``TSTORE`` / Vec ``TLOAD`` (``chunk_h_kernel`` ``WS_*``); sizes below are **per buffer**
    # ``workspace_ws`` [C×D] — **C·D/512** KiB @ fp16 (e.g. **32 KiB** @ C=D=128)
    workspace_ws = torch.empty(chunk_size, d, device=device, dtype=torch.float16)
    # ``workspace_k`` [D×C] — **C·D/512** KiB @ fp16 (Vec→Cube)
    workspace_k = torch.empty(d, chunk_size, device=device, dtype=torch.float16)
    # ``workspace_kv`` [D×D] — **D²/512** KiB @ fp16 (e.g. **32 KiB** @ D=128)
    workspace_kv = torch.empty(d, d, device=device, dtype=torch.float16)
    # Vec UB fp32 — ``TLOAD`` from ``workspace_ws`` / ``workspace_kv`` (**C·D/256** KiB and **D²/256** KiB)
    ws_ub_fp32 = torch.zeros(chunk_size, d, device=device, dtype=torch.float32)
    kv_ub_fp32 = torch.zeros(d, d, device=device, dtype=torch.float32)

    # Row index into h_out[:, h, :, :] — advances by n_chunks_this_seq after each sequence.
    global_chunk_base = 0
    for seq_idx, (bos, eos) in enumerate(ranges):
        # Tokens for this sequence live at packed indices [bos, eos). Split into C-wide tiles.
        n_tokens = eos - bos
        n_chunks_this_seq = (n_tokens + chunk_size - 1) // chunk_size  # ceil_div(n_tokens, C)
        for h in range(hd):
            S.zero_()  # recurrent state S is per (sequence, head), not shared across chunks
            for chunk_idx in range(n_chunks_this_seq):
                # Chunk `chunk_idx`: token range [s, e) ⊆ [bos, eos); last chunk may have e-s < C.
                s = bos + chunk_idx * chunk_size
                e = min(bos + (chunk_idx + 1) * chunk_size, eos)
                valid = e - s  # active rows in [C×D] L1 tiles (TFILLPAD fills the rest with 0)
                gc = gf[0, s:e, h]
                gl = gc[valid - 1]  # g at last token of chunk (scalar); used in K̃ scaling and S update

                h_out[global_chunk_base + chunk_idx, h] = S.clone()

                # ── GEMM 1: ``WS = W @ S`` ──
                tload_bsnd_chunk_rows_to_l1(
                    w_l1,
                    wf[0],
                    token_start=s,
                    valid_rows=valid,
                    head_idx=h,
                    hidden_size=d,
                )
                tfillpad_k_l1_tail_rows(w_l1, valid_rows=valid, chunk_size=chunk_size)
                tload_gm_fp32_dd_to_l1_half(s_l1, S)
                ws_l0 = gemm_v0_accum_fp16(
                    w_l1,
                    s_l1,
                    l0c_out=l0c_ws,
                    l0a_buf=l0a_buf,
                    l0b_buf=l0b_buf,
                )
                # Cube→Vec: ``TSTORE`` ``WS`` L0C → GM ``workspace_ws``; Vec ``TLOAD`` → UB → ``v_new = U - WS``
                tstore_l0c_fp32_to_workspace_cd_half(
                    workspace_ws, ws_l0, nrows=valid, ncols=d
                )
                tload_workspace_cd_half_to_fp32_ub(
                    ws_ub_fp32, workspace_ws, valid_rows=valid, ncols=d
                )
                vc = uf[0, s:e, h, :] - ws_ub_fp32[:valid, :]
                v_new[0, s:e, h, :] = vc

                # ── GEMM 2: ``KV = K̃^T @ V`` with ``k_l1`` ``[D×C]``, ``v_l1`` ``[C×D]`` ──
                kt = kf[0, s:e, h, :] * torch.exp(gl - gc)[:, None]
                # Vec→Cube: ``TSTORE`` ``K̃`` → ``workspace_k``; Cube ``TLOAD`` → ``k_l1``
                tstore_vec_ktilde_to_workspace_dc_half(
                    workspace_k, kt, valid_cols=valid
                )
                tload_workspace_dc_half_to_k_l1(
                    k_l1, workspace_k, valid_cols=valid
                )
                tmov_l1_half_rows(v_l1, vc.half(), valid_rows=valid)
                tfillpad_k_l1_tail_rows(v_l1, valid_rows=valid, chunk_size=chunk_size)
                kv_l0 = gemm_v0_accum_fp16(
                    k_l1,
                    v_l1,
                    l0c_out=l0c_kv,
                    l0a_buf=l0a_buf,
                    l0b_buf=l0b_buf,
                )
                # Cube→Vec: ``TSTORE`` ``KV`` → ``workspace_kv``; Vec ``TLOAD`` for ``S += KV``
                tstore_l0c_fp32_to_workspace_dd_half(
                    workspace_kv, kv_l0, d=d
                )
                tload_workspace_dd_half_to_fp32(kv_ub_fp32, workspace_kv, d=d)
                S = torch.exp(gl) * S + kv_ub_fp32
            final[seq_idx, h] = S
        global_chunk_base += n_chunks_this_seq

    return h_out, v_new, final


def chunk_h_fwd_explained(*args, **kwargs):
    return chunk_h_fwd(*args, **kwargs)
