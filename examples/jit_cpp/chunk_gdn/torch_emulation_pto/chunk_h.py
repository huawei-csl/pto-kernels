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

SRAM tiles are **pre-allocated once at the start of** ``chunk_h_fwd`` and reused for every
sequence, head, and chunk; GM state ``S`` is a single ``[D×D]`` buffer reset with ``zero_()`` per
head. Data paths use helpers in ``_memory.py`` (``TLOAD``/``TFILLPAD``/``TMOV``/``gemm_v0``).

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
    tmov_l1_half_dc_cols,
    tmov_l1_half_rows,
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
    n_seq = len(ranges)
    tc = total_chunks(n_seq, t, chunk_size, cu_seqlens)
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

    ci_base = 0
    for si, (bos, eos) in enumerate(ranges):
        nc = (eos - bos + chunk_size - 1) // chunk_size
        for h in range(hd):
            S.zero_()
            for ci in range(nc):
                s, e = bos + ci * chunk_size, min(bos + (ci + 1) * chunk_size, eos)
                valid = e - s
                gc = gf[0, s:e, h]
                gl = gc[e - s - 1]

                h_out[ci_base + ci, h] = S.clone()

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

                vc = uf[0, s:e, h, :] - ws_l0[:valid, :]
                v_new[0, s:e, h, :] = vc

                # ── GEMM 2: ``KV = K̃^T @ V`` with ``k_l1`` ``[D×C]``, ``v_l1`` ``[C×D]`` ──
                kt = kf[0, s:e, h, :] * torch.exp(gl - gc)[:, None]
                tmov_l1_half_dc_cols(k_l1, kt, valid_cols=valid)
                tmov_l1_half_rows(v_l1, vc.half(), valid_rows=valid)
                tfillpad_k_l1_tail_rows(v_l1, valid_rows=valid, chunk_size=chunk_size)
                kv_l0 = gemm_v0_accum_fp16(
                    k_l1,
                    v_l1,
                    l0c_out=l0c_kv,
                    l0a_buf=l0a_buf,
                    l0b_buf=l0b_buf,
                )

                S = torch.exp(gl) * S + kv_l0
            final[si, h] = S
        ci_base += nc

    return h_out, v_new, final


def chunk_h_fwd_explained(*args, **kwargs):
    return chunk_h_fwd(*args, **kwargs)
