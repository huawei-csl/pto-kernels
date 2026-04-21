"""
Educational emulation of ``scaled_dot_kkt_kernel.cpp``.

Mathematics (per sequence, head, chunk)
---------------------------------------
See C++ header. **Python reference** in ``verify_dynamic_bsnd`` uses::

    coeff[i,j] = safe_exp(g_i - g_j) · β_i

with a strict-lower causal mask (not the ``g + log β`` Vec path in the C++ comment block).

Memory / PTO mapping
--------------------
**Cube (``__DAV_C220_CUBE__``)**

1. ``TLOAD`` — ``K`` chunk BSND → ``k_l1`` ``[C×D]`` (``L1Mat`` NZ stand-in = row-major).
2. ``TFILLPAD`` — tail rows if ``valid < C``.
3. ``TRESHAPE`` → ``K^T`` (``transpose_b`` in ``gemm_v0_accum_fp16``), then ``TEXTRACT`` K‑tiles
   into L0A/L0B and ``TMATMUL`` / ``TMATMUL_ACC`` into fp32 ``L0C`` (see ``_memory.tmatmul_kkt_l1_to_l0c``).
4. **Cube→Vec** ``TSTORE`` — ``L0C`` fp32 → fp16 in GM **`workspace_kk`** (same GM channel as ``chunk_o`` / ``chunk_h`` workspace; double-buffer slots ``ci & 1`` on device).

**Vec (``__DAV_C220_VEC__``)**

5. ``TLOAD`` — causal mask stripe, ``G``, ``Beta`` rows into UB (omitted as full-tensor math).
6. ``wait_flag_dev`` / cross-core — not emulated.
7. **Vec** ``TLOAD`` — ``KK^T`` stripe from **`workspace_kk`** → ``a_ub_half`` ``[C/2×C]`` per sub-block (GM→UB).
8. Gating + ``TMUL`` with mask; **Vec** ``TSTORE`` — ``A`` BSND rows (Vec→GM output, not Cube).

``k_l1``, ``l0c_kkt``, L0 stripes, ``workspace_kk``, and ``a_ub_half`` are **pre-allocated once**
at the start of ``scaled_dot_kkt_fwd`` and reused for every sequence, head, and chunk.

**Cube↔Vec** GM buffer: ``workspace_kk`` fp16 **``[C×C]``** — **C²/512** KiB (e.g. **32 KiB** @ C=128); Vec reads stripes into ``a_ub_half`` **``[C/2×C]``** — **C²/1024** KiB.

**Index conventions** — same ``bos``/``eos``/``chunk_start_rel``/``s``/``e``/``valid`` as ``wy_fast_fwd``.
The Vec loop uses ``vid ∈ {0,1}`` to cover ``C/2`` rows per half-chunk stripe; ``row_off = vid * (C/2)``.

Global tensors (Torch layout)
-----------------------------
``k``: ``[B, T, H, D]``; ``beta``, ``g_cumsum``: ``[B, T, H]``; output ``A``: ``[B, T, H, C]``.
"""

from __future__ import annotations

import torch

from ._common import safe_exp_torch, seq_ranges
from ._memory import (
    alloc_l0_stripes_gemm_v0,
    alloc_l1_cd,
    tfillpad_k_l1_tail_rows,
    tload_bsnd_chunk_rows_to_l1,
    tload_workspace_kk_half_to_ub_rows,
    tmatmul_kkt_l1_to_l0c,
    tstore_l0c_to_workspace_kk_half,
    tstore_ub_half_to_gm_a_rows,
)


def scaled_dot_kkt_fwd(
    k: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    chunk_size: int,
    cu_seqlens: torch.LongTensor | None = None,
) -> torch.Tensor:
    """
    Returns ``A`` with shape ``[B, T, H, C]`` in fp32 (cast to fp16 for NPU parity).
    """
    b, t, hd, d = k.shape
    assert b == 1
    device = k.device
    half_c = chunk_size // 2
    out = torch.zeros(b, t, hd, chunk_size, device=device, dtype=torch.float32)
    kf = k.float()
    bf = beta.float()
    gf = g_cumsum.float()
    k_tile = 128
    mx = max(chunk_size, d)

    # L1 fp16 ``k_l1`` [C×D] — **C·D/512** KiB (e.g. **32 KiB** @ C=D=128)
    k_l1 = alloc_l1_cd(chunk_size, d, device=device, dtype=torch.float16)
    # GM ``workspace_kk`` fp16 [C×C] (Cube→Vec ``TSTORE``) — **C²/512** KiB (e.g. **32 KiB** @ C=128)
    workspace_kk = torch.empty(
        chunk_size, chunk_size, device=device, dtype=torch.float16
    )
    # UB fp16 ``a_ub_half`` [C/2×C] — **C²/1024** KiB (e.g. **16 KiB** @ C=128)
    a_ub_half = torch.empty(half_c, chunk_size, device=device, dtype=torch.float16)
    # L0C fp32 ``K K^T`` [C×C] — **C²/128** KiB (e.g. **64 KiB** @ C=128)
    l0c_kkt = torch.zeros(
        chunk_size, chunk_size, device=device, dtype=torch.float32
    )
    # L0A/L0B fp16 stripes — **mx·K_tile/512** KiB each (e.g. **32 KiB** @ mx=K_tile=128)
    l0a_buf, l0b_buf = alloc_l0_stripes_gemm_v0(
        mx, mx, k_tile, device=device, dtype=torch.float16
    )

    for bos, eos in seq_ranges(t, cu_seqlens):
        n_tokens = eos - bos
        for h in range(hd):
            for chunk_start_rel in range(0, n_tokens, chunk_size):
                s = bos + chunk_start_rel
                e = min(s + chunk_size, eos)
                valid = e - s

                # ── Cube: GM → L1 → L0C → **Cube→Vec** ``TSTORE`` ``workspace_kk`` (fp16) ──
                tload_bsnd_chunk_rows_to_l1(
                    k_l1,
                    k[0],
                    token_start=s,
                    valid_rows=valid,
                    head_idx=h,
                    hidden_size=d,
                )
                tfillpad_k_l1_tail_rows(k_l1, valid_rows=valid, chunk_size=chunk_size)

                a_l0_fp32 = tmatmul_kkt_l1_to_l0c(
                    k_l1,
                    k_tile=k_tile,
                    l0c_out=l0c_kkt,
                    l0a_buf=l0a_buf,
                    l0b_buf=l0b_buf,
                )

                tstore_l0c_to_workspace_kk_half(
                    workspace_kk,
                    a_l0_fp32,
                    slot=0,
                    chunk_square=chunk_size * chunk_size,
                )

                # ── Vec: ``TLOAD`` ``workspace_kk`` → UB stripes (two ``vid`` halves), gating, ``TSTORE`` out ──
                gc = gf[0, s:e, h]
                coeff = safe_exp_torch(gc[:, None] - gc[None, :]) * bf[0, s:e, h, None]
                mask_vv = torch.arange(valid, device=device)[:, None] > torch.arange(
                    valid, device=device
                )[None, :]
                for vid in (0, 1):
                    row_off = vid * half_c
                    local_valid = min(max(valid - row_off, 0), half_c)
                    if local_valid <= 0:
                        continue
                    tload_workspace_kk_half_to_ub_rows(
                        a_ub_half,
                        workspace_kk,
                        row_begin=row_off,
                        n_rows=local_valid,
                        chunk_size=chunk_size,
                    )
                    cstripe = coeff[row_off : row_off + local_valid, :valid]
                    mstripe = mask_vv[row_off : row_off + local_valid, :]
                    gated = (
                        a_ub_half[:local_valid, :valid].float() * cstripe * mstripe.float()
                    )
                    a_ub_half_out = gated.half()
                    tstore_ub_half_to_gm_a_rows(
                        out[0],
                        a_ub_half_out,
                        token_begin=s + row_off,
                        head_idx=h,
                        n_rows=local_valid,
                        n_cols=valid,
                        chunk_size=chunk_size,
                    )

    return out


def scaled_dot_kkt_fwd_explained(*args, **kwargs):
    return scaled_dot_kkt_fwd(*args, **kwargs)
