"""
Educational emulation of ``wy_fast_kernel.cpp``.

Mathematics
-----------
``U = A2 @ V``, ``W = A1 @ K`` with the same **column / row** scaling convention as
``verify_dynamic_bsnd.ref_wy`` (see existing docstring in this file's history).

Memory / PTO mapping (``wy_fast_kernel.cpp``)
---------------------------------------------
**Vec** builds ``A1`` / ``A2`` in UB, ``TSTORE`` top-left ``[valid×valid]`` to GM **``workspace_a``** fp16 ``[C×C]``.

**Cube**:

- ``TLOAD(a_l1, workspace_a)`` — ``[C×C]`` half into L1 (explicit GM staging, not direct GM ``A``).
- ``TLOAD(v_l1, v_gm)`` — ``[C×D]`` (``DynMatL1``) into L1 at offset 32768.
- ``TMATMUL`` → ``u_l0`` ``[C×D]`` fp32, ``TSTORE`` to ``U`` GM.

Second branch: ``a1_l1`` + ``k_l1`` → ``w_l0``.

Emulation uses shared **``workspace_a``** fp16 **``[C×C]``** as the Vec→Cube channel: ``TSTORE`` from Vec,
``TLOAD`` into ``a_l1``. Size: ``2·C²`` B → **C²/512** KiB (e.g. **32 KiB** @ C=128).

``a_l1``, ``v_l1``, ``k_l1``, L0 stripes, and a shared L0C buffer are **pre-allocated once** at the
start of ``wy_fast_fwd`` and reused for every chunk (PTO-style fixed SRAM).

**Index conventions** — ``(bos, eos)`` from ``seq_ranges``; ``chunk_start_rel`` steps by ``C`` along
``[bos, eos)``; ``s``, ``e``, ``valid`` bound the current tile (``valid < C`` on the last chunk only).

Reference: ``verify_dynamic_bsnd.ref_wy``.
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
    tmov_l1_half_rows,
    tstore,
)


def wy_fast_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g_cumsum: torch.Tensor,
    chunk_size: int,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns ``(w, u)`` with shapes ``[B, T, H, D]`` and ``[B, T, H, V]`` (fp32 compute).
    """
    b, t, hd, d = k.shape
    vdim = v.shape[-1]
    assert b == 1
    device = k.device
    w = torch.zeros(b, t, hd, d, device=device, dtype=torch.float32)
    u = torch.zeros(b, t, hd, vdim, device=device, dtype=torch.float32)
    kf, vf, bf, Af, gf = k.float(), v.float(), beta.float(), A.float(), g_cumsum.float()
    k_tile = 128
    mx = max(chunk_size, vdim, d)

    # L1 fp16 ``a_l1`` [C×C] — **C²/256** KiB (e.g. **32 KiB** @ C=128)
    a_l1 = torch.empty((chunk_size, chunk_size), device=device, dtype=torch.float16)
    # L1 fp16 ``v_l1`` [C×V] — **C·V/512** KiB (e.g. **32 KiB** @ C=V=128)
    v_l1 = alloc_l1_cd(chunk_size, vdim, device=device, dtype=torch.float16)
    # L1 fp16 ``k_l1`` [C×D] — **C·D/512** KiB (e.g. **32 KiB** @ C=D=128)
    k_l1 = alloc_l1_cd(chunk_size, d, device=device, dtype=torch.float16)
    # L0C fp32 (U / W branches time-shared) [C×max(V,D)] — **C·max(V,D)/256** KiB
    l0c_uv = torch.zeros(
        chunk_size, max(vdim, d), device=device, dtype=torch.float32
    )
    # L0A/L0B fp16 stripes — **mx·K_tile/512** KiB each
    l0a_buf, l0b_buf = alloc_l0_stripes_gemm_v0(
        mx, mx, k_tile, device=device, dtype=torch.float16
    )
    # GM ``workspace_a`` fp16 [C×C] — **C²/512** KiB — Vec ``TSTORE`` ``A`` tile; Cube ``TLOAD`` → ``a_l1``
    workspace_a = torch.empty(
        chunk_size, chunk_size, device=device, dtype=torch.float16
    )

    for bos, eos in seq_ranges(t, cu_seqlens):
        n_tokens = eos - bos
        for h in range(hd):
            # Walk chunks: chunk_start_rel is the offset from bos (0, C, 2C, …) within this sequence.
            for chunk_start_rel in range(0, n_tokens, chunk_size):
                s = bos + chunk_start_rel
                e = min(s + chunk_size, eos)
                valid = e - s
                Ab = Af[0, s:e, h, :valid]
                gc = gf[0, s:e, h]
                vb = vf[0, s:e, h, :] * bf[0, s:e, h, None]
                kb = kf[0, s:e, h, :] * bf[0, s:e, h, None] * torch.exp(gc)[:, None]

                # Vec→Cube: ``TSTORE`` top-left ``A`` → ``workspace_a``; Cube ``TLOAD`` → ``a_l1``
                tstore(
                    workspace_a,
                    Ab.half(),
                    direction="ub_to_gm",
                    nrows=valid,
                    ncols=valid,
                    clear_dst=True,
                )
                tload(
                    a_l1,
                    workspace_a,
                    direction="gm_to_l1",
                    nrows=chunk_size,
                    ncols=chunk_size,
                )

                tmov_l1_half_rows(v_l1, vb.half(), valid_rows=valid)
                tfillpad_k_l1_tail_rows(v_l1, valid_rows=valid, chunk_size=chunk_size)

                tmov_l1_half_rows(k_l1, kb.half(), valid_rows=valid)
                tfillpad_k_l1_tail_rows(k_l1, valid_rows=valid, chunk_size=chunk_size)

                u_l0 = gemm_v0_accum_fp16(
                    a_l1,
                    v_l1,
                    k_tile=k_tile,
                    l0c_out=l0c_uv[:, :vdim],
                    l0a_buf=l0a_buf,
                    l0b_buf=l0b_buf,
                )
                u[0, s:e, h, :] = u_l0[:valid, :]

                w_l0 = gemm_v0_accum_fp16(
                    a_l1,
                    k_l1,
                    k_tile=k_tile,
                    l0c_out=l0c_uv[:, :d],
                    l0a_buf=l0a_buf,
                    l0b_buf=l0b_buf,
                )
                w[0, s:e, h, :] = w_l0[:valid, :]

    return w.to(k.dtype), u.to(v.dtype)


def wy_fast_fwd_explained(*args, **kwargs):
    return wy_fast_fwd(*args, **kwargs)
