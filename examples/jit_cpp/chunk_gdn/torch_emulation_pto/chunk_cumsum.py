"""
Educational emulation of ``chunk_cumsum_kernel.cpp``.

Mathematics
-----------
For each **chunk** of ``C`` tokens (``GDN_C``, e.g. 128), independently per head:

    g_sum[t] = Σ_{i=0}^{t} g[i]    for t = 0 .. valid-1

There is **no** carry across chunk boundaries.

Memory / PTO mapping (``chunk_cumsum_kernel.cpp``)
--------------------------------------------------
**Vec-only** — no Cube core, no L1/L0, and **no Cube↔Vec GM ``workspace``** handoff (only GM↔UB on the vector path). UB tiles ``g_ub`` / ``s_ub`` / ``acc_ub`` are **pre-allocated once** at the
start of ``chunk_cumsum_fwd`` and reused for every sequence and chunk (same fixed SRAM budget as PTO). Data path::

    GM --TLOAD(MTE2)--> UB ``g_ub`` --Vec scan--> UB ``s_ub`` --TSTORE(MTE3)--> GM ``g_sum``

- ``TLOAD(g_load, g_gm)``: ``g_ub[:valid, :H] = g_gm[chunk]``; ``TFILLPAD_INPLACE`` zeros
  rows ``valid:C`` and cols ``H:HTC`` (8-float alignment).
- Row 0: ``TMOV(acc_ub, g_row_0)``; ``TMOV(s_row_0, acc_ub)`` (see C++).
- Rows ``1..valid-1``: ``TADD(acc_ub, acc_ub, g_row_i)``; ``TMOV(s_row_i, acc_ub)``.
- Tail rows ``valid..C-1``: ``s_ub[i] = 0`` (``TEXPANDS`` + row copies in C++).
- ``TSTORE``: write ``s_ub[:valid]`` back to ``g_sum_gm``.

**Index conventions** — ``chunk_start_rel`` steps by ``C`` within ``[bos, eos)``; ``chunk_start`` is the
global packed token index of the chunk’s first row; ``valid`` tokens may be ``< C`` on the last chunk.

Reference: ``verify_dynamic_bsnd.ref_cumsum``.
"""

from __future__ import annotations

import torch

from ._common import seq_ranges
from ._memory import (
    htc_align,
    tadd,
    tfillpad_ub_g_inplace,
    tload_gm_to_ub_g_chunk,
    tmov,
    tstore_ub_to_gm_gsum,
)


def chunk_cumsum_fwd(
    g: torch.Tensor,
    chunk_size: int,
    cu_seqlens: torch.LongTensor | None = None,
) -> torch.Tensor:
    """
    Parameters
    ----------
    g :
        ``[B, T, H]`` float32 (batch 1 typical for varlen).
    chunk_size :
        ``GDN_C`` (compile-time chunk length, e.g. 128).

    Returns
    -------
    g_sum : same shape/dtype as ``g`` (float32), chunk-local cumulative sums.
    """
    _, t, h = g.shape
    device = g.device
    htc = htc_align(h)
    g32 = g.float()
    out = torch.zeros_like(g32)

    # UB fp32 ``g_ub`` [C×HTC] — ``4·C·HTC`` B → **C·HTC/256** KiB (e.g. **8 KiB** @ C=128, H=16 → HTC=16); ``chunk_cumsum_kernel`` row pool
    g_ub = torch.zeros(chunk_size, htc, device=device, dtype=torch.float32)
    # UB fp32 ``s_ub`` [C×HTC] — same as ``g_ub`` (**C·HTC/256** KiB)
    s_ub = torch.zeros(chunk_size, htc, device=device, dtype=torch.float32)
    # UB fp32 ``acc_ub`` [1×HTC] — ``4·HTC`` B → **HTC/256** KiB (≈**0.0625 KiB** @ HTC=16)
    acc_ub = torch.zeros(1, htc, device=device, dtype=torch.float32)

    for bos, eos in seq_ranges(t, cu_seqlens):
        n_tokens = eos - bos
        for chunk_start_rel in range(0, n_tokens, chunk_size):
            # Global token index where this chunk begins in the packed batch; [s, e) ⊆ [bos, eos).
            chunk_start = bos + chunk_start_rel
            s, e = chunk_start, min(chunk_start + chunk_size, eos)
            valid = e - s

            # TLOAD: GM → UB
            tload_gm_to_ub_g_chunk(
                g_ub,
                g32[0, s:e, :],
                valid=valid,
                num_heads=h,
                htc=htc,
            )
            tfillpad_ub_g_inplace(
                g_ub, valid=valid, chunk_size=chunk_size, num_heads=h, htc=htc
            )

            # Vec: prefix scan — ``TMOV`` / ``TADD`` (``chunk_cumsum_kernel.cpp``)
            tmov(acc_ub, g_ub[0:1, :])
            tmov(s_ub[0:1, :], acc_ub)
            for i in range(1, valid):
                tadd(acc_ub, acc_ub, g_ub[i : i + 1, :])
                tmov(s_ub[i : i + 1, :], acc_ub)

            # ``TEXPANDS(acc_ub, 0)`` then per-row ``TMOV(s_row_i, acc_ub)`` for tail rows
            if valid < chunk_size:
                acc_ub.zero_()
                for i in range(valid, chunk_size):
                    tmov(s_ub[i : i + 1, :], acc_ub)

            # TSTORE: UB → GM
            tstore_ub_to_gm_gsum(out[0], s_ub, chunk_start=chunk_start, valid=valid, num_heads=h)

    return out.to(dtype=g.dtype)


def chunk_cumsum_fwd_explained(*args, **kwargs):
    """Alias for readers grepping ``*_explained`` like the Triton tree."""
    return chunk_cumsum_fwd(*args, **kwargs)
