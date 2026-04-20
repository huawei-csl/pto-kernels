"""
Educational emulation of ``chunk_scaled_dot_kkt_fwd`` (``fla_vendor/chunk_scaled_dot_kkt.py``).

Mathematics
-----------
For one time-chunk of length ``BT`` (64 by default), build the **local** Gram matrix over
timesteps in that chunk, then apply per-timestep ``β`` and cumulative gate ``G`` (optional):

.. math::

    M_{ij} = \\langle k_i, k_j \\rangle, \\quad
    A_{ij} = \\beta_i\\, \\exp(G_i - G_j)\\, M_{ij}, \\quad i > j

(strictly **lower** triangular in causal order; upper triangle and diagonal zeroed). This block
feeds the WY / Cholesky-style pipeline (``solve_tril``, ``wy_fast``, ``chunk_delta_h``).

Memory: global vs tile
----------------------
**Global tensors** (layout matches Triton):

- ``k``: ``[B, T, Hg, K]`` — keys along packed time.
- ``beta``: ``[B, T, H]`` — scalar per time and output head.
- ``g_cumsum``: ``[B, T, H]`` — cumulative gate (already prefix-summed inside each sequence).
- **Output** ``out``: ``[B, T, H, BT]``. For global time row ``t``, ``out[b,t,h,:]`` holds one
  **row** of the ``BT × BT`` block that the chunk containing ``t`` belongs to: the row’s index
  within that block is ``(t - chunk_start)``.

**Tile / SRAM (emulated):** For each chunk program we form float32 pads:

- ``k_pad``: shape ``[BT, K]`` — rows are ``k`` for ``BT`` timesteps; rows past ``span-1`` are
  **zero** (same as ``tl.load`` with ``boundary_check`` on a partial tail chunk).
- ``beta_pad``, ``g_pad``: shape ``[BT]``.
- ``blk``: shape ``[BT, BT]`` — full Gram after gating and ``β``; multiply by strict-lower mask.
  Only rows ``0:span`` are **stored** back to ``out`` (``tl.store`` with boundary).

Iteration uses ``iter_packed_bt_chunks`` so **partial** last chunks match Triton ``chunk_indices``.
"""

from __future__ import annotations

import torch

from ._common import iter_packed_bt_chunks, k_head_index, prepare_chunk_indices, safe_exp_torch


def chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Same API as ``fla_vendor.chunk_scaled_dot_kkt.chunk_scaled_dot_kkt_fwd``.

    Returns ``out`` with shape ``[B, T, H, BT]`` (``B`` must be 1 for varlen in downstream code).
    """
    b, t, hg, kdim = k.shape
    h = beta.shape[-1]
    bt = chunk_size
    # GLOBAL out [B, T, H, BT]: out[b,t,h,r] is row (t - chunk_start) of the local BT×BT block, column r.
    out = torch.zeros(b, t, h, bt, device=k.device, dtype=output_dtype)

    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, bt)

    dev = k.device
    # Chunk-relative causal mask: idx [BT]; mask [BT, BT] True where row_i > col_j (strict lower).
    idx = torch.arange(bt, device=dev, dtype=torch.long)
    mask = idx[:, None] > idx[None, :]

    for bos, _i_tc, span in iter_packed_bt_chunks(
        cu_seqlens=cu_seqlens, total_t=t, bt=bt, chunk_indices=chunk_indices
    ):
        if span <= 0:
            continue
        # Global index of timestep 0 in this chunk: rows s .. s+span-1 in GLOBAL k/beta/out.
        s = bos + _i_tc * bt
        for i_h in range(h):
            hk = k_head_index(i_h, h, hg)
            # k_pad [BT, K]: GLOBAL keys for this chunk; rows span..BT-1 stay zero (masked load).
            k_pad = torch.zeros(bt, kdim, device=dev, dtype=torch.float32)
            k_pad[:span] = k[0, s : s + span, hk, :].float()
            # beta_pad [BT]: per-timestep scalar β; same zero tail as k_pad.
            beta_pad = torch.zeros(bt, device=dev, dtype=torch.float32)
            beta_pad[:span] = beta[0, s : s + span, i_h].float()
            # kk [BT, BT] = k_pad @ k_pad.T — local Gram M_ij = <k_i, k_j> (fp32, full square).
            kk = torch.matmul(k_pad, k_pad.transpose(0, 1))
            if g_cumsum is not None:
                # g_pad [BT]; gi [BT,1], gj [1,BT] → exp(G_i - G_j) broadcast [BT,BT] onto kk.
                g_pad = torch.zeros(bt, device=dev, dtype=torch.float32)
                g_pad[:span] = g_cumsum[0, s : s + span, i_h].float()
                gi = g_pad[:, None]
                gj = g_pad[None, :]
                kk = kk * safe_exp_torch(gi - gj)
            # blk [BT, BT]: row-wise β — beta_pad[:, None] is [BT,1] → multiply each row i by β_i.
            blk = kk * beta_pad[:, None]
            # Zero upper triangle + diagonal; keep only i > j (strict lower), matching math A_ij.
            blk = torch.where(mask, blk, torch.zeros_like(blk))
            # GLOBAL out [B,T,H,BT]: each time row gets one **line** of blk; only span rows written here.
            out[0, s : s + span, i_h, :] = blk[:span, :].to(output_dtype)

    return out
