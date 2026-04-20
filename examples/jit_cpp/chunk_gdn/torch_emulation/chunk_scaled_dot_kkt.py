"""
Educational emulation of ``chunk_scaled_dot_kkt_fwd`` (``fla_vendor/chunk_scaled_dot_kkt.py``).

Within each time chunk of length ``BT``, form the local Gram matrix and apply the gate:

.. math::

    A_{ij} = \\beta_i\\, \\exp(G_i - G_j)\\, \\langle k_i, k_j \\rangle,
    \\quad i > j

(strictly lower triangular; causal mask). This is the local KKT / local attention block
used to build the WY / delta-rule factors.

Iteration follows Triton ``chunk_indices``: every chunk tile (including partial tails) is a
separate program; invalid rows are zero-padded to ``BT`` like ``tl.load(..., boundary_check)``.
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
    Same arguments as ``fla_vendor.chunk_scaled_dot_kkt.chunk_scaled_dot_kkt_fwd``.
    Output layout ``[B, T, H, BT]``: row ``r`` within a chunk stores :math:`A_{r,0:BT}`.
    """
    b, t, hg, kdim = k.shape
    h = beta.shape[-1]
    bt = chunk_size
    out = torch.zeros(b, t, h, bt, device=k.device, dtype=output_dtype)

    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, bt)

    dev = k.device
    idx = torch.arange(bt, device=dev, dtype=torch.long)
    mask = idx[:, None] > idx[None, :]

    for bos, _i_tc, span in iter_packed_bt_chunks(
        cu_seqlens=cu_seqlens, total_t=t, bt=bt, chunk_indices=chunk_indices
    ):
        if span <= 0:
            continue
        s = bos + _i_tc * bt
        for i_h in range(h):
            hk = k_head_index(i_h, h, hg)
            k_pad = torch.zeros(bt, kdim, device=dev, dtype=torch.float32)
            k_pad[:span] = k[0, s : s + span, hk, :].float()
            beta_pad = torch.zeros(bt, device=dev, dtype=torch.float32)
            beta_pad[:span] = beta[0, s : s + span, i_h].float()
            kk = torch.matmul(k_pad, k_pad.transpose(0, 1))
            if g_cumsum is not None:
                g_pad = torch.zeros(bt, device=dev, dtype=torch.float32)
                g_pad[:span] = g_cumsum[0, s : s + span, i_h].float()
                gi = g_pad[:, None]
                gj = g_pad[None, :]
                kk = kk * safe_exp_torch(gi - gj)
            blk = kk * beta_pad[:, None]
            blk = torch.where(mask, blk, torch.zeros_like(blk))
            out[0, s : s + span, i_h, :] = blk[:span, :].to(output_dtype)

    return out
