"""
Educational emulation of ``chunk_scaled_dot_kkt_fwd`` (``fla_vendor/chunk_scaled_dot_kkt.py``).

Within each time chunk of length ``BT``, form the local Gram matrix and apply the gate:

.. math::

    A_{ij} = \\beta_i\\, \\exp(G_i - G_j)\\, \\langle k_i, k_j \\rangle,
    \\quad i > j

(strictly lower triangular; causal mask). This is the local KKT / local attention block
used to build the WY / delta-rule factors.
"""

from __future__ import annotations

import torch

from ._common import k_head_index, safe_exp_torch


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

    if cu_seqlens is None:
        seg_ranges = [(0, t - (t % bt))]
    else:
        cu = cu_seqlens.detach().cpu().tolist()
        seg_ranges = []
        for i in range(len(cu) - 1):
            bos, eos = cu[i], cu[i + 1]
            seg_ranges.append((bos, eos - ((eos - bos) % bt)))

    for bos, eos in seg_ranges:
        for i in range((eos - bos) // bt):
            s = bos + i * bt
            e = s + bt
            # GLOBAL: full chunk tensors (DRAM)
            k_c = k[:, s:e, :, :]
            g_c = g_cumsum[:, s:e, :] if g_cumsum is not None else None
            b_c = beta[:, s:e, :]

            for i_h in range(h):
                hk = k_head_index(i_h, h, hg)
                # Conceptual SRAM tiles (float32 on device; mirrors tl.load blocks)
                k_tile = k_c[0, :, hk, :].float()
                beta_tile = b_c[0, :, i_h].float()
                kk = torch.matmul(k_tile, k_tile.transpose(0, 1))
                if g_c is not None:
                    g_tile = g_c[0, :, i_h].float()
                    gi = g_tile[:, None]
                    gj = g_tile[None, :]
                    kk = kk * safe_exp_torch(gi - gj)
                blk = kk * beta_tile[:, None]
                idx = torch.arange(bt, device=k.device, dtype=torch.long)
                mask = idx[:, None] > idx[None, :]
                blk = torch.where(mask, blk, torch.zeros_like(blk))
                out[0, s:e, i_h, :] = blk.to(output_dtype)

    return out
