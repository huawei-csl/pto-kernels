"""
Educational emulation of ``recompute_w_u_fwd`` (``fla_vendor/wy_fast.py``).

Given the lower-triangular factor :math:`A` (same layout as ``chunk_scaled_dot_kkt_fwd``)
and gates :math:`\\exp(G^{\\mathrm{cum}})`, compute

.. math::

    u_t = \\sum_j A_{tj} \\, \\beta_j v_j, \\qquad
    w_t = \\sum_j A_{tj} \\, \\beta_j \\exp(G^{\\mathrm{cum}}_j)\\, k_j,

i.e. :math:`u = A(\\beta \\odot v)` and :math:`w = A(\\beta \\odot e^G \\odot k)` in block form.
"""

from __future__ import annotations

import numpy as np
import torch

from ._common import k_head_index


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Same arguments as ``fla_vendor.wy_fast.recompute_w_u_fwd``.
    """
    b, t, hg, kdim = k.shape
    vdim = v.shape[-1]
    h = v.shape[-2]
    bt = A.shape[-1]

    w = k.new_empty(b, t, h, kdim)
    u = torch.empty_like(v)

    if cu_seqlens is None:
        seg_ranges = [(0, t - (t % bt))]
    else:
        cu = cu_seqlens.detach().cpu().tolist()
        seg_ranges = []
        for i in range(len(cu) - 1):
            bos, eos = cu[i], cu[i + 1]
            seg_ranges.append((bos, eos - ((eos - bos) % bt)))

    for bos, eos in seg_ranges:
        for ic in range((eos - bos) // bt):
            s = bos + ic * bt
            e = s + bt
            for i_h in range(h):
                hk = k_head_index(i_h, h, hg)
                # SRAM: tile of A [BT, BT] — conceptual buffer after tl.load rows
                a_tile = (
                    A[0, s:e, i_h, :].detach().float().cpu().numpy().astype(np.float32).copy()
                )
                g_np = (
                    g_cumsum[0, s:e, i_h].detach().float().cpu().numpy().astype(np.float32).copy()
                )
                b_np = beta[0, s:e, i_h].detach().float().cpu().numpy().astype(np.float32).copy()
                exp_g = np.exp(g_np)

                k_tile = k[0, s:e, hk, :].detach().float().cpu().numpy().astype(np.float32).copy()
                v_tile = v[0, s:e, i_h, :].detach().float().cpu().numpy().astype(np.float32).copy()

                # u = A @ (beta * v)
                vb = v_tile * b_np[:, None]
                u_tile = (a_tile @ vb).astype(np.float32)
                # w = A @ (beta * exp(g) * k)
                kb = k_tile * b_np[:, None] * exp_g[:, None]
                w_tile = (a_tile @ kb).astype(np.float32)

                u[0, s:e, i_h, :] = torch.from_numpy(np.ascontiguousarray(u_tile)).to(
                    device=u.device, dtype=u.dtype
                )
                w[0, s:e, i_h, :] = torch.from_numpy(np.ascontiguousarray(w_tile)).to(
                    device=w.device, dtype=w.dtype
                )

    return w, u
