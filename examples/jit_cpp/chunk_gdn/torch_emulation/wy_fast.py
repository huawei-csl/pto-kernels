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
                a_tile = A[0, s:e, i_h, :].float()
                g_vec = g_cumsum[0, s:e, i_h].float()
                b_vec = beta[0, s:e, i_h].float()
                exp_g = torch.exp(g_vec)

                k_tile = k[0, s:e, hk, :].float()
                v_tile = v[0, s:e, i_h, :].float()

                # u = A @ (beta * v)
                vb = v_tile * b_vec[:, None]
                u_tile = torch.matmul(a_tile, vb)
                # w = A @ (beta * exp(g) * k)
                kb = k_tile * b_vec[:, None] * exp_g[:, None]
                w_tile = torch.matmul(a_tile, kb)

                u[0, s:e, i_h, :] = u_tile.to(u.dtype)
                w[0, s:e, i_h, :] = w_tile.to(w.dtype)

    return w, u
