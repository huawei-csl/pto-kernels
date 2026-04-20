"""
Educational emulation of ``recompute_w_u_fwd`` (``fla_vendor/wy_fast.py``).

Given the lower-triangular factor :math:`A` (same layout as ``chunk_scaled_dot_kkt_fwd``)
and gates :math:`\\exp(G^{\\mathrm{cum}})`, compute

.. math::

    u_t = \\sum_j A_{tj} \\, \\beta_j v_j, \\qquad
    w_t = \\sum_j A_{tj} \\, \\beta_j \\exp(G^{\\mathrm{cum}}_j)\\, k_j,

i.e. :math:`u = A(\\beta \\odot v)` and :math:`w = A(\\beta \\odot e^G \\odot k)` in block form.

Chunk iteration matches Triton ``chunk_indices`` (partial tiles zero-padded to ``BT``).
"""

from __future__ import annotations

import torch

from ._common import iter_packed_bt_chunks, k_head_index, prepare_chunk_indices


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

    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, bt)

    dev = k.device
    for bos, _i_tc, span in iter_packed_bt_chunks(
        cu_seqlens=cu_seqlens, total_t=t, bt=bt, chunk_indices=chunk_indices
    ):
        if span <= 0:
            continue
        s = bos + _i_tc * bt
        for i_h in range(h):
            hk = k_head_index(i_h, h, hg)
            a_pad = torch.zeros(bt, bt, device=dev, dtype=torch.float32)
            a_pad[:span, :] = A[0, s : s + span, i_h, :].float()
            g_pad = torch.zeros(bt, device=dev, dtype=torch.float32)
            g_pad[:span] = g_cumsum[0, s : s + span, i_h].float()
            b_pad = torch.zeros(bt, device=dev, dtype=torch.float32)
            b_pad[:span] = beta[0, s : s + span, i_h].float()
            exp_g = torch.exp(g_pad)

            k_pad = torch.zeros(bt, kdim, device=dev, dtype=torch.float32)
            k_pad[:span] = k[0, s : s + span, hk, :].float()
            v_pad = torch.zeros(bt, vdim, device=dev, dtype=torch.float32)
            v_pad[:span] = v[0, s : s + span, i_h, :].float()

            vb = v_pad * b_pad[:, None]
            u_tile = torch.matmul(a_pad, vb)
            kb = k_pad * b_pad[:, None] * exp_g[:, None]
            w_tile = torch.matmul(a_pad, kb)

            u[0, s : s + span, i_h, :] = u_tile[:span, :].to(u.dtype)
            w[0, s : s + span, i_h, :] = w_tile[:span, :].to(w.dtype)

    return w, u
