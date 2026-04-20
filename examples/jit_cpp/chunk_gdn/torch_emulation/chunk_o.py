"""
Pure PyTorch emulation of ``fla_vendor.chunk_o.chunk_fwd_o`` (numpy tiles = conceptual SRAM).

Within each chunk, compute the local attention contribution to the output:

.. math::

    o^{\\mathrm{local}}_t = \\sum_k q_{t,k} \\, h_{k,:}, \\qquad
    A_{ts} = \\sum_k q_{t,k} \\, k_{s,k}

Apply the gate :math:`\\exp(G_t)` to :math:`o^{\\mathrm{local}}` and
:math:`\\exp(G_t - G_s)` to :math:`A` (with ``safe_exp`` for invalid pairs),
mask :math:`A` to the causal (lower) part, then

.. math::

    o_t = \\mathrm{scale} \\cdot o^{\\mathrm{local}}_t
          + \\mathrm{scale} \\cdot \\sum_{s \\le t} A_{ts} \\, v_s.

Padding and block sizes ``BK=128``, ``BV=128`` match the Triton kernel so bf16
``tl.dot`` behavior aligns with ``torch.matmul`` on padded tiles (no CPU numpy path).
"""

from __future__ import annotations

import torch

from ._common import k_head_index, safe_exp_torch

# Match ``chunk_fwd_kernel_o`` constexprs
_BK = 128
_BV = 128


def _prepare_chunk_offsets_cpu(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    nchunks = (lens + chunk_size - 1) // chunk_size
    z = cu_seqlens.new_zeros(1)
    return torch.cat([z, nchunks], dim=0).cumsum(-1)


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    Same arguments as ``fla_vendor.chunk_o.chunk_fwd_o``.
    ``h`` has shape ``[B, NT, H, K, V]`` (chunk-stored hidden states).
    """
    b, t_max, hg, kdim = q.shape
    vdim = v.shape[-1]
    h_heads = v.shape[-2]
    bt = chunk_size
    if scale is None:
        scale = kdim**-0.5

    wd = q.dtype
    o = torch.empty_like(v)
    g_ht = g.transpose(1, 2).contiguous() if g is not None else None

    # Pad K/V to the same multiples as Triton block pointers (zeros outside valid region).
    nk = (kdim + _BK - 1) // _BK
    k_pad_len = nk * _BK
    nv = (vdim + _BV - 1) // _BV
    v_pad_len = nv * _BV

    def emit_chunk(
        i_b: int,
        bos: int,
        t_seg: int,
        boh: int,
        nt_loc: int,
    ) -> None:
        dev = q.device
        for i_h in range(h_heads):
            hq = k_head_index(i_h, h_heads, hg)
            for i_tc in range(nt_loc):
                t0 = i_tc * bt
                t1 = min(t0 + bt, t_seg)
                span = t1 - t0

                h_blk = h[i_b, boh + i_tc, i_h, :, :]

                q_pad = torch.zeros(bt, k_pad_len, device=dev, dtype=wd)
                q_pad[:span, :kdim] = q[i_b, bos + t0 : bos + t1, hq, :]

                k_pad = torch.zeros(k_pad_len, bt, device=dev, dtype=k.dtype)
                k_pad[:kdim, :span] = k[i_b, bos + t0 : bos + t1, hq, :].transpose(0, 1)

                h_pad = torch.zeros(k_pad_len, v_pad_len, device=dev, dtype=h_blk.dtype)
                h_pad[:kdim, :vdim] = h_blk

                v_pad = torch.zeros(bt, v_pad_len, device=dev, dtype=v.dtype)
                v_pad[:span, :vdim] = v[i_b, bos + t0 : bos + t1, i_h, :]

                # [BT, K'] @ [K', V'] -> [BT, V']; same accumulation pattern as tl.dot tiles
                o_loc = torch.matmul(q_pad.to(wd), h_pad.to(wd)).float()
                a_mat = torch.matmul(q_pad.to(wd), k_pad.to(wd)).float()

                if g_ht is not None:
                    g_chunk = g_ht[i_b, i_h, bos + t0 : bos + t1].float()
                    g_pad = torch.zeros(bt, device=g.device, dtype=torch.float32)
                    g_pad[:span] = g_chunk
                    gi = g_pad[:, None]
                    gj = g_pad[None, :]
                    a_mat = a_mat * safe_exp_torch(gi - gj)
                    o_loc = o_loc * torch.exp(g_pad)[:, None]

                idx = torch.arange(bt, device=dev, dtype=torch.long)
                mask = idx[:, None] >= idx[None, :]
                a_mat = torch.where(mask, a_mat, torch.zeros_like(a_mat))

                # Match Triton: second dot uses A cast to v dtype
                o_out = o_loc * scale + (a_mat.to(v_pad.dtype) @ v_pad).float() * scale
                o[i_b, bos + t0 : bos + t1, i_h, :] = o_out[:span, :vdim].to(o.dtype)

    if cu_seqlens is None:
        nt = (t_max + bt - 1) // bt
        for i_b in range(b):
            emit_chunk(i_b, 0, t_max, i_b * nt, nt)
    else:
        cu = cu_seqlens.detach().cpu().tolist()
        offs = _prepare_chunk_offsets_cpu(cu_seqlens, bt)
        for i_n in range(len(cu) - 1):
            bos, eos = cu[i_n], cu[i_n + 1]
            t_seg = eos - bos
            nt_loc = (t_seg + bt - 1) // bt
            boh = int(offs[i_n].item())
            emit_chunk(0, bos, t_seg, boh, nt_loc)

    return o


chunk_fwd_o_explained = chunk_fwd_o
