"""
Pure PyTorch emulation of ``fla_vendor.chunk_o.chunk_fwd_o``.

Mathematics
-----------
For each output head and each time-chunk of length ``BT``, compute local attention-style terms
using chunk-stored hidden state ``h``:

.. math::

    o^{\\mathrm{local}}_t = \\sum_k q_{t,k} h_{k,:}, \\qquad
    A_{ts} = \\sum_k q_{t,k} k_{s,k}

Gate with cumulative ``G`` (same convention as elsewhere): scale ``o^{local}`` by ``e^{G_t}``,
scale pairwise ``A`` by ``exp(G_t - G_s)`` with ``safe_exp`` for invalid pairs, mask ``A`` to
the causal lower triangle, then

.. math::

    o_t = \\mathrm{scale}\\, o^{\\mathrm{local}}_t
          + \\mathrm{scale} \\sum_{s \\le t} A_{ts} v_s .

``scale`` defaults to ``1/\\sqrt{K}``.

Memory: global vs padded tiles
------------------------------
**Global tensors (DRAM):**

- ``q``, ``k``: ``[B, T, Hg, K]`` — queries/keys (GQA head map via ``k_head_index``).
- ``v``: ``[B, T, H, V]`` — values (often ``v_new`` from upstream).
- ``h``: ``[B, NT, H, K, V]`` — **chunk-indexed** hidden tensor (one slice per chunk, not per time).
- ``g``: ``[B, T, H]`` — cumulative gate; we use ``g_ht``: ``[B, H, T]`` for slicing.
- **Output** ``o``: ``[B, T, H, V]``.

**Padded tiles (emulate Triton block pointers with ``BK=128``, ``BV=128``):**

The kernel walks ``K`` in tiles of ``BK`` and ``V`` in tiles of ``BV``. Here we allocate **one**
padded workspace per chunk (zeros outside valid ``K``/``V``):

- ``q_pad``: ``[BT, K']`` with ``K' = ceil(K/BK)*BK`` — left ``[span, K]`` holds the chunk’s ``q``;
  mirrors ``tl.make_block_ptr`` on ``q``.
- ``k_pad``: ``[K', BT]`` — ``k`` block for the chunk, same padding along ``K``.
- ``h_pad``: ``[K', V']`` — chunk’s slice of **global** ``h[i_b, chunk_idx, i_h, :, :]`` embedded in
  the top-left ``[K, V]`` corner.
- ``v_pad``: ``[BT, V']`` — chunk’s ``v``.

**Intermediate results (before scatter to ``o``):**

- ``o_loc``, ``a_mat``: ``[BT, V']`` and ``[BT, BT]`` in fp32 — analogs of ``b_o`` / ``b_A`` in Triton
  before gating and causal mask; second matmul uses ``A`` cast to ``v`` dtype like ``tl.dot``.
"""

from __future__ import annotations

import torch

from ._common import k_head_index, safe_exp_torch

# Match ``chunk_fwd_kernel_o`` constexprs (Triton tile sizes for K/V splits).
_BK = 128
_BV = 128


def _prepare_chunk_offsets_cpu(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Global chunk base index per sequence (where ``h`` rows live in ``NT`` dimension)."""
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

    ``h`` shape ``[B, NT, H, K, V]``: **NT** is total chunk slots (concatenated sequences when varlen).
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

    # Padded K/V dims: K' = nk*128, V' = nv*128 (ceil to tile); q_pad is [BT, K'], h_pad [K', V'], etc.
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
        """
        One **segment** of packed time: global times ``t ∈ [bos, bos + t_seg)``.

        - ``i_b``: batch row into ``q,k,v,o`` (varlen uses 0 with concatenated ``T``).
        - ``boh``: first **chunk row** in ``h``’s ``NT`` dimension for this segment.
        - ``nt_loc``: number of BT chunks ``ceil(t_seg / BT)``; inner loop ``i_tc`` is 0..nt_loc-1.
        """
        dev = q.device
        for i_h in range(h_heads):
            hq = k_head_index(i_h, h_heads, hg)
            for i_tc in range(nt_loc):
                t0 = i_tc * bt
                t1 = min(t0 + bt, t_seg)
                span = t1 - t0

                # GLOBAL: this chunk’s slice of h from DRAM [K, V]
                h_blk = h[i_b, boh + i_tc, i_h, :, :]

                # Padded tiles (conceptual SRAM / register blocks before dot)
                q_pad = torch.zeros(bt, k_pad_len, device=dev, dtype=wd)
                q_pad[:span, :kdim] = q[i_b, bos + t0 : bos + t1, hq, :]

                k_pad = torch.zeros(k_pad_len, bt, device=dev, dtype=k.dtype)
                k_pad[:kdim, :span] = k[i_b, bos + t0 : bos + t1, hq, :].transpose(0, 1)

                h_pad = torch.zeros(k_pad_len, v_pad_len, device=dev, dtype=h_blk.dtype)
                h_pad[:kdim, :vdim] = h_blk

                v_pad = torch.zeros(bt, v_pad_len, device=dev, dtype=v.dtype)
                v_pad[:span, :vdim] = v[i_b, bos + t0 : bos + t1, i_h, :]

                # --- On-chip fp32 tiles (pre-gate): o_loc [BT, V'], a_mat [BT, BT] ---
                # o_loc[t,:] = sum_k q_pad[t,k] h_pad[k,:]  →  "local" linear-attn path using chunk h.
                o_loc = torch.matmul(q_pad.to(wd), h_pad.to(wd)).float()
                # a_mat[t,s] = sum_k q_pad[t,k] k_pad[k,s]  →  unscaled QK logits within this chunk.
                a_mat = torch.matmul(q_pad.to(wd), k_pad.to(wd)).float()

                if g_ht is not None:
                    # g_chunk: [span] = G_t for t in this chunk; embed in g_pad [BT] (zeros = masked).
                    g_chunk = g_ht[i_b, i_h, bos + t0 : bos + t1].float()
                    g_pad = torch.zeros(bt, device=g.device, dtype=torch.float32)
                    g_pad[:span] = g_chunk
                    # gi [BT,1], gj [1,BT] → (gi-gj) [BT,BT] gives G_t - G_s for every (t,s) pair.
                    gi = g_pad[:, None]
                    gj = g_pad[None, :]
                    # A_ts *= exp(G_t - G_s); safe_exp_torch zeros invalid/padded pairs like Triton mask.
                    a_mat = a_mat * safe_exp_torch(gi - gj)
                    # Local path picks up exp(G_t) per row (docstring: gate on o^local).
                    o_loc = o_loc * torch.exp(g_pad)[:, None]

                # Causal mask: keep only s ≤ t (lower triangle including diagonal); upper → 0.
                idx = torch.arange(bt, device=dev, dtype=torch.long)
                mask = idx[:, None] >= idx[None, :]
                a_mat = torch.where(mask, a_mat, torch.zeros_like(a_mat))

                # o_out [BT, V']: scale * ( o_loc + (A @ v) ); A cast to v dtype before second dot.
                o_out = o_loc * scale + (a_mat.to(v_pad.dtype) @ v_pad).float() * scale
                # GLOBAL o [B,T,H,V]: write only real timesteps bos+t0 .. bos+t1-1.
                o[i_b, bos + t0 : bos + t1, i_h, :] = o_out[:span, :vdim].to(o.dtype)

    if cu_seqlens is None:
        # Each batch row i_b has its own h chunk rows: NT stride nt = ceil(T/BT); base boh = i_b * nt.
        nt = (t_max + bt - 1) // bt
        for i_b in range(b):
            emit_chunk(i_b, 0, t_max, i_b * nt, nt)
    else:
        # Varlen: one physical batch row (i_b=0); sequences concatenated on T. Per sequence i_n:
        # global times [bos,eos), chunk base boh in h's NT axis, nt_loc chunks for that segment.
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
