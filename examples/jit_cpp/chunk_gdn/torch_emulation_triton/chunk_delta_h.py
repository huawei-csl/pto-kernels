"""
Pure PyTorch emulation of ``fla_vendor.chunk_delta_h.chunk_gated_delta_rule_fwd_h``.

Mathematics (gated delta rule on chunk state)
----------------------------------------------
For each sequence and head, maintain a **hidden state** ``h`` over keys × values. Within a time
chunk of length ``BT``, the recurrence loads ``w``, ``k``, gated ``u``, and cumulative gate ``G``,
updates the **new value** ``v_new = u - W h`` (then applies gates), and integrates

.. math::

    h \\leftarrow g_{\\mathrm{last}} \\, h + K^{\\top} (v_{\\mathrm{new}}' )

(with ``v_new'`` the gated new-value tensor in key dtype for the ``K @ v`` dot). Two **value
bands** split ``V`` into ``[0, 64)`` and ``[64, 128)`` when ``V > 64``, implemented as two fixed
``128 × 64`` register tiles (Triton ``tl.zeros([128, 64])``).

Memory: global vs on-chip tiles
-------------------------------
**Global tensors (DRAM, typical shapes for batch 1):**

- ``k``: ``[1, T, Hg, K]`` — key head layout (GQA via ``k_head_index``).
- ``w``, ``u``: ``[1, T, H, K]`` / ``[1, T, H, V]`` — WY factors and value input.
- ``g``: ``[1, T, H]`` cumulative gate (same convention as rest of chain); internally we use
  ``g_ht``: ``[1, H, T]`` for time slicing.
- ``h_out``: ``[B, NT, H, K, V]`` — **chunk-wise** snapshot of ``h``: index ``(b, chunk, h)``
  stores ``h`` **before** processing that chunk’s timesteps (matches kernel store order).
- ``v_new``: ``[1, T, H, V]`` — per-time updated value (optional).
- ``initial_state``: ``[N, H, K, V]`` — per-sequence initial ``h`` when varlen.

**On-chip tiles (SRAM stand-ins — float32 unless noted):**

- ``b_h1_bv1``, ``b_h1_bv2``: each ``[128, 64]`` — **state tiles** for the two V-bands; these are
  the accumulators that ``tl.dot`` updates each micro-step (analogous to ``b_h1_bv*`` in Triton).
- ``w_pad``: ``[BT, 128]`` — one chunk of ``w`` with keys padded to the fixed tile width ``128``.
- ``k_pad``: ``[128, BT]`` — ``k`` block transposed to match ``K @ v_new`` layout.
- ``b_v1``, ``b_v2``: ``[BT, 64]`` — loaded ``u`` slices for each band (float32 scratch).
- ``b_v_new1``, ``b_v_new2``: same shape — **after** ``u - W@h`` and optional gating; cast to key
  dtype ``kd`` before ``matmul`` with ``k_pad`` to match ``tl.dot`` accumulation.

The **pack** step ``_pack_h_from_tiles`` maps the two tiles back to a dense ``[K, V]`` matrix for
**global** ``h_out`` (bf16/fp16 store in reference).
"""

from __future__ import annotations

import torch

from ._common import k_head_index, prepare_chunk_indices, safe_exp_torch


def _prepare_chunk_offsets_cpu(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """
    Global **metadata** only: **exclusive prefix sum** of per-sequence **chunk counts**.

    If sequence ``n`` has length ``L_n``, it occupies ``ceil(L_n / BT)`` rows in ``h_out``’s ``NT``
    dimension. ``chunk_offsets[n]`` is the **first chunk index** belonging to sequence ``n`` when
    all sequences’ chunks are laid out consecutively (same ordering as ``prepare_chunk_indices``).
    """
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    nchunks = (lens + chunk_size - 1) // chunk_size
    z = cu_seqlens.new_zeros(1)
    return torch.cat([z, nchunks], dim=0).cumsum(-1)


def _pack_h_from_tiles(
    b_h1_bv1: torch.Tensor,
    b_h1_bv2: torch.Tensor,
    kdim: int,
    vdim: int,
    tile_v: int,
) -> torch.Tensor:
    """
    **Global** dense ``h`` slice ``[K, V]`` (fp32) from two **tiles** ``128×64``.

    Indices ``v ∈ [0, tile_v)`` map to ``b_h1_bv1``; ``v ∈ [tile_v, 2*tile_v)`` to ``b_h1_bv2``.
    """
    # h [K, V] fp32: scatter from tiles [128,64] + [128,64] into dense global layout for storage.
    h = torch.zeros(kdim, vdim, device=b_h1_bv1.device, dtype=torch.float32)
    c1 = min(tile_v, vdim)
    h[:, :c1] = b_h1_bv1[:kdim, :c1]
    if vdim > tile_v:
        c2 = min(tile_v, vdim - tile_v)
        h[:, tile_v : tile_v + c2] = b_h1_bv2[:kdim, :c2]
    return h


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """
    Same arguments as ``fla_vendor.chunk_delta_h.chunk_gated_delta_rule_fwd_h``.
    """
    b, t_max, hg, kdim = k.shape
    vdim = u.shape[-1]
    h_heads = u.shape[-2]
    bt = chunk_size
    # Fixed Triton tile geometry (must match kernel constexprs)
    tile_k, tile_v = 128, 64

    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    if cu_seqlens is None:
        # Fixed layout: one “segment” per batch row, but this emulation reads **batch index 0** and
        # lays batch items **back-to-back on the time axis**: global time t runs 0..B*T-1 in slot 0.
        n, nt = b, (t_max + bt - 1) // bt
        chunk_offsets_t = None
    else:
        if chunk_offsets is None:
            chunk_offsets_t = _prepare_chunk_offsets_cpu(cu_seqlens, bt)
        else:
            chunk_offsets_t = chunk_offsets
        n = len(cu_seqlens) - 1  # number of logical sequences
        nt = len(chunk_indices)  # total chunk rows across all sequences (length of packed index list)

    # GLOBAL outputs (DRAM): h_out [B, NT, H, K, V] chunk snapshots; v_new [B,T,H,V] per-timestep v_new;
    # final_state [N, H, K, V] one dense h per sequence when requested (varlen N sequences).
    h_out = k.new_empty(b, nt, h_heads, kdim, vdim)
    v_new = torch.empty_like(u) if save_new_value else None
    final_state = k.new_empty(n, h_heads, kdim, vdim, dtype=torch.float32) if output_final_state else None

    # g_ht [B, H, T]: contiguous time last — g_ht[b,h,t] = G_t for indexing with bos+t0:t1 slices.
    g_ht = g.transpose(1, 2).contiguous() if g is not None else None

    cu_list = cu_seqlens.detach().cpu().tolist() if cu_seqlens is not None else None

    for i_n in range(n if cu_seqlens is not None else b):
        # --- Map outer index i_n to (global time interval) × (chunk row window in h_out) ----------
        # Math: the recurrence is over **absolute time indices** t indexing k(t), w(t), u(t), g(t).
        # For each segment, we process timesteps t ∈ [bos, eos) in blocks of BT; chunk index in h_out
        # is boh + i_tc with i_tc = 0 .. nt_loc-1. Snapshot h_out[boh+i_tc] = h **before** that block.
        if cu_seqlens is not None:
            # Varlen: cu_seqlens is exclusive prefix lengths; sequence i_n uses global times
            # t ∈ [bos, eos) with length t_seg = eos - bos (same t as in the formulas in the module doc).
            bos, eos = cu_list[i_n], cu_list[i_n + 1]
            t_seg = eos - bos
            # First chunk row for this sequence in the **packed** NT dimension (all sequences concat).
            boh = int(chunk_offsets_t[i_n].item())
            # Chunks needed to cover [bos, eos): i_tc runs 0..nt_loc-1; last chunk may be partial (span < BT).
            nt_loc = (t_seg + bt - 1) // bt
        else:
            # No cu_seqlens: batch item i_n is stored at global times [i_n*t_max, (i_n+1)*t_max) in **batch 0**.
            bos, eos = i_n * t_max, (i_n + 1) * t_max
            t_seg = t_max
            # Each batch row contributes nt = ceil(t_max/BT) consecutive rows in h_out[:, :, ...].
            boh = i_n * ((t_max + bt - 1) // bt)
            nt_loc = (t_max + bt - 1) // bt

        for i_h in range(h_heads):
            hk = k_head_index(i_h, h_heads, hg)
            wd, kd = w.dtype, k.dtype

            # --- SRAM: two persistent state tiles (fp32 accum, match tl.zeros([128,64])) ---
            b_h1_bv1 = torch.zeros(tile_k, tile_v, device=k.device, dtype=torch.float32)
            b_h1_bv2 = torch.zeros(tile_k, tile_v, device=k.device, dtype=torch.float32)

            if initial_state is not None:
                # GLOBAL h0 → tile init
                h0 = initial_state[i_n, i_h, :, :].float()
                b_h1_bv1[:kdim, : min(tile_v, vdim)] += h0[:, : min(tile_v, vdim)]
                if vdim > tile_v:
                    b_h1_bv2[:kdim, : min(tile_v, vdim - tile_v)] += h0[:, tile_v : vdim]

            for i_tc in range(nt_loc):
                # Store **current** tile state to GLOBAL h_out (kernel stores before micro-updates).
                h_out[0, boh + i_tc, i_h, :, :] = _pack_h_from_tiles(
                    b_h1_bv1, b_h1_bv2, kdim, vdim, tile_v
                ).to(h_out.dtype)

                # Within-segment time for this chunk: local τ ∈ [0, BT) maps to global t = bos + t0 + τ.
                # i_tc indexes which BT-wide **sliding window** along the segment (math: chunk c = i_tc).
                t0 = i_tc * bt
                t1 = min(t0 + bt, t_seg)
                span = t1 - t0  # valid rows in this chunk (last chunk may have span < BT)
                dev = k.device

                # Tiles: GLOBAL chunk slices → w_pad [BT,128], k_pad [128,BT] (Triton fixed tile width).
                w_pad = torch.zeros(bt, tile_k, device=dev, dtype=wd)
                w_pad[:span, :kdim] = w[0, bos + t0 : bos + t1, i_h, :]

                k_pad = torch.zeros(tile_k, bt, device=dev, dtype=kd)
                k_pad[:kdim, :span] = k[0, bos + t0 : bos + t1, hk, :].T

                if g_ht is not None:
                    # Gate uses cumulative G at chunk end vs each step: matches h ← g_last*h + K^T(...)
                    # with per-step scaling of v_new by exp(G_last - G_t) (see safe_exp on the slice).
                    g_last_scalar = g_ht[0, i_h, bos + t1 - 1].float()
                    g_chunk = g_ht[0, i_h, bos + t0 : bos + t1].float()
                    b_g = safe_exp_torch(g_last_scalar - g_chunk)
                    b_g_last = torch.exp(g_last_scalar)
                    b_g_pad = torch.zeros(bt, device=dev, dtype=torch.float32)
                    b_g_pad[:span] = b_g
                else:
                    b_g_pad = torch.ones(bt, device=dev, dtype=torch.float32)
                    b_g_last = torch.tensor(1.0, device=dev, dtype=torch.float32)

                # --- Band 1: first V tile, global columns [0, tile_v) ---
                b_v1 = torch.zeros(bt, tile_v, device=dev, dtype=torch.float32)
                c1 = min(tile_v, vdim)
                b_v1[:span, :c1] = u[0, bos + t0 : bos + t1, i_h, :c1].float()
                # v_new1 = u1 - W @ h1: [BT,128]@[128,64] → [BT,64] (fp32 accum).
                b_v_new1 = b_v1 - torch.matmul(w_pad, b_h1_bv1.to(wd)).to(torch.float32)
                if save_new_value and v_new is not None:
                    v_new[0, bos + t0 : bos + t1, i_h, :c1] = b_v_new1[:span, :c1].to(v_new.dtype)

                if g_ht is not None:
                    b_v_new1 = b_v_new1 * b_g_pad[:, None]
                    b_h1_bv1 = b_h1_bv1 * b_g_last
                b_v_new1_bf = b_v_new1.to(kd)
                # k_pad [128, BT] @ b_v_new1_bf [BT, 64] → contrib1 [128, 64]; h += contrib (same as band 2).
                contrib1 = torch.matmul(k_pad, b_v_new1_bf).to(torch.float32)
                b_h1_bv1 = b_h1_bv1 + contrib1
                if vdim < tile_v:
                    b_h1_bv1[:kdim, vdim:tile_v] = 0.0
                b_h1_bv1[kdim:, :] = 0.0

                # --- Band 2: second V tile [tile_v, 2*tile_v) → columns tile_v..min(2*tile_v, vdim)-1 in GLOBAL u ---
                # b_v2 [BT, 64]: same layout as b_v1; only first c2 columns used if V ≤ 128 (c2 = vdim - tile_v).
                b_v2 = torch.zeros(bt, tile_v, device=dev, dtype=torch.float32)
                if vdim > tile_v:
                    c2 = min(tile_v, vdim - tile_v)
                    b_v2[:span, :c2] = u[0, bos + t0 : bos + t1, i_h, tile_v : tile_v + c2].float()
                # v_new2 = u2 - W @ h2: w_pad [BT,K] @ b_h1_bv2 [128,64] → [BT,64] (same shapes as band 1).
                b_v_new2 = b_v2 - torch.matmul(w_pad, b_h1_bv2.to(wd)).to(torch.float32)
                if save_new_value and v_new is not None and vdim > tile_v:
                    c2 = min(tile_v, vdim - tile_v)
                    v_new[0, bos + t0 : bos + t1, i_h, tile_v : tile_v + c2] = b_v_new2[:span, :c2].to(
                        v_new.dtype
                    )

                if g_ht is not None:
                    # Same gating as band 1: row scale b_g_pad [BT] on v_new, scalar g_last on h tile.
                    b_v_new2 = b_v_new2 * b_g_pad[:, None]
                    b_h1_bv2 = b_h1_bv2 * b_g_last
                # K^T @ v_new on tile: k_pad [128, BT] @ b_v_new2_bf [BT, 64] → contrib2 [128, 64].
                b_v_new2_bf = b_v_new2.to(kd)
                contrib2 = torch.matmul(k_pad, b_v_new2_bf).to(torch.float32)
                b_h1_bv2 = b_h1_bv2 + contrib2
                if vdim > tile_v:
                    c2 = min(tile_v, vdim - tile_v)
                    # Zero padded V columns inside the 64-wide tile when V not multiple of 64.
                    if c2 < tile_v:
                        b_h1_bv2[:kdim, c2:tile_v] = 0.0
                # Zero padded K rows past kdim in the fixed 128×64 register tile.
                b_h1_bv2[kdim:, :] = 0.0

            if output_final_state and final_state is not None:
                final_state[i_n, i_h, :, :] = _pack_h_from_tiles(b_h1_bv1, b_h1_bv2, kdim, vdim, tile_v)

    return h_out, v_new, final_state


chunk_gated_delta_rule_fwd_h_explained = chunk_gated_delta_rule_fwd_h
