"""
Pure PyTorch emulation of ``fla_vendor.chunk_delta_h.chunk_gated_delta_rule_fwd_h``.

Uses two float32 tiles ``b_h1_bv1`` and ``b_h1_bv2``, each ``128 × 64``,
matching ``tl.zeros([128, 64])``. Value indices ``[0, 64)`` map to the first tile, ``[64, 128)``
to the second. The second band loop still executes when ``V ≤ 64``; masked loads are zero but
internal FMAs can still update tile memory, so emulation must mirror both tiles.

Gates: ``safe_exp(G_last - G_t)`` on cumulative ``G``, and ``exp(G_last)`` for the state decay.
"""

from __future__ import annotations

import torch

from ._common import k_head_index, safe_exp_torch


def _prepare_chunk_offsets_cpu(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    nchunks = (lens + chunk_size - 1) // chunk_size
    z = cu_seqlens.new_zeros(1)
    return torch.cat([z, nchunks], dim=0).cumsum(-1)


def _prepare_chunk_indices_cpu(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    nc = (lens + chunk_size - 1) // chunk_size
    parts = [torch.arange(int(x), device=cu_seqlens.device, dtype=torch.long) for x in nc.tolist()]
    indices = torch.cat(parts, dim=0) if parts else cu_seqlens.new_empty(0, dtype=torch.long)
    seq_ids = (indices == 0).cumsum(0) - 1
    return torch.stack([seq_ids, indices], dim=1).to(cu_seqlens)


def _pack_h_from_tiles(
    b_h1_bv1: torch.Tensor,
    b_h1_bv2: torch.Tensor,
    kdim: int,
    vdim: int,
    tile_v: int,
) -> torch.Tensor:
    """Map two 128×64 tiles to ``h`` of shape ``[K, V]`` (float32)."""
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
    tile_k, tile_v = 128, 64

    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = _prepare_chunk_indices_cpu(cu_seqlens, chunk_size)
    if cu_seqlens is None:
        n, nt = b, (t_max + bt - 1) // bt
        chunk_offsets_t = None
    else:
        if chunk_offsets is None:
            chunk_offsets_t = _prepare_chunk_offsets_cpu(cu_seqlens, bt)
        else:
            chunk_offsets_t = chunk_offsets
        n = len(cu_seqlens) - 1
        nt = len(chunk_indices)

    h_out = k.new_empty(b, nt, h_heads, kdim, vdim)
    v_new = torch.empty_like(u) if save_new_value else None
    final_state = k.new_empty(n, h_heads, kdim, vdim, dtype=torch.float32) if output_final_state else None

    g_ht = g.transpose(1, 2).contiguous() if g is not None else None

    cu_list = cu_seqlens.detach().cpu().tolist() if cu_seqlens is not None else None

    for i_n in range(n if cu_seqlens is not None else b):
        if cu_seqlens is not None:
            bos, eos = cu_list[i_n], cu_list[i_n + 1]
            t_seg = eos - bos
            boh = int(chunk_offsets_t[i_n].item())
            nt_loc = (t_seg + bt - 1) // bt
        else:
            bos, eos = i_n * t_max, (i_n + 1) * t_max
            t_seg = t_max
            boh = i_n * ((t_max + bt - 1) // bt)
            nt_loc = (t_max + bt - 1) // bt

        for i_h in range(h_heads):
            hk = k_head_index(i_h, h_heads, hg)
            wd, kd = w.dtype, k.dtype

            b_h1_bv1 = torch.zeros(tile_k, tile_v, device=k.device, dtype=torch.float32)
            b_h1_bv2 = torch.zeros(tile_k, tile_v, device=k.device, dtype=torch.float32)

            if initial_state is not None:
                h0 = initial_state[i_n, i_h, :, :].float()
                b_h1_bv1[:kdim, : min(tile_v, vdim)] += h0[:, : min(tile_v, vdim)]
                if vdim > tile_v:
                    b_h1_bv2[:kdim, : min(tile_v, vdim - tile_v)] += h0[:, tile_v : vdim]

            for i_tc in range(nt_loc):
                h_out[0, boh + i_tc, i_h, :, :] = _pack_h_from_tiles(
                    b_h1_bv1, b_h1_bv2, kdim, vdim, tile_v
                ).to(h_out.dtype)

                t0 = i_tc * bt
                t1 = min(t0 + bt, t_seg)
                span = t1 - t0
                dev = k.device

                w_pad = torch.zeros(bt, tile_k, device=dev, dtype=wd)
                w_pad[:span, :kdim] = w[0, bos + t0 : bos + t1, i_h, :]

                k_pad = torch.zeros(tile_k, bt, device=dev, dtype=kd)
                k_pad[:kdim, :span] = k[0, bos + t0 : bos + t1, hk, :].T

                if g_ht is not None:
                    g_last_scalar = g_ht[0, i_h, bos + t1 - 1].float()
                    g_chunk = g_ht[0, i_h, bos + t0 : bos + t1].float()
                    b_g = safe_exp_torch(g_last_scalar - g_chunk)
                    b_g_last = torch.exp(g_last_scalar)
                    b_g_pad = torch.zeros(bt, device=dev, dtype=torch.float32)
                    b_g_pad[:span] = b_g
                else:
                    b_g_pad = torch.ones(bt, device=dev, dtype=torch.float32)
                    b_g_last = torch.tensor(1.0, device=dev, dtype=torch.float32)

                # --- Band 1: v ∈ [0, 64) ---
                b_v1 = torch.zeros(bt, tile_v, device=dev, dtype=torch.float32)
                c1 = min(tile_v, vdim)
                b_v1[:span, :c1] = u[0, bos + t0 : bos + t1, i_h, :c1].float()
                # tl.dot(b_w, b_h1_bv1.to(b_w.dtype)): match bf16×bf16 → fp32 accum
                b_v_new1 = b_v1 - torch.matmul(w_pad, b_h1_bv1.to(wd)).to(torch.float32)
                if save_new_value and v_new is not None:
                    v_new[0, bos + t0 : bos + t1, i_h, :c1] = b_v_new1[:span, :c1].to(v_new.dtype)

                if g_ht is not None:
                    b_v_new1 = b_v_new1 * b_g_pad[:, None]
                    b_h1_bv1 = b_h1_bv1 * b_g_last
                b_v_new1_bf = b_v_new1.to(kd)
                # tl.dot(b_k, b_v_new1): k and v_new in key dtype; accumulate in fp32
                contrib1 = torch.matmul(k_pad, b_v_new1_bf).to(torch.float32)
                b_h1_bv1 = b_h1_bv1 + contrib1
                # Mask unused V columns in the tile (Triton loads u with mask; no signal past vdim)
                if vdim < tile_v:
                    b_h1_bv1[:kdim, vdim:tile_v] = 0.0
                b_h1_bv1[kdim:, :] = 0.0

                # --- Band 2: v ∈ [64, 128) ---
                b_v2 = torch.zeros(bt, tile_v, device=dev, dtype=torch.float32)
                if vdim > tile_v:
                    c2 = min(tile_v, vdim - tile_v)
                    b_v2[:span, :c2] = u[0, bos + t0 : bos + t1, i_h, tile_v : tile_v + c2].float()
                b_v_new2 = b_v2 - torch.matmul(w_pad, b_h1_bv2.to(wd)).to(torch.float32)
                if save_new_value and v_new is not None and vdim > tile_v:
                    c2 = min(tile_v, vdim - tile_v)
                    v_new[0, bos + t0 : bos + t1, i_h, tile_v : tile_v + c2] = b_v_new2[:span, :c2].to(
                        v_new.dtype
                    )

                if g_ht is not None:
                    b_v_new2 = b_v_new2 * b_g_pad[:, None]
                    b_h1_bv2 = b_h1_bv2 * b_g_last
                b_v_new2_bf = b_v_new2.to(kd)
                contrib2 = torch.matmul(k_pad, b_v_new2_bf).to(torch.float32)
                b_h1_bv2 = b_h1_bv2 + contrib2
                if vdim > tile_v:
                    c2 = min(tile_v, vdim - tile_v)
                    if c2 < tile_v:
                        b_h1_bv2[:kdim, c2:tile_v] = 0.0
                b_h1_bv2[kdim:, :] = 0.0

            if output_final_state and final_state is not None:
                final_state[i_n, i_h, :, :] = _pack_h_from_tiles(b_h1_bv1, b_h1_bv2, kdim, vdim, tile_v)

    return h_out, v_new, final_state


# Backward-compatible alias
chunk_gated_delta_rule_fwd_h_explained = chunk_gated_delta_rule_fwd_h
