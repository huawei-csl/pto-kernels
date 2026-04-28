"""
Educational emulation of ``recompute_w_u_fwd`` (``fla_vendor/wy_fast.py``).

Mathematics
-----------
Given packed lower-block matrix ``A`` (same layout as ``chunk_scaled_dot_kkt_fwd`` output: each
global time row holds one row of the local ``BT × BT`` block), and cumulative gate ``G`` on the
same times, compute **within each chunk**:

.. math::

    u_t = \\sum_{j < t} A_{tj}\\, \\beta_j v_j, \\qquad
    w_t = \\sum_{j < t} A_{tj}\\, \\beta_j\\, e^{G_j}\\, k_j

(block matrix multiply: ``u = A (β ⊙ v)``, ``w = A (β ⊙ e^G ⊙ k)`` in the causal lower part).

Memory: global vs tile
----------------------
**Global (DRAM):**

- ``k``: ``[B, T, Hg, K]``, ``v``: ``[B, T, H, V]``, ``beta``: ``[B, T, H]``.
- ``g_cumsum``: ``[B, T, H]`` — note: kernel uses **exp** of this when combining with ``k``.
- ``A``: ``[B, T, H, BT]`` — rows of the local triangular blocks as produced by KKT.

**Tiles (emulated on-chip blocks, float32 math then cast):**

- ``a_pad``: ``[BT, BT]`` — one chunk’s rows of ``A``; only ``[:span]`` rows filled from global,
  remainder **zero** (``tl.load`` + mask).
- ``v_pad``, ``k_pad``: ``[BT, V]`` and ``[BT, K]``; ``g_pad``, ``b_pad``: ``[BT]``.
- ``u_tile``, ``w_tile``: ``[BT, V]`` and ``[BT, K]`` — **matmul results** before ``tl.store``;
  only ``[:span]`` rows are written to global ``u`` and ``w``.

Partial chunks use the same ``iter_packed_bt_chunks`` schedule as KKT / Triton.
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

    Returns ``w`` with shape ``[B, T, H, K]``, ``u`` with shape ``[B, T, H, V]``.
    """
    b, t, hg, kdim = k.shape
    vdim = v.shape[-1]
    h = v.shape[-2]
    bt = A.shape[-1]

    # GLOBAL outputs (DRAM)
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
        # Global time of row 0 in this chunk: s .. s+span-1 (span ≤ BT).
        s = bos + _i_tc * bt
        for i_h in range(h):
            hk = k_head_index(i_h, h, hg)
            # --- Tile a_pad [BT, BT]: one chunk of lower-triangular block rows from GLOBAL A [B,T,H,BT] ---
            a_pad = torch.zeros(bt, bt, device=dev, dtype=torch.float32)
            a_pad[:span, :] = A[0, s : s + span, i_h, :].float()
            # --- Tile g_pad, b_pad [BT]: gate and β per timestep (zeros past span emulate mask) ---
            g_pad = torch.zeros(bt, device=dev, dtype=torch.float32)
            g_pad[:span] = g_cumsum[0, s : s + span, i_h].float()
            b_pad = torch.zeros(bt, device=dev, dtype=torch.float32)
            b_pad[:span] = beta[0, s : s + span, i_h].float()
            # exp_g: [BT], same layout as g_pad; multiplies k in the w recurrence (see kb below).
            exp_g = torch.exp(g_pad)

            # --- Tiles k_pad [BT, K], v_pad [BT, V]: GLOBAL k/v loaded into fixed-height chunk buffers ---
            k_pad = torch.zeros(bt, kdim, device=dev, dtype=torch.float32)
            k_pad[:span] = k[0, s : s + span, hk, :].float()
            v_pad = torch.zeros(bt, vdim, device=dev, dtype=torch.float32)
            v_pad[:span] = v[0, s : s + span, i_h, :].float()

            # β ⊙ v: b_pad[:, None] is [BT,1] → vb [BT, V] (broadcast multiply per row).
            vb = v_pad * b_pad[:, None]
            # u_tile [BT, V] = A [BT,BT] @ (β⊙v) [BT, V] — full matmul; causal zeros in A rows enforce j<t.
            u_tile = torch.matmul(a_pad, vb)
            # β ⊙ e^G ⊙ k: exp_g[:, None] is [BT,1] → kb [BT, K].
            kb = k_pad * b_pad[:, None] * exp_g[:, None]
            # w_tile [BT, K] = A @ kb; same comment as u_tile.
            w_tile = torch.matmul(a_pad, kb)

            # GLOBAL store: only timesteps s..s+span-1 (omit padded tail of each tile).
            u[0, s : s + span, i_h, :] = u_tile[:span, :].to(u.dtype)
            w[0, s : s + span, i_h, :] = w_tile[:span, :].to(w.dtype)

    return w, u
