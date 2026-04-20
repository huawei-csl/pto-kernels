"""
Educational emulation of ``chunk_local_cumsum`` (``fla_vendor/cumsum.py``).

Mathematics
-----------
Within each **sequence** (segment between ``cu_seqlens[i]`` and ``cu_seqlens[i+1]``), reset the
prefix sum at the segment start. Along time, within micro-windows of length ``chunk_size``,
compute the cumulative sum of the per-time gate (e.g. ``log σ(·)``):

.. math::

    G^{\\mathrm{cum}}_t = \\sum_{s = t_0}^{t} g_s

where ``t_0`` is the start of the **micro-tile** that contains ``t`` (concatenated tiles cover the
whole segment). **Important:** cumsum **resets at each tile boundary**—within ``[j, e)`` of length
``≤ chunk_size``, ``G`` is the prefix sum of ``g`` only inside that tile, not a full-segment
prefix from time 0 (matches ``tl.cumsum`` on each loaded tile separately). Optional ``reverse``
flips the tile before/after cumsum to match Triton’s direction. The result is the cumulative gate
fed into ``exp`` later in the GDN chain.

Memory: global vs tile
----------------------
**Global:**

- Input ``g``: ``[B, T, H]`` (this emulation requires ``B == 1`` when ``cu_seqlens`` is set).
- Output: same shape — **full** ``G^{cum}`` per position (DRAM).

**Tile:**

- ``tile``: shape ``[tile_len, H]`` where ``tile_len ≤ chunk_size`` — one micro-slice
  ``g_seg[j:e, :]`` in float32. This is the conceptual **SRAM strip** Triton loads before
  ``tl.cumsum``; results are concatenated and written to the **global** segment slice
  ``out[0, bos:eos, :]``.
"""

from __future__ import annotations

import torch


def chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
    **kwargs,
) -> torch.Tensor:
    """
    Same arguments as ``fla_vendor.cumsum.chunk_local_cumsum``.

    ``head_first=False``: ``g`` is ``[B, T, H]``.
    """
    if cu_seqlens is not None:
        assert g.shape[0] == 1, "Only batch size 1 is supported when cu_seqlens are provided"
    if len(g.shape) != 3:
        raise ValueError(
            f"Unsupported input shape {g.shape}, expected (B, T, H) with head_first=False"
        )
    if head_first:
        raise NotImplementedError("head_first emulation follows the same math; use Triton path if needed")

    out_dt = output_dtype if output_dtype is not None else g.dtype
    b, t, h = g.shape
    out = torch.empty(b, t, h, device=g.device, dtype=out_dt)

    # Sequence ranges in **global** packed time (metadata; indices only).
    if cu_seqlens is None:
        ranges = [(0, t)]
    else:
        cu = cu_seqlens.detach().cpu().tolist()
        ranges = [(cu[i], cu[i + 1]) for i in range(len(cu) - 1)]

    for bos, eos in ranges:
        seg_len = eos - bos
        # g_seg [seg_len, H]: GLOBAL segment in **packed** time (batch 0); one sequence per [bos,eos).
        g_seg = g[0, bos:eos, :].float()

        acc_list = []
        for j in range(0, seg_len, chunk_size):
            e = min(j + chunk_size, seg_len)
            tile_len = e - j
            # tile [tile_len, H]: local strip — conceptual SRAM after tl.load; cumsum along time only.
            tile = g_seg[j:e, :]
            if reverse:
                tile = torch.flip(tile, dims=[0])
                tile = torch.cumsum(tile, dim=0)
                tile = torch.flip(tile, dims=[0])
            else:
                tile = torch.cumsum(tile, dim=0)
            if scale is not None:
                tile = tile * scale
            acc_list.append(tile)

        # acc [seg_len, H]: concat tiles in order → full GLOBAL segment (same layout as g_seg).
        acc = torch.cat(acc_list, dim=0) if acc_list else g_seg.new_zeros((0, h))
        out[0, bos:eos, :] = acc.to(out_dt)

    return out
