"""
Educational emulation of ``chunk_local_cumsum`` (``fla_vendor/cumsum.py``).

Math: within each length-``chunk_size`` window along time, compute the prefix sum
:math:`G^{\\mathrm{cum}}_t = \\sum_{s=t_0}^{t} g_s` where :math:`t_0` is the chunk start.
This is the cumulative gate used later as :math:`e^{G}` in the gated delta rule.
"""

from __future__ import annotations

import numpy as np
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

    Global tensor: ``g`` is the full sequence gate (e.g. ``log \\sigma(\\cdot)``) in
    ``[B, T, H]`` layout when ``head_first=False``.

    For each SRAM conceptual tile (one time block), we copy the slice to a float32 numpy
    buffer, apply ``cumsum`` (optionally reversed), matching the Triton ``tl.cumsum``
    over the micro-chunks inside the optimization block.
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

    # --- Sequence boundaries (global metadata, host / DRAM) ---
    if cu_seqlens is None:
        ranges = [(0, t)]
    else:
        cu = cu_seqlens.detach().cpu().tolist()
        ranges = [(cu[i], cu[i + 1]) for i in range(len(cu) - 1)]

    for bos, eos in ranges:
        seg_len = eos - bos
        # GLOBAL view: one segment [seg_len, H] as torch for final write
        g_seg = g[0, bos:eos, :].float()

        acc_list = []
        for j in range(0, seg_len, chunk_size):
            e = min(j + chunk_size, seg_len)
            # SRAM tile: numpy copy of the micro-chunk (mirrors tl.load + reshape + cumsum path)
            tile_np = g_seg[j:e, :].detach().cpu().numpy().astype(np.float32).copy()
            # Prefix along time inside the chunk: G_cum[t] = sum_{s=j}^{t} g[s]
            if reverse:
                tile_np = np.flip(tile_np, axis=0)
                tile_np = np.cumsum(tile_np, axis=0)
                tile_np = np.flip(tile_np, axis=0)
            else:
                tile_np = np.cumsum(tile_np, axis=0)
            if scale is not None:
                tile_np = tile_np * float(scale)
            acc_list.append(torch.from_numpy(np.ascontiguousarray(tile_np)).to(device=g.device))

        acc = torch.cat(acc_list, dim=0) if acc_list else g_seg.new_zeros((0, h))
        out[0, bos:eos, :] = acc.to(out_dt)

    return out
