"""
Educational emulation of ``chunk_local_cumsum`` (``fla_vendor/cumsum.py``).

Math: within each length-``chunk_size`` window along time, compute the prefix sum
:math:`G^{\\mathrm{cum}}_t = \\sum_{s=t_0}^{t} g_s` where :math:`t_0` is the chunk start.
This is the cumulative gate used later as :math:`e^{G}` in the gated delta rule.
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

    Global tensor: ``g`` is the full sequence gate (e.g. ``log \\sigma(\\cdot)``) in
    ``[B, T, H]`` layout when ``head_first=False``.

    For each conceptual tile (one time block), take a float32 slice on device and apply
    ``cumsum`` (optionally reversed), matching the Triton ``tl.cumsum`` over the block.
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

        acc = torch.cat(acc_list, dim=0) if acc_list else g_seg.new_zeros((0, h))
        out[0, bos:eos, :] = acc.to(out_dt)

    return out
