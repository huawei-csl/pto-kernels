"""
PyTorch references for vLLM ``[B, T, H, …]`` layout (small-shape checks).

- ``ref_chunk_local_cumsum``: chunk-local prefix sum along T (blocks of ``chunk_size``).
- ``ref_scaled_dot_kkt_bthd``: strict-lower KKT blocks; output layout ``[B, T, H, BT]``
  consistent with ``chunk_scaled_dot_kkt_fwd``.
"""
from __future__ import annotations

import torch


def _safe_exp_gate_diff(x: torch.Tensor) -> torch.Tensor:
    """Match ``utils.safe_exp`` applied to pairwise ``g[t]-g[s]`` in KKT."""
    return torch.where(x <= 0, torch.exp(x), torch.zeros_like(x))


def ref_chunk_local_cumsum(
    g: torch.Tensor, chunk_size: int, cu_seqlens: torch.Tensor | None
) -> torch.Tensor:
    """Chunk-local cumulative sum within each length-``chunk_size`` window along T."""
    B, T, H = g.shape
    assert B == 1
    out = torch.empty_like(g, dtype=torch.float32)
    g32 = g.float()
    ranges: list[tuple[int, int]]
    if cu_seqlens is None:
        ranges = [(0, T)]
    else:
        cu = cu_seqlens.cpu().tolist()
        ranges = [(cu[i], cu[i + 1]) for i in range(len(cu) - 1)]

    for bos, eos in ranges:
        seg = g32[0, bos:eos, :]
        L = eos - bos
        acc = torch.empty_like(seg)
        for j in range(0, L, chunk_size):
            e = min(j + chunk_size, L)
            acc[j:e] = seg[j:e].cumsum(dim=0)
        out[0, bos:eos, :] = acc
    return out


def ref_scaled_dot_kkt_bthd(
    k: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    """Reference KKT in ``[B, T, H, BT]`` layout (Hg == H)."""
    B, T, H, Kdim = k.shape
    out = torch.zeros(B, T, H, chunk_size, device=k.device, dtype=torch.float32)
    kf = k.float()
    beta_f = beta.float()
    gf = g_cumsum.float()

    def fill_seg(bos: int, eos: int):
        for i in range((eos - bos) // chunk_size):
            s = bos + i * chunk_size
            e = s + chunk_size
            k_c = kf[:, s:e, :, :]
            g_c = gf[:, s:e, :]
            b_c = beta_f[:, s:e, :]
            for h in range(H):
                kc = k_c[0, :, h, :].float()
                kk = kc @ kc.T
                gam = g_c[0, :, h].unsqueeze(-1) - g_c[0, :, h].unsqueeze(-2)
                blk = kk * _safe_exp_gate_diff(gam)
                blk = blk * b_c[0, :, h].unsqueeze(-1)
                bt = blk.shape[0]
                mask = (
                    torch.arange(bt, device=blk.device)[:, None]
                    > torch.arange(bt, device=blk.device)[None, :]
                )
                blk = blk * mask.to(blk.dtype)
                out[:, s:e, h, :].copy_(blk)

    if cu_seqlens is None:
        fill_seg(0, T - (T % chunk_size))
    else:
        cu = cu_seqlens.cpu().tolist()
        for i in range(len(cu) - 1):
            bos, eos = cu[i], cu[i + 1]
            fill_seg(bos, eos - ((eos - bos) % chunk_size))

    return out
