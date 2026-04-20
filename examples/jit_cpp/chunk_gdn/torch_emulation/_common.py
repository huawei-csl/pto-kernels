"""
Shared helpers for educational torch emulation of GDN Triton kernels.

``safe_exp`` matches ``fla_vendor.utils.safe_exp`` (Triton): exp(x) where x<=0, else 0.
This is the pairwise gate factor exp(g_i - g_j) with causal decay outside the valid cone.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch


def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """
    Match ``fla_vendor.utils.prepare_chunk_indices``: rows ``(seq_id, chunk_idx_in_seq)``
    for every ``chunk_size`` block along packed time (including partial tail chunks).
    """
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    nc = (lens + chunk_size - 1) // chunk_size
    parts = [torch.arange(int(n), device=cu_seqlens.device, dtype=torch.long) for n in nc.tolist()]
    indices = torch.cat(parts, dim=0) if parts else cu_seqlens.new_empty(0, dtype=torch.long)
    seq_ids = (indices == 0).cumsum(0) - 1
    return torch.stack([seq_ids, indices], dim=1).to(cu_seqlens)


def iter_packed_bt_chunks(
    *,
    cu_seqlens: torch.Tensor | None,
    total_t: int,
    bt: int,
    chunk_indices: torch.Tensor | None,
) -> Iterator[tuple[int, int, int]]:
    """
    Yield ``(bos, i_tc, span)`` for each block of width ``bt`` in Triton program order.

    ``bos`` is the sequence start offset in the packed ``[B, T, ...]`` tensor; ``i_tc`` is the
    chunk index within that sequence; ``global_slice = bos + i_tc * bt : bos + i_tc * bt + span``.
    ``span`` may be ``< bt`` for the last chunk of a sequence (or when ``total_t`` is not a
    multiple of ``bt`` and ``cu_seqlens is None``).
    """
    if cu_seqlens is None:
        nt = (total_t + bt - 1) // bt
        for i_tc in range(nt):
            span = min(bt, total_t - i_tc * bt)
            yield 0, i_tc, span
    else:
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, bt)
        for row in chunk_indices:
            i_n = int(row[0].item())
            i_tc = int(row[1].item())
            bos = int(cu_seqlens[i_n].item())
            eos = int(cu_seqlens[i_n + 1].item())
            t_seg = eos - bos
            span = min(bt, t_seg - i_tc * bt)
            yield bos, i_tc, span


def safe_exp_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0, torch.exp(x), torch.zeros_like(x))


def k_head_index(i_h: int, num_heads: int, num_k_heads: int) -> int:
    """Map output head ``i_h`` to key head index (GQA): ``i_h // (H // Hg)`` (see Triton kernels)."""
    return i_h // (num_heads // num_k_heads)


def tensor_r2_score(reference: torch.Tensor, prediction: torch.Tensor) -> float:
    """
    Coefficient of determination :math:`R^2` with ``reference`` as the ground truth (e.g. Triton).

    Uses the standard definition :math:`1 - \\mathrm{SS}_{\\mathrm{res}} / \\mathrm{SS}_{\\mathrm{tot}}`.
    If ``SS_tot`` is negligible (near-constant reference), returns ``1.0`` when residuals are tiny.
    """
    ref = reference.detach().float().reshape(-1)
    pred = prediction.detach().float().reshape(-1)
    ss_res = torch.sum((ref - pred) ** 2)
    mean_ref = ref.mean()
    ss_tot = torch.sum((ref - mean_ref) ** 2)
    if float(ss_tot.item()) < 1e-20:
        return 1.0 if float(ss_res.item()) < 1e-12 else 0.0
    return float((1.0 - ss_res / ss_tot).item())


def relative_rmse(reference: torch.Tensor, prediction: torch.Tensor) -> float:
    """
    :math:`\\mathrm{RMSE}(\\mathrm{ref}, \\mathrm{pred}) / \\sqrt{\\mathbb{E}[\\mathrm{ref}^2]}`.

    Scale-invariant vs the reference magnitude (Triton output).
    """
    ref = reference.detach().float().reshape(-1)
    pred = prediction.detach().float().reshape(-1)
    rmse = torch.sqrt(torch.mean((ref - pred) ** 2))
    denom = torch.sqrt(torch.mean(ref**2)).clamp(min=1e-30)
    return float((rmse / denom).item())
