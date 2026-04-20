"""
Shared helpers for educational torch/numpy emulation of GDN Triton kernels.

``safe_exp`` matches ``fla_vendor.utils.safe_exp`` (Triton): exp(x) where x<=0, else 0.
This is the pairwise gate factor exp(g_i - g_j) with causal decay outside the valid cone.
"""

from __future__ import annotations

import numpy as np
import torch


def safe_exp_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0, torch.exp(x), torch.zeros_like(x))


def safe_exp_np(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0, np.exp(x), np.zeros_like(x, dtype=np.float64))


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
