# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------
"""Tests for kernel_kda_kkt: within-chunk gated attention matrix for KDA.

Math (per chunk, per head h, strictly lower-tri r > c):
  L[r, c] = beta[r] * sum_d k[r,d] * k[c,d] * exp(min(g_cs[r,d]-g_cs[c,d], 0))

All tensors passed to the kernel are in head-major layout:
  K    [H, total_tokens, D]  fp16
  G_cs [H, total_tokens, D]  fp32   within-chunk cumulative sum of log-gates
  Beta [H, total_tokens]     fp16   post-sigmoid scalar

Output L [total_tokens, H, C] fp16  BSND, strictly-lower-tri per chunk.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from pto_kernels import pto_kda_kkt

# Compile-time kernel constants (default build: GDN_H=4, GDN_D=128, GDN_C=128)
C = 128  # chunk size
D = 128  # head dimension
H = 4  # number of heads

RTOL = 1e-2
ATOL = 1e-5
MAX_RMSE_RATIO = 0.05
MIN_R2 = 0.99
HARD_FAIL_MAX = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seq_ranges(T: int, cu_seqlens=None):
    """Return (bos, eos) pairs for each sequence."""
    if cu_seqlens is None:
        return [(0, T)]
    cu = cu_seqlens if isinstance(cu_seqlens, list) else cu_seqlens.tolist()
    return [(cu[i], cu[i + 1]) for i in range(len(cu) - 1)]


def _chunk_cumsum_per_dim(g: torch.Tensor, cu_seqlens=None) -> torch.Tensor:
    """Within-chunk cumulative sum of per-dimension log-gates.

    g: [H, T, D] float32
    Returns [H, T, D] float32 — resets at sequence and chunk boundaries.
    """
    out = torch.zeros_like(g, dtype=torch.float32)
    _, T, _ = g.shape
    for bos, eos in _seq_ranges(T, cu_seqlens):
        for j in range(bos, eos, C):
            s, e = j, min(j + C, eos)
            out[:, s:e, :] = g[:, s:e, :].float().cumsum(dim=1)
    return out


def ref_kda_kkt(
    k: torch.Tensor,  # [H, T, D] float32
    g_cs: torch.Tensor,  # [H, T, D] float32
    beta: torch.Tensor,  # [H, T]    float32
    cu_seqlens=None,
) -> torch.Tensor:
    """CPU float32 reference for pto_kda_kkt.

    Returns L [T, H, C] float32, strictly-lower-tri per chunk.
    """
    _, T, _ = k.shape
    out = torch.zeros(T, H, C, dtype=torch.float32)

    for bos, eos in _seq_ranges(T, cu_seqlens):
        for j in range(bos, eos, C):
            s, e = j, min(j + C, eos)
            v = e - s
            for h in range(H):
                kc = k[h, s:e, :].float()  # [v, D]
                gc = g_cs[h, s:e, :].float()  # [v, D]
                bc = beta[h, s:e].float()  # [v]

                # diff[r, c, d] = g_cs[r,d] - g_cs[c,d], clamped to <= 0
                diff = gc[:, None, :] - gc[None, :, :]  # [v, v, D]
                diff = torch.clamp(diff, max=0.0)

                # prod[r, c] = sum_d k[r,d] * k[c,d] * exp(diff[r,c,d])
                prod = (kc[:, None, :] * kc[None, :, :] * diff.exp()).sum(-1)  # [v, v]

                # Scale by beta[r] and apply strict-lower-tri mask
                prod = prod * bc[:, None]
                mask = torch.arange(v)[:, None] > torch.arange(v)[None, :]
                out[s:e, h, :v] = prod * mask.float()

    return out


# ---------------------------------------------------------------------------
# Statistical accuracy check (shared with other kernel tests)
# ---------------------------------------------------------------------------


def _r2(y_ref: torch.Tensor, y_pred: torch.Tensor) -> float:
    ref = y_ref.detach().cpu().numpy().ravel().astype(np.float64)
    pred = y_pred.detach().cpu().numpy().ravel().astype(np.float64)
    ss_res = np.sum((ref - pred) ** 2)
    ss_tot = np.sum((ref - np.mean(ref)) ** 2)
    return float("nan") if ss_tot < 1e-30 else 1.0 - ss_res / ss_tot


def stats_ok(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    diff = (actual - expected).abs()
    if diff.max().item() > HARD_FAIL_MAX:
        return False
    if (diff <= ATOL + RTOL * expected.abs()).all():
        return True
    mean_abs = float(expected.float().flatten().abs().mean())
    rmse = float(torch.sqrt((diff.float().flatten() ** 2).mean()))
    ratio = rmse / max(mean_abs, 1e-15)
    r2 = _r2(expected, actual)
    if mean_abs < 1e-9:
        return rmse < 5e-4
    return ratio <= MAX_RMSE_RATIO and np.isfinite(r2) and r2 >= MIN_R2


# ---------------------------------------------------------------------------
# Input generation
# ---------------------------------------------------------------------------


def _make_inputs(T: int):
    """Return (k, g_cs, beta) in head-major [H, T, D] / [H, T] layout."""
    torch.manual_seed(42)
    # Normalised keys — avoids ill-conditioned L with large condition numbers
    k = F.normalize(torch.randn(H, T, D), dim=-1, p=2).float()
    # Production-like gates: cumulative sum can reach ~-64 per dim inside a chunk
    g_log = -torch.rand(H, T, D)  # log-gates in (-1, 0)
    g_cs = _chunk_cumsum_per_dim(g_log)  # [H, T, D] fp32
    beta = torch.sigmoid(torch.randn(H, T)).float()
    return k, g_cs, beta


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_len", [128, 256, 384, 512])
def test_kda_kkt_fixed(npu_device, seq_len: int):
    T = seq_len
    k, g_cs, beta = _make_inputs(T)

    k_npu = k.half().to(npu_device)  # [H, T, D] fp16
    g_cs_npu = g_cs.float().to(npu_device)  # [H, T, D] fp32
    beta_npu = beta.half().to(npu_device)  # [H, T]    fp16
    torch.npu.synchronize()

    L_npu = pto_kda_kkt(k_npu, g_cs_npu, beta_npu, batch_size=1, seq_len=seq_len)
    torch.npu.synchronize()

    L_ref = ref_kda_kkt(k, g_cs, beta)

    assert stats_ok(
        L_npu.float().cpu(), L_ref
    ), f"kda_kkt output does not match reference (seq_len={seq_len})"


@pytest.mark.parametrize(
    "seqlens",
    [
        [128],
        [256],
        [128, 256],
        [256, 128],
        [128, 128, 128],
        [256, 128, 384],
    ],
)
def test_kda_kkt_varlen(npu_device, seqlens: list):
    cu_seqlens_list = [0]
    for s in seqlens:
        cu_seqlens_list.append(cu_seqlens_list[-1] + s)

    T = cu_seqlens_list[-1]
    N_seq = len(seqlens)

    # Compute g_cs with varlen boundaries
    k = F.normalize(torch.randn(H, T, D), dim=-1, p=2).float()
    g_log = -torch.rand(H, T, D)
    g_cs = _chunk_cumsum_per_dim(g_log, cu_seqlens_list)
    beta = torch.sigmoid(torch.randn(H, T)).float()

    k_npu = k.half().to(npu_device)
    g_cs_npu = g_cs.float().to(npu_device)
    beta_npu = beta.half().to(npu_device)
    cu_npu = torch.tensor(cu_seqlens_list, dtype=torch.int32).to(npu_device)
    torch.npu.synchronize()

    L_npu = pto_kda_kkt(
        k_npu, g_cs_npu, beta_npu, batch_size=N_seq, seq_len=0, cu_seqlens=cu_npu
    )
    torch.npu.synchronize()

    L_ref = ref_kda_kkt(k, g_cs, beta, cu_seqlens_list)

    assert stats_ok(
        L_npu.float().cpu(), L_ref
    ), f"kda_kkt varlen output does not match reference (seqlens={seqlens})"
