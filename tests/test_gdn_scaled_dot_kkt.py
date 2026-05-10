# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from pto_kernels import pto_gdn_scaled_dot_kkt

# Compile-time kernel constants (default build: GDN_H=16, GDN_HG=16, GDN_D=128, GDN_C=128)
C = 128  # chunk size
D = 128  # head dimension
H = 16  # number of value heads
Hg = 16  # number of key heads (= H: no GQA in default build)

# Accuracy thresholds (from upstream reference test)
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


def _chunk_cumsum(g: torch.Tensor, cu_seqlens=None) -> torch.Tensor:
    """Per-chunk cumulative sum of gates (resets at sequence and chunk boundaries).

    g: [T, H] float32
    Returns [T, H] float32.
    """
    out = torch.zeros_like(g, dtype=torch.float32)
    for bos, eos in _seq_ranges(g.shape[0], cu_seqlens):
        for j in range(bos, eos, C):
            s, e = j, min(j + C, eos)
            out[s:e, :] = g[s:e, :].float().cumsum(dim=0)
    return out


def _safe_exp(x: torch.Tensor) -> torch.Tensor:
    """exp(min(x, 0)) — returns 0 for positive inputs, exp(x) otherwise."""
    return torch.where(x <= 0, torch.exp(x), torch.zeros_like(x))


def ref_scaled_dot_kkt(
    k: torch.Tensor,  # [T, Hg, D]
    beta: torch.Tensor,  # [T, H]
    g_cumsum: torch.Tensor,  # [T, H]
    cu_seqlens=None,
) -> torch.Tensor:
    """CPU fp32 reference for the scaled_dot_kkt kernel.

    Per chunk, for each head h:
        A[i, j] = (K[i] · K[j]) * exp(min(g[i] - g[j], 0)) * beta[i]
                  with strictly-lower-triangular mask (j < i only).

    Returns [T, H, C] float32 (zero-padded beyond the valid chunk width).
    """
    T = k.shape[0]
    H_local = beta.shape[1]
    grp = H_local // k.shape[1]

    kf = k.float()
    bf = beta.float()
    gf = g_cumsum.float()

    out = torch.zeros(T, H_local, C, dtype=torch.float32)

    for bos, eos in _seq_ranges(T, cu_seqlens):
        for j in range(bos, eos, C):
            s, e = j, min(j + C, eos)
            v = e - s
            for h in range(H_local):
                hg = h // grp
                kc = kf[s:e, hg, :]  # [v, D]
                gc = gf[s:e, h]  # [v]
                bc = bf[s:e, h]  # [v]
                blk = (kc @ kc.T) * _safe_exp(gc[:, None] - gc[None, :]) * bc[:, None]
                mask = torch.arange(v)[:, None] > torch.arange(v)[None, :]
                out[s:e, h, :v] = blk * mask.float()

    return out


# ---------------------------------------------------------------------------
# Statistical accuracy check
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


@pytest.mark.parametrize("seq_len", [128, 256, 384, 512])
def test_pto_gdn_scaled_dot_kkt_fixed(npu_device, seq_len: int):
    torch.manual_seed(42)
    T = seq_len

    k_cpu = F.normalize(torch.randn(T, Hg, D, dtype=torch.float16), dim=-1, p=2)
    beta_cpu = torch.rand(T, H, dtype=torch.float16)
    g_in = F.logsigmoid(torch.randn(T, H, dtype=torch.float32))
    g_cumsum = _chunk_cumsum(g_in)  # [T, H] fp32

    # Kernel expects Beta and G pre-transposed to [H, T]
    Beta_npu = beta_cpu.T.contiguous().to(npu_device)  # [H, T] fp16
    G_npu = g_cumsum.T.contiguous().to(npu_device)  # [H, T] fp32
    # Strictly lower-triangular causal mask (diagonal=-1)
    Msk = torch.tril(torch.ones(C, C, dtype=torch.float32), diagonal=-1)

    a_npu = pto_gdn_scaled_dot_kkt(
        k_cpu.to(npu_device),
        Beta_npu,
        G_npu,
        Msk.to(npu_device),
        batch_size=1,
        seq_len=seq_len,
    )
    torch.npu.synchronize()

    a_ref = ref_scaled_dot_kkt(k_cpu, beta_cpu, g_cumsum)

    assert stats_ok(
        a_npu.float().cpu(), a_ref
    ), "scaled_dot_kkt output does not match reference"


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
def test_pto_gdn_scaled_dot_kkt_varlen(npu_device, seqlens: list):
    torch.manual_seed(42)

    cu_seqlens_list = [0]
    for s in seqlens:
        cu_seqlens_list.append(cu_seqlens_list[-1] + s)

    T = cu_seqlens_list[-1]
    N_seq = len(seqlens)

    k_cpu = F.normalize(torch.randn(T, Hg, D, dtype=torch.float16), dim=-1, p=2)
    beta_cpu = torch.rand(T, H, dtype=torch.float16)
    g_in = F.logsigmoid(torch.randn(T, H, dtype=torch.float32))
    g_cumsum = _chunk_cumsum(g_in, cu_seqlens_list)  # [T, H] fp32

    Beta_npu = beta_cpu.T.contiguous().to(npu_device)  # [H, T] fp16
    G_npu = g_cumsum.T.contiguous().to(npu_device)  # [H, T] fp32
    Msk = torch.tril(torch.ones(C, C, dtype=torch.float32), diagonal=-1)
    cu_npu = torch.tensor(cu_seqlens_list, dtype=torch.int32).to(npu_device)

    a_npu = pto_gdn_scaled_dot_kkt(
        k_cpu.to(npu_device),
        Beta_npu,
        G_npu,
        Msk.to(npu_device),
        batch_size=N_seq,
        seq_len=0,
        cu_seqlens=cu_npu,
    )
    torch.npu.synchronize()

    a_ref = ref_scaled_dot_kkt(k_cpu, beta_cpu, g_cumsum, cu_seqlens_list)

    assert stats_ok(
        a_npu.float().cpu(), a_ref
    ), "scaled_dot_kkt varlen output does not match reference"
