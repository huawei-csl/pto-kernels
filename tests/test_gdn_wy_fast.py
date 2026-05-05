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

from pto_kernels import pto_gdn_wy_fast

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


# ---------------------------------------------------------------------------
# CPU fp32 reference
# ---------------------------------------------------------------------------


def ref_wy_fast(
    k: torch.Tensor,  # [T, Hg, D]
    v: torch.Tensor,  # [T, H, D]
    beta: torch.Tensor,  # [T, H]
    A: torch.Tensor,  # [T, H, C]
    g_cumsum: torch.Tensor,  # [T, H]
    cu_seqlens=None,
):
    """CPU fp32 reference for the wy_fast kernel.

    Per chunk, for each head h and valid window [s:e]:
        A_b  = A[s:e, h, :valid]                              [valid, valid]
        v_b  = V[s:e, h, :] * beta[s:e, h, None]             [valid, D]
        k_b  = K[s:e, hg,:] * beta[s:e, h, None] * exp(g)    [valid, D]
        U[s:e, h, :] = A_b @ v_b
        W[s:e, h, :] = A_b @ k_b

    Returns (W, U) both float32 [T, H, D].
    """
    T = k.shape[0]
    H_local = v.shape[1]
    grp = H_local // k.shape[1]

    kf = k.float()
    vf = v.float()
    bf = beta.float()
    Af = A.float()
    gf = g_cumsum.float()

    w_out = torch.zeros(T, H_local, D, dtype=torch.float32)
    u_out = torch.zeros(T, H_local, D, dtype=torch.float32)

    for bos, eos in _seq_ranges(T, cu_seqlens):
        for j in range(bos, eos, C):
            s, e = j, min(j + C, eos)
            valid = e - s
            for h in range(H_local):
                hg = h // grp
                Ab = Af[s:e, h, :valid]  # [valid, valid]
                gc = gf[s:e, h]  # [valid]
                vb = vf[s:e, h, :] * bf[s:e, h, None]  # [valid, D]
                kb = (
                    kf[s:e, hg, :] * bf[s:e, h, None] * torch.exp(gc)[:, None]
                )  # [valid, D]
                u_out[s:e, h, :] = Ab @ vb
                w_out[s:e, h, :] = Ab @ kb

    return w_out, u_out


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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_len", [128, 256, 384, 512])
def test_pto_gdn_wy_fast_fixed(npu_device, seq_len: int):
    torch.manual_seed(42)
    T = seq_len

    k_cpu = F.normalize(torch.randn(T, Hg, D, dtype=torch.float16), dim=-1, p=2)
    v_cpu = torch.randn(T, H, D, dtype=torch.float16)
    beta_cpu = torch.rand(T, H, dtype=torch.float16)
    A_cpu = torch.randn(T, H, C, dtype=torch.float16)
    g_in = F.logsigmoid(torch.randn(T, H, dtype=torch.float32))
    g_cumsum = _chunk_cumsum(g_in)  # [T, H] fp32

    # Kernel expects Beta and G pre-transposed to [H, T]
    Beta_npu = beta_cpu.T.contiguous().to(npu_device)  # [H, T] fp16
    G_npu = g_cumsum.T.contiguous().to(npu_device)  # [H, T] fp32

    w_npu, u_npu = pto_gdn_wy_fast(
        k_cpu.to(npu_device),
        v_cpu.to(npu_device),
        Beta_npu,
        G_npu,
        A_cpu.to(npu_device),
        batch_size=1,
        seq_len=seq_len,
    )
    torch.npu.synchronize()

    w_ref, u_ref = ref_wy_fast(k_cpu, v_cpu, beta_cpu, A_cpu, g_cumsum)

    assert stats_ok(w_npu.float().cpu(), w_ref), "W output does not match reference"
    assert stats_ok(u_npu.float().cpu(), u_ref), "U output does not match reference"


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
def test_pto_gdn_wy_fast_varlen(npu_device, seqlens: list):
    torch.manual_seed(42)

    cu_seqlens_list = [0]
    for s in seqlens:
        cu_seqlens_list.append(cu_seqlens_list[-1] + s)

    T = cu_seqlens_list[-1]
    N_seq = len(seqlens)

    k_cpu = F.normalize(torch.randn(T, Hg, D, dtype=torch.float16), dim=-1, p=2)
    v_cpu = torch.randn(T, H, D, dtype=torch.float16)
    beta_cpu = torch.rand(T, H, dtype=torch.float16)
    A_cpu = torch.randn(T, H, C, dtype=torch.float16)
    g_in = F.logsigmoid(torch.randn(T, H, dtype=torch.float32))
    g_cumsum = _chunk_cumsum(g_in, cu_seqlens_list)  # [T, H] fp32

    Beta_npu = beta_cpu.T.contiguous().to(npu_device)  # [H, T] fp16
    G_npu = g_cumsum.T.contiguous().to(npu_device)  # [H, T] fp32
    cu_npu = torch.tensor(cu_seqlens_list, dtype=torch.int32).to(npu_device)

    w_npu, u_npu = pto_gdn_wy_fast(
        k_cpu.to(npu_device),
        v_cpu.to(npu_device),
        Beta_npu,
        G_npu,
        A_cpu.to(npu_device),
        batch_size=N_seq,
        seq_len=0,
        cu_seqlens=cu_npu,
    )
    torch.npu.synchronize()

    w_ref, u_ref = ref_wy_fast(k_cpu, v_cpu, beta_cpu, A_cpu, g_cumsum, cu_seqlens_list)

    assert stats_ok(
        w_npu.float().cpu(), w_ref
    ), "W varlen output does not match reference"
    assert stats_ok(
        u_npu.float().cpu(), u_ref
    ), "U varlen output does not match reference"
