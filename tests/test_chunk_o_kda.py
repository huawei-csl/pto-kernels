# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------
"""Unit tests for the chunk_o_kda NPU kernel.

Ported and adapted from
  huawei-csl/megagdn-pto @ f10b9f2 tests/test_kda_single_kernels.py
  (test_chunk_o_kda stage, lines 333–380).

Math (per chunk):
  q_eff = Q * exp(g_cs)
  k_eff = K * exp(-g_cs)
  Aqk   = tril(q_eff @ k_eff^T, diagonal=0)   (inclusive causal mask)
  O     = q_eff @ S + Aqk @ V_corr

Tensor layouts (NPU kernel conventions):
  Q, K, G — [HV, T, D]  head-major  fp16 / fp32
  V_corr  — [T, HV, D]  BSND        fp16
  S       — [total_chunks, HV, D, D] fp16
  Mask    — [C, C]       fp32, 1 where row >= col
  O out   — [T, HV, D]  BSND        fp16
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from pto_kernels import pto_chunk_o_kda

# Compile-time kernel constants (default build: GDN_H=16, GDN_D=128, GDN_C=128)
C = 128  # chunk size  (must match GDN_C)
D = 128  # head dimension
HV = 16  # number of heads

# Accuracy thresholds
RTOL = 1e-2
ATOL = 1e-5
MAX_RMSE_RATIO = 0.05
MIN_R2 = 0.99
HARD_FAIL_MAX = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seq_ranges(T: int, cu_seqlens=None):
    if cu_seqlens is None:
        return [(0, T)]
    cu = cu_seqlens if isinstance(cu_seqlens, list) else cu_seqlens.tolist()
    return [(cu[i], cu[i + 1]) for i in range(len(cu) - 1)]


def _count_total_chunks(T: int, cu_seqlens=None) -> int:
    return sum((eos - bos + C - 1) // C for bos, eos in _seq_ranges(T, cu_seqlens))


def _chunk_cumsum_kda(g: torch.Tensor, cu_seqlens=None) -> torch.Tensor:
    """Per-chunk cumulative sum of per-dim gates.

    g   : [T, HV, D] float32
    Returns [T, HV, D] float32 — resets at sequence and chunk boundaries.
    """
    T = g.shape[0]
    out = torch.zeros_like(g, dtype=torch.float32)
    for bos, eos in _seq_ranges(T, cu_seqlens):
        for j in range(bos, eos, C):
            s, e = j, min(j + C, eos)
            out[s:e] = g[s:e].float().cumsum(dim=0)
    return out


# ---------------------------------------------------------------------------
# CPU fp32 references
# ---------------------------------------------------------------------------


def _ref_chunk_h_kda(
    K: torch.Tensor,  # [T, HV, D] fp32
    W: torch.Tensor,  # [T, HV, D] fp32
    U: torch.Tensor,  # [T, HV, D] fp32
    G_cs: torch.Tensor,  # [T, HV, D] fp32
    cu_seqlens=None,
):
    """CPU fp32 reference for chunk_h_kda — generates (S, V_corr) for chunk_o tests."""
    T, HV_local, D_local = K.shape
    ranges = _seq_ranges(T, cu_seqlens)
    tc = _count_total_chunks(T, cu_seqlens)

    S_snap = torch.zeros(tc, HV_local, D_local, D_local, dtype=torch.float32)
    V_corr = torch.zeros(T, HV_local, D_local, dtype=torch.float32)

    ci_base = 0
    for bos, eos in ranges:
        nc = (eos - bos + C - 1) // C
        for h in range(HV_local):
            S = torch.zeros(D_local, D_local, dtype=torch.float32)
            for ci in range(nc):
                s = bos + ci * C
                e = min(bos + (ci + 1) * C, eos)

                g_cs = G_cs[s:e, h, :]
                g_total = g_cs[e - s - 1, :]

                S_snap[ci_base + ci, h] = S.clone()
                v_c = U[s:e, h, :] - W[s:e, h, :] @ S
                V_corr[s:e, h, :] = v_c
                k_rest = K[s:e, h, :] * torch.exp(g_total[None, :] - g_cs)
                S = torch.exp(g_total)[:, None] * S + k_rest.T @ v_c

        ci_base += nc

    return S_snap, V_corr


def ref_chunk_o_kda(
    Q: torch.Tensor,  # [T, HV, D] fp32
    K: torch.Tensor,  # [T, HV, D] fp32
    V_corr: torch.Tensor,  # [T, HV, D] fp32
    S_snap: torch.Tensor,  # [total_chunks, HV, D, D] fp32
    G_cs: torch.Tensor,  # [T, HV, D] fp32
    cu_seqlens=None,
) -> torch.Tensor:
    """CPU fp32 reference for the chunk_o_kda output computation.

    Per chunk:
        q_eff = Q * exp(g_cs)
        k_eff = K * exp(-g_cs)
        inter = q_eff @ S
        Aqk   = tril(q_eff @ k_eff^T)   (inclusive diagonal)
        O     = inter + Aqk @ V_corr

    Returns [T, HV, D] float32.
    """
    T, HV_local, D_local = Q.shape
    O = torch.zeros(T, HV_local, D_local, dtype=torch.float32)
    ranges = _seq_ranges(T, cu_seqlens)
    ci_base = 0

    for bos, eos in ranges:
        nc = (eos - bos + C - 1) // C
        for h in range(HV_local):
            for ci in range(nc):
                s = bos + ci * C
                e = min(bos + (ci + 1) * C, eos)

                g_cs = G_cs[s:e, h, :]  # [valid, D]
                q = Q[s:e, h, :]  # [valid, D]
                k = K[s:e, h, :]  # [valid, D]
                v = V_corr[s:e, h, :]  # [valid, D]
                S = S_snap[ci_base + ci, h]  # [D, D]

                q_eff = q * torch.exp(g_cs)  # [valid, D]
                k_eff = k * torch.exp(-g_cs)  # [valid, D]

                inter = q_eff @ S  # [valid, D]
                Aqk = torch.tril(q_eff @ k_eff.T)  # [valid, valid], inclusive

                O[s:e, h, :] = inter + Aqk @ v

        ci_base += nc

    return O


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
# Tests — fixed sequence length
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_len", [128, 256, 384, 512])
def test_chunk_o_kda_fixed(npu_device, seq_len: int):
    torch.manual_seed(42)
    T = seq_len

    q_cpu = F.normalize(torch.randn(T, HV, D), dim=-1, p=2).float()
    k_cpu = F.normalize(torch.randn(T, HV, D), dim=-1, p=2).float()
    w_cpu = torch.randn(T, HV, D, dtype=torch.float32)
    u_cpu = torch.randn(T, HV, D, dtype=torch.float32)
    # Small gate magnitudes keep exp(±g_cs) bounded, avoiding fp16 overflow.
    g_raw = -torch.rand(T, HV, D, dtype=torch.float32) * 0.05
    g_cs = _chunk_cumsum_kda(g_raw)  # [T, HV, D] fp32

    tc_total = _count_total_chunks(T)

    # Build S and V_corr via the CPU reference; round-trip to fp16 to match
    # the precision that the NPU kernel actually operates on.
    S_ref_f32, V_corr_ref_f32 = _ref_chunk_h_kda(k_cpu, w_cpu, u_cpu, g_cs)
    S_fp16 = S_ref_f32.half()  # [total_chunks, HV, D, D]
    V_corr_fp16 = V_corr_ref_f32.half()  # [T, HV, D]

    # Inclusive lower-triangular mask: 1 where row >= col.
    Mask = (torch.arange(C)[:, None] >= torch.arange(C)[None, :]).float()

    # NPU tensors — Q, K, G are head-major [HV, T, D]; V_corr and S use BSND / 4D.
    Q_npu = q_cpu.half().permute(1, 0, 2).contiguous().to(npu_device)  # [HV, T, D]
    K_npu = k_cpu.half().permute(1, 0, 2).contiguous().to(npu_device)  # [HV, T, D]
    G_npu = g_cs.permute(1, 0, 2).contiguous().to(npu_device)  # [HV, T, D]

    O_npu = pto_chunk_o_kda(
        Q_npu,
        K_npu,
        V_corr_fp16.to(npu_device),
        S_fp16.to(npu_device),
        G_npu,
        Mask.to(npu_device),
        batch_size=1,
        seq_len=seq_len,
        total_chunks=tc_total,
    )
    torch.npu.synchronize()

    # Reference uses fp16-rounded S and V_corr to match kernel precision.
    O_ref = ref_chunk_o_kda(q_cpu, k_cpu, V_corr_fp16.float(), S_fp16.float(), g_cs)

    assert stats_ok(
        O_npu.float().cpu(), O_ref
    ), "chunk_o_kda output does not match reference"


# ---------------------------------------------------------------------------
# Tests — variable-length sequences (cu_seqlens)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seqlens",
    [
        [128],
        [256],
        [128, 256],
        [256, 128],
        [128, 128, 128],
        [256, 128, 384],
        [384, 128],
        [128, 256, 256],
    ],
)
def test_chunk_o_kda_varlen(npu_device, seqlens: list):
    torch.manual_seed(42)

    cu = [0]
    for s in seqlens:
        cu.append(cu[-1] + s)
    T = cu[-1]
    N_seq = len(seqlens)

    q_cpu = F.normalize(torch.randn(T, HV, D), dim=-1, p=2).float()
    k_cpu = F.normalize(torch.randn(T, HV, D), dim=-1, p=2).float()
    w_cpu = torch.randn(T, HV, D, dtype=torch.float32)
    u_cpu = torch.randn(T, HV, D, dtype=torch.float32)
    g_raw = -torch.rand(T, HV, D, dtype=torch.float32) * 0.05
    g_cs = _chunk_cumsum_kda(g_raw, cu)  # [T, HV, D] fp32

    tc_total = _count_total_chunks(T, cu)

    S_ref_f32, V_corr_ref_f32 = _ref_chunk_h_kda(k_cpu, w_cpu, u_cpu, g_cs, cu)
    S_fp16 = S_ref_f32.half()
    V_corr_fp16 = V_corr_ref_f32.half()

    Mask = (torch.arange(C)[:, None] >= torch.arange(C)[None, :]).float()
    cu_npu = torch.tensor(cu, dtype=torch.int32).to(npu_device)

    Q_npu = q_cpu.half().permute(1, 0, 2).contiguous().to(npu_device)
    K_npu = k_cpu.half().permute(1, 0, 2).contiguous().to(npu_device)
    G_npu = g_cs.permute(1, 0, 2).contiguous().to(npu_device)

    O_npu = pto_chunk_o_kda(
        Q_npu,
        K_npu,
        V_corr_fp16.to(npu_device),
        S_fp16.to(npu_device),
        G_npu,
        Mask.to(npu_device),
        cu_seqlens=cu_npu,
        batch_size=N_seq,
        seq_len=0,
        total_chunks=tc_total,
    )
    torch.npu.synchronize()

    O_ref = ref_chunk_o_kda(q_cpu, k_cpu, V_corr_fp16.float(), S_fp16.float(), g_cs, cu)

    assert stats_ok(
        O_npu.float().cpu(), O_ref
    ), "chunk_o_kda varlen output does not match reference"
