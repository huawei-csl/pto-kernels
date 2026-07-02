# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------
"""Unit tests for the wy_kda NPU kernel.

Adapted from upstream test_kda_single_kernels.py::test_wy (line 299).
Reference: https://github.com/huawei-csl/megagdn-pto/blob/096ea8c41ae7a1c13fa25d45c9ad8093a75ca1e0/tests/test_kda_single_kernels.py#L299

The test isolates wy_kda correctness by feeding CPU-computed INV = (I+L)^{-1}
(via torch.linalg.inv) directly into the kernel, decoupling it from any
inversion-stage error.

Math (per chunk, per head):
  A2[r, c]      = INV[r, c] * beta[c]           (column-scale by beta)
  K_eff[c, d]   = k[c, d] * exp(g_cs[c, d])     (per-dim gate, KDA-specific)
  U[r, :] = A2 @ V                               (fp16, BSND layout)
  W[r, :] = A2 @ K_eff
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pto_kernels import pto_kda_wy

# Compile-time kernel constants (default build: GDN_H=16, GDN_D=128, GDN_C=128)
CHUNK = 128
K = 128
V_DIM = 128
H = 16

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


def _gate_cumsum(g_log: torch.Tensor, cu_seqlens=None) -> torch.Tensor:
    """Per-dim, per-chunk cumulative sum of gates (resets at seq and chunk boundaries).

    g_log: [1, T, H, K] — returns [1, T, H, K] float32.
    """
    T = g_log.shape[1]
    out = torch.zeros_like(g_log, dtype=torch.float32)
    for bos, eos in _seq_ranges(T, cu_seqlens):
        for j in range(bos, eos, CHUNK):
            s, e = j, min(j + CHUNK, eos)
            out[0, s:e] = g_log[0, s:e].float().cumsum(dim=0)
    return out


def _kkt_kda(
    k: torch.Tensor,
    g_cs: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens=None,
) -> torch.Tensor:
    """Compute the strictly lower-triangular L matrix per chunk.

    L[r, c] = beta[r] * dot(k[r], k[c] * exp(g_cs[r] - g_cs[c]))  for r > c

    k, g_cs: [1, T, H, K]; beta: [1, T, H]
    Returns L: [1, T, H, CHUNK] float32.
    """
    T, H_dim = k.shape[1], k.shape[2]
    L = torch.zeros(1, T, H_dim, CHUNK, dtype=torch.float32)
    for bos, eos in _seq_ranges(T, cu_seqlens):
        for j in range(bos, eos, CHUNK):
            s, e = j, min(j + CHUNK, eos)
            valid = e - s
            for h in range(H_dim):
                kf = k[0, s:e, h].float()  # [valid, K]
                gf = g_cs[0, s:e, h].float()  # [valid, K]
                bf = beta[0, s:e, h].float()  # [valid]
                # L[r, c] = beta[r] * (k[r]*exp(g[r])) @ (k[c]*exp(-g[c]))
                k_exp = kf * torch.exp(gf)  # [valid, K]
                k_nexp = kf * torch.exp(-gf)  # [valid, K]
                Lf = bf[:, None] * (k_exp @ k_nexp.T)  # [valid, valid]
                L[0, s:e, h, :valid] = torch.tril(Lf, diagonal=-1)
    return L


def _inversion_kda(L: torch.Tensor, cu_seqlens=None) -> torch.Tensor:
    """Compute INV = (I + L)^{-1} per chunk via torch.linalg.inv (double precision).

    L: [1, T, H, CHUNK] — returns INV: [1, T, H, CHUNK] float32.
    """
    T, H_dim = L.shape[1], L.shape[2]
    INV = torch.zeros_like(L)
    for bos, eos in _seq_ranges(T, cu_seqlens):
        for j in range(bos, eos, CHUNK):
            s, e = j, min(j + CHUNK, eos)
            valid = e - s
            for h in range(H_dim):
                Lc = L[0, s:e, h, :valid].float()
                inv = torch.linalg.inv(
                    torch.eye(valid, dtype=torch.float64) + Lc.double()
                ).float()
                INV[0, s:e, h, :valid] = inv
    return INV


def ref_wy_kda(
    k: torch.Tensor,
    v: torch.Tensor,
    g_cs: torch.Tensor,
    beta: torch.Tensor,
    INV: torch.Tensor,
    cu_seqlens=None,
):
    """CPU float32 reference for wy_kda.

    All tensors BSND: k/g_cs [1,T,H,K], v [1,T,H,V], beta [1,T,H], INV [1,T,H,CHUNK].
    Returns (u, w) each [1, T, H, V/K] float32.
    """
    T, H_dim, V_dim, K_dim = k.shape[1], k.shape[2], v.shape[3], k.shape[3]
    u = torch.zeros(1, T, H_dim, V_dim, dtype=torch.float32)
    w = torch.zeros(1, T, H_dim, K_dim, dtype=torch.float32)
    for bos, eos in _seq_ranges(T, cu_seqlens):
        for j in range(bos, eos, CHUNK):
            s, e = j, min(j + CHUNK, eos)
            valid = e - s
            for h in range(H_dim):
                inv_c = INV[0, s:e, h, :valid].float()  # [valid, valid]
                beta_c = beta[0, s:e, h].float()  # [valid]
                v_c = v[0, s:e, h].float()  # [valid, V]
                k_c = k[0, s:e, h].float()  # [valid, K]
                g_c = g_cs[0, s:e, h].float()  # [valid, K]
                A2 = inv_c * beta_c[None, :]  # column-scale: [valid, valid]
                K_eff = k_c * torch.exp(g_c)  # per-dim: [valid, K]
                u[0, s:e, h] = A2 @ v_c
                w[0, s:e, h] = A2 @ K_eff
    return u, w


def _make_inputs(T: int, cu_seqlens=None):
    """Generate well-conditioned inputs and compute CPU-reference INV."""
    torch.manual_seed(42)
    k = torch.randn(1, T, H, K)
    # L2-normalise keys: unnormalised keys produce ill-conditioned L matrices
    # (condition numbers ~1e6), making linalg.inv inaccurate in float32.
    k = k / k.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    v = torch.randn(1, T, H, V_DIM)
    g_log = -torch.rand(1, T, H, K)  # per-dim gates in (-1, 0)
    beta = torch.sigmoid(torch.randn(1, T, H))
    g_cs = _gate_cumsum(g_log, cu_seqlens)
    L = _kkt_kda(k, g_cs, beta, cu_seqlens)
    INV = _inversion_kda(L, cu_seqlens)
    return k, v, g_cs, beta, INV


# ---------------------------------------------------------------------------
# Accuracy helpers
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


def _to_npu_inputs(k, v, g_cs, beta, INV, device):
    """Convert CPU reference tensors to the layouts expected by pto_kda_wy.

    The kernel (and host wrapper) use head-major layout for K, G, Beta and
    BSND layout for V and INV — matching the convention in torch_gdn_wy_fast.h
    where Beta and G are already [H, T].

      k, g_cs : [1, T, H, D]  →  [H, T, D]  (squeeze batch, then permute)
      beta    : [1, T, H]     →  [H, T]
      v, INV  : [1, T, H, *]  →  [T, H, *]  (squeeze batch only)
    """
    K_npu = k[0].permute(1, 0, 2).contiguous().half().to(device)  # [H, T, D]
    V_npu = v[0].contiguous().half().to(device)  # [T, H, D]
    G_npu = g_cs[0].permute(1, 0, 2).contiguous().float().to(device)  # [H, T, D]
    Beta_npu = beta[0].permute(1, 0).contiguous().half().to(device)  # [H, T]
    INV_npu = INV[0].contiguous().half().to(device)  # [T, H, C]
    return K_npu, V_npu, G_npu, Beta_npu, INV_npu


@pytest.mark.parametrize("seq_len", [128, 256, 384, 512, 1024])
def test_pto_kda_wy_fixed(npu_device, seq_len: int):
    """Fixed-length sequence: compare wy_kda NPU output (u, w) to CPU reference."""
    T = seq_len
    k, v, g_cs, beta, INV = _make_inputs(T)
    u_ref, w_ref = ref_wy_kda(k, v, g_cs, beta, INV)

    K_npu, V_npu, G_npu, Beta_npu, INV_npu = _to_npu_inputs(
        k, v, g_cs, beta, INV, npu_device
    )
    u_npu, w_npu = pto_kda_wy(
        K_npu,
        V_npu,
        G_npu,
        Beta_npu,
        INV_npu,
        batch_size=1,
        seq_len=T,
    )
    torch.npu.synchronize()

    assert stats_ok(u_npu.float().cpu(), u_ref[0]), "U output does not match reference"
    assert stats_ok(w_npu.float().cpu(), w_ref[0]), "W output does not match reference"


@pytest.mark.parametrize(
    "seqlens",
    [
        [128],
        [256],
        [384],
        [512],
        [128, 256],
        [256, 128],
        [128, 128, 128],
        [256, 128, 384],
    ],
)
def test_pto_kda_wy_varlen(npu_device, seqlens: list):
    """Variable-length sequences: compare wy_kda NPU output (u, w) to CPU reference."""
    cu = [0]
    for s in seqlens:
        cu.append(cu[-1] + s)
    T = cu[-1]
    N_seq = len(seqlens)

    k, v, g_cs, beta, INV = _make_inputs(T, cu)
    u_ref, w_ref = ref_wy_kda(k, v, g_cs, beta, INV, cu)

    K_npu, V_npu, G_npu, Beta_npu, INV_npu = _to_npu_inputs(
        k, v, g_cs, beta, INV, npu_device
    )
    cu_npu = torch.tensor(cu, dtype=torch.int32).to(npu_device)
    u_npu, w_npu = pto_kda_wy(
        K_npu,
        V_npu,
        G_npu,
        Beta_npu,
        INV_npu,
        batch_size=N_seq,
        seq_len=0,
        cu_seqlens=cu_npu,
    )
    torch.npu.synchronize()

    assert stats_ok(
        u_npu.float().cpu(), u_ref[0]
    ), "U varlen output does not match reference"
    assert stats_ok(
        w_npu.float().cpu(), w_ref[0]
    ), "W varlen output does not match reference"
