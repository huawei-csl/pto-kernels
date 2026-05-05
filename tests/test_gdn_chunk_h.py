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

from pto_kernels import pto_chunk_h

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


def _count_total_chunks(T: int, cu_seqlens=None) -> int:
    """Total number of C-sized chunks across all sequences."""
    return sum((eos - bos + C - 1) // C for bos, eos in _seq_ranges(T, cu_seqlens))


def _chunk_cumsum(g: torch.Tensor, cu_seqlens=None) -> torch.Tensor:
    """Per-chunk cumulative sum of gates (resets at sequence and chunk boundaries).

    g: [T, H] float32
    Returns [T, H] float32.
    """
    T, _ = g.shape
    out = torch.zeros_like(g, dtype=torch.float32)
    for bos, eos in _seq_ranges(T, cu_seqlens):
        for j in range(bos, eos, C):
            s, e = j, min(j + C, eos)
            out[s:e, :] = g[s:e, :].float().cumsum(dim=0)
    return out


# ---------------------------------------------------------------------------
# CPU fp32 reference
# ---------------------------------------------------------------------------


def ref_chunk_h(
    k: torch.Tensor,  # [T, Hg, D]
    w: torch.Tensor,  # [T, H, D]
    u: torch.Tensor,  # [T, H, D]
    g_cumsum: torch.Tensor,  # [T, H]
    cu_seqlens=None,
):
    """CPU fp32 reference for the chunk_h recurrence.

    Advances the D×D hidden state S chunk by chunk:
        ws    = W @ S
        v_new = U - ws
        kv    = (exp(g_last - g) * K).T @ v_new
        S     = exp(g_last) * S + kv

    Returns:
        h_out        [total_chunks, H, D, D]  state snapshot at chunk start
        v_new        [T, H, D]                residual-corrected values
        final_states [N_seq, H, D, D]         final state per sequence
    """
    T = k.shape[0]
    H_local = w.shape[1]
    grp = H_local // k.shape[1]

    kf = k.float()
    wf = w.float()
    uf = u.float()
    gf = g_cumsum.float()

    ranges = _seq_ranges(T, cu_seqlens)
    tc = _count_total_chunks(T, cu_seqlens)

    h_out = torch.zeros(tc, H_local, D, D, dtype=torch.float32)
    v_new = torch.zeros(T, H_local, D, dtype=torch.float32)
    final_states = torch.zeros(len(ranges), H_local, D, D, dtype=torch.float32)

    ci_base = 0
    for si, (bos, eos) in enumerate(ranges):
        nc = (eos - bos + C - 1) // C
        for h in range(H_local):
            hg = h // grp
            S = torch.zeros(D, D, dtype=torch.float32)
            for ci in range(nc):
                s = bos + ci * C
                e = min(bos + (ci + 1) * C, eos)
                gc = gf[s:e, h]  # [valid]
                gl = gc[e - s - 1]  # last gate in chunk
                h_out[ci_base + ci, h] = S.clone()
                vc = uf[s:e, h, :] - wf[s:e, h, :] @ S
                v_new[s:e, h, :] = vc
                kv = kf[s:e, hg, :].T @ (vc * torch.exp(gl - gc)[:, None])
                S = torch.exp(gl) * S + kv
            final_states[si, h] = S
        ci_base += nc

    return h_out, v_new, final_states


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
def test_pto_chunk_h_fixed(npu_device, seq_len: int):
    torch.manual_seed(42)
    T = seq_len

    k_cpu = F.normalize(torch.randn(T, Hg, D, dtype=torch.float16), dim=-1, p=2)
    w_cpu = torch.randn(T, H, D, dtype=torch.float16)
    u_cpu = torch.randn(T, H, D, dtype=torch.float16)
    g_in = F.logsigmoid(torch.randn(T, H, dtype=torch.float32))
    g_cumsum = _chunk_cumsum(g_in)  # [T, H] fp32
    G_npu = g_cumsum.T.contiguous().to(npu_device)  # [H, T] fp32

    tc_total = _count_total_chunks(T)

    s_out, v_out, _fs_out = pto_chunk_h(
        k_cpu.to(npu_device),
        w_cpu.to(npu_device),
        u_cpu.to(npu_device),
        G_npu,
        batch_size=1,
        seq_len=seq_len,
        total_chunks=tc_total,
    )
    torch.npu.synchronize()

    h_ref, v_ref, _ = ref_chunk_h(k_cpu, w_cpu, u_cpu, g_cumsum)

    assert stats_ok(
        s_out.float().cpu(), h_ref
    ), "State snapshots do not match reference"
    assert stats_ok(
        v_out.float().cpu(), v_ref
    ), "Residual values do not match reference"


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
def test_pto_chunk_h_varlen(npu_device, seqlens: list):
    torch.manual_seed(42)

    cu_seqlens_list = [0]
    for s in seqlens:
        cu_seqlens_list.append(cu_seqlens_list[-1] + s)

    T = cu_seqlens_list[-1]
    N_seq = len(seqlens)

    k_cpu = F.normalize(torch.randn(T, Hg, D, dtype=torch.float16), dim=-1, p=2)
    w_cpu = torch.randn(T, H, D, dtype=torch.float16)
    u_cpu = torch.randn(T, H, D, dtype=torch.float16)
    g_in = F.logsigmoid(torch.randn(T, H, dtype=torch.float32))
    g_cumsum = _chunk_cumsum(g_in, cu_seqlens_list)  # [T, H] fp32
    G_npu = g_cumsum.T.contiguous().to(npu_device)  # [H, T] fp32

    cu_npu = torch.tensor(cu_seqlens_list, dtype=torch.int32).to(npu_device)
    tc_total = _count_total_chunks(T, cu_seqlens_list)

    s_out, v_out, _fs_out = pto_chunk_h(
        k_cpu.to(npu_device),
        w_cpu.to(npu_device),
        u_cpu.to(npu_device),
        G_npu,
        cu_seqlens=cu_npu,
        batch_size=N_seq,
        seq_len=0,
        total_chunks=tc_total,
    )
    torch.npu.synchronize()

    h_ref, v_ref, _ = ref_chunk_h(k_cpu, w_cpu, u_cpu, g_cumsum, cu_seqlens_list)

    assert stats_ok(
        s_out.float().cpu(), h_ref
    ), "State snapshots do not match reference"
    assert stats_ok(
        v_out.float().cpu(), v_ref
    ), "Residual values do not match reference"
