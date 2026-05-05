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

from pto_kernels import pto_gdn_chunk_o

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
    out = torch.zeros_like(g, dtype=torch.float32)
    for bos, eos in _seq_ranges(g.shape[0], cu_seqlens):
        for j in range(bos, eos, C):
            s, e = j, min(j + C, eos)
            out[s:e, :] = g[s:e, :].float().cumsum(dim=0)
    return out


# ---------------------------------------------------------------------------
# CPU fp32 references
# ---------------------------------------------------------------------------


def _ref_chunk_h(
    k: torch.Tensor,  # [T, Hg, D]
    w: torch.Tensor,  # [T, H, D]
    u: torch.Tensor,  # [T, H, D]
    g_cumsum: torch.Tensor,  # [T, H]
    cu_seqlens=None,
):
    """CPU fp32 reference for chunk_h — used to generate states for chunk_o tests.

    Returns:
        h_out   [total_chunks, H, D, D]  state snapshot at the start of each chunk
        v_new   [T, H, D]                residual-corrected values (U - W @ S)
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

    ci_base = 0
    for bos, eos in ranges:
        nc = (eos - bos + C - 1) // C
        for h in range(H_local):
            hg = h // grp
            S = torch.zeros(D, D, dtype=torch.float32)
            for ci in range(nc):
                s = bos + ci * C
                e = min(bos + (ci + 1) * C, eos)
                gc = gf[s:e, h]
                gl = gc[e - s - 1]
                h_out[ci_base + ci, h] = S.clone()
                vc = uf[s:e, h, :] - wf[s:e, h, :] @ S
                v_new[s:e, h, :] = vc
                kv = kf[s:e, hg, :].T @ (vc * torch.exp(gl - gc)[:, None])
                S = torch.exp(gl) * S + kv
        ci_base += nc

    return h_out, v_new


def ref_chunk_o(
    q: torch.Tensor,  # [T, Hg, D]
    k: torch.Tensor,  # [T, Hg, D]
    v_new: torch.Tensor,  # [T, H, D]
    h_states: torch.Tensor,  # [total_chunks, H, D, D]
    g_cumsum: torch.Tensor,  # [T, H]
    cu_seqlens=None,
) -> torch.Tensor:
    """CPU fp32 reference for chunk_o output computation.

    Per chunk:
        inter = exp(g) * (Q @ S)           — inter-chunk contribution from state
        intra = (Q @ K.T * gate * causal) @ V  — intra-chunk causal attention
        O = inter + intra

    Returns [T, H, D] float32.
    """
    T = q.shape[0]
    H_local = v_new.shape[1]
    grp = H_local // q.shape[1]

    qf = q.float()
    kf = k.float()
    vf = v_new.float()
    gf = g_cumsum.float()

    o = torch.zeros(T, H_local, D, dtype=torch.float32)
    ranges = _seq_ranges(T, cu_seqlens)
    ci_base = 0
    for bos, eos in ranges:
        nc = (eos - bos + C - 1) // C
        for h in range(H_local):
            hg = h // grp
            for ci in range(nc):
                s = bos + ci * C
                e = min(bos + (ci + 1) * C, eos)
                vlen = e - s
                qc = qf[s:e, hg, :]  # [vlen, D]
                kc = kf[s:e, hg, :]  # [vlen, D]
                vc = vf[s:e, h, :]  # [vlen, D]
                gc = gf[s:e, h]  # [vlen]
                inter = (qc @ h_states[ci_base + ci, h]) * torch.exp(gc)[:, None]
                qk = qc @ kc.T  # [vlen, vlen]
                causal = torch.arange(vlen)[:, None] >= torch.arange(vlen)[None, :]
                gate = torch.exp(
                    torch.minimum(gc[:, None] - gc[None, :], torch.zeros(vlen, vlen))
                )
                o[s:e, h, :] = inter + (qk * gate * causal.float()) @ vc
        ci_base += nc
    return o


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
def test_pto_gdn_chunk_o_fixed(npu_device, seq_len: int):
    torch.manual_seed(42)
    T = seq_len

    q_cpu = F.normalize(torch.randn(T, Hg, D, dtype=torch.float16), dim=-1, p=2)
    k_cpu = F.normalize(torch.randn(T, Hg, D, dtype=torch.float16), dim=-1, p=2)
    w_cpu = torch.randn(T, H, D, dtype=torch.float16)
    u_cpu = torch.randn(T, H, D, dtype=torch.float16)
    g_in = F.logsigmoid(torch.randn(T, H, dtype=torch.float32))
    g_cumsum = _chunk_cumsum(g_in)  # [T, H] fp32

    # Use CPU ref for chunk_h to generate states and v_new, then convert to fp16
    # to match the precision that the NPU kernel will operate on.
    h_out_f32, v_new_f32 = _ref_chunk_h(k_cpu, w_cpu, u_cpu, g_cumsum)
    h_out_fp16 = h_out_f32.half()  # [total_chunks, H, D, D]
    v_new_fp16 = v_new_f32.half()  # [T, H, D]

    tc_total = _count_total_chunks(T)
    G_npu = g_cumsum.T.contiguous().to(npu_device)  # [H, T] fp32
    Msk = torch.tril(torch.ones(C, C, dtype=torch.float32))  # [C, C]
    # chunk_o expects S as [total_chunks * H, D, D]
    S_npu = h_out_fp16.reshape(tc_total * H, D, D).to(npu_device)

    o_npu = pto_gdn_chunk_o(
        q_cpu.to(npu_device),
        k_cpu.to(npu_device),
        v_new_fp16.to(npu_device),
        S_npu,
        G_npu,
        Msk.to(npu_device),
        batch_size=1,
        seq_len=seq_len,
    )
    torch.npu.synchronize()

    # Reference uses fp16 inputs (cast back to float32) to match kernel precision
    o_ref = ref_chunk_o(q_cpu, k_cpu, v_new_fp16, h_out_fp16.float(), g_cumsum)

    assert stats_ok(
        o_npu.float().cpu(), o_ref
    ), "chunk_o output does not match reference"


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
def test_pto_gdn_chunk_o_varlen(npu_device, seqlens: list):
    torch.manual_seed(42)

    cu_seqlens_list = [0]
    for s in seqlens:
        cu_seqlens_list.append(cu_seqlens_list[-1] + s)

    T = cu_seqlens_list[-1]
    N_seq = len(seqlens)

    q_cpu = F.normalize(torch.randn(T, Hg, D, dtype=torch.float16), dim=-1, p=2)
    k_cpu = F.normalize(torch.randn(T, Hg, D, dtype=torch.float16), dim=-1, p=2)
    w_cpu = torch.randn(T, H, D, dtype=torch.float16)
    u_cpu = torch.randn(T, H, D, dtype=torch.float16)
    g_in = F.logsigmoid(torch.randn(T, H, dtype=torch.float32))
    g_cumsum = _chunk_cumsum(g_in, cu_seqlens_list)  # [T, H] fp32

    h_out_f32, v_new_f32 = _ref_chunk_h(k_cpu, w_cpu, u_cpu, g_cumsum, cu_seqlens_list)
    h_out_fp16 = h_out_f32.half()
    v_new_fp16 = v_new_f32.half()

    tc_total = _count_total_chunks(T, cu_seqlens_list)
    G_npu = g_cumsum.T.contiguous().to(npu_device)  # [H, T] fp32
    Msk = torch.tril(torch.ones(C, C, dtype=torch.float32))  # [C, C]
    cu_npu = torch.tensor(cu_seqlens_list, dtype=torch.int32).to(npu_device)
    S_npu = h_out_fp16.reshape(tc_total * H, D, D).to(npu_device)

    o_npu = pto_gdn_chunk_o(
        q_cpu.to(npu_device),
        k_cpu.to(npu_device),
        v_new_fp16.to(npu_device),
        S_npu,
        G_npu,
        Msk.to(npu_device),
        batch_size=N_seq,
        seq_len=0,
        cu_seqlens=cu_npu,
    )
    torch.npu.synchronize()

    o_ref = ref_chunk_o(
        q_cpu, k_cpu, v_new_fp16, h_out_fp16.float(), g_cumsum, cu_seqlens_list
    )

    assert stats_ok(
        o_npu.float().cpu(), o_ref
    ), "chunk_o varlen output does not match reference"
