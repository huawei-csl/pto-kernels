"""CPU float32 reference implementations and unit tests for the kda_chunk_h stage."""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass

import torch

from ref_kda import RefKDA
from utils import NumericalAccuracy

from pto_kernels import pto_kda_chunk_h

ACCURACY = NumericalAccuracy()

CHUNK = 128
K = 128
V_DIM = 128

RTOL = 1e-2
ATOL = 1e-5
MAX_RMSE_RATIO = 0.05
MIN_R2 = 0.99


# ---------------------------------------------------------------------------
# Test-case registry
# ---------------------------------------------------------------------------


@dataclass
class TestCase:
    label: str
    cu_seqlens_list: list[int] | None
    T: int
    dtype: torch.dtype = torch.double

    def __post_init__(self):
        self.ref_kda = RefKDA(self.dtype)


def _cu_from_seqlens(seqlens: list[int]) -> list[int]:
    cu = [0]
    for s in seqlens:
        cu.append(cu[-1] + s)
    return cu


def _rand_cu(n_seq: int, total: int, rng: random.Random) -> list[int]:
    if n_seq == 1:
        return [0, total]
    bnd = sorted(rng.sample(range(1, total), n_seq - 1))
    return [0] + bnd + [total]


def _align_cu(raw: list[int], cs: int) -> list[int]:
    aligned = [0]
    for i in range(1, len(raw) - 1):
        val = ((raw[i] + cs - 1) // cs) * cs
        aligned.append(max(val, aligned[-1] + cs))
    total = max(raw[-1], aligned[-1] + cs)
    aligned.append(((total + cs - 1) // cs) * cs)
    return aligned


def _build_test_cases(quick: bool) -> list[TestCase]:
    if quick:
        return [TestCase("quick T=128", None, 128)]
    cases = []
    for T in [128, 256, 385, 512, 1024]:
        cases.append(TestCase(f"fixed T={T}", None, T))
    for seqlens in [
        [128],
        [256],
        [384],
        [512],
        [256, 256],
        [128, 256],
        [384, 128],
        [128, 128, 128],
        [256, 128, 384],
    ]:
        cu = _cu_from_seqlens(seqlens)
        cases.append(TestCase(f"varlen {seqlens}", cu, cu[-1]))
    for seqlens in [
        [1, 63, 64, 65, 127, 128, 129, 447],
        [1, 17, 31, 32, 33, 95, 127, 128, 129, 191, 192, 193, 367],
    ]:
        cu = _cu_from_seqlens(seqlens)
        cases.append(TestCase(f"varlen {seqlens}", cu, cu[-1]))
    rng = random.Random(42)
    for n_seq, total in [(3, 768), (7, 1792), (10, 2560)]:
        cu = _align_cu(_rand_cu(n_seq, total, rng), CHUNK)
        cases.append(TestCase(f"varlen rand {n_seq}seq T={cu[-1]}", cu, cu[-1]))
    return cases


def _make_inputs(tc: TestCase, H: int, HV: int | None = None):
    HV = HV or H
    T = tc.T
    torch.manual_seed(42)
    q = torch.randn(1, T, H, K)
    k = torch.randn(1, T, H, K)
    # L2-normalize q and k — unnormalized keys with K=128 give condition numbers
    # ~1e6 making float32 linalg.inv inaccurate.
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    k = k / k.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    v = torch.randn(1, T, HV, V_DIM)
    # Keep cumulative gate magnitudes within fp16 range.
    g_log = -torch.rand(1, T, HV, K) * 0.05
    beta_sig = torch.sigmoid(torch.randn(1, T, HV))
    scale = K**-0.5
    return q, k, v, g_log, beta_sig, scale


# ---------------------------------------------------------------------------
# chunk_h_kda test
# ---------------------------------------------------------------------------


def test_kda_chunk_h(tc: TestCase, H: int, dev=None) -> bool:
    """Compare kda_chunk_h NPU kernel output (s_snapshots, v_corr) to CPU ref."""
    if dev is None:
        raise ValueError("test_kda_chunk_h requires --device (NPU device).")

    _, k, v, g_log, beta_sig, _ = _make_inputs(tc, H)

    # Build CPU reference pipeline.
    g_cs = tc.ref_kda.gate_cumsum(g_log, CHUNK, tc.cu_seqlens_list)
    L_ref = tc.ref_kda.kkt_kda(k.float(), g_cs, beta_sig, CHUNK, tc.cu_seqlens_list)
    INV_ref = tc.ref_kda.inversion_kda(L_ref, CHUNK, tc.cu_seqlens_list)
    u_ref, w_ref = tc.ref_kda.wy_kda(
        k.float(), v.float(), g_cs, beta_sig, INV_ref, CHUNK, tc.cu_seqlens_list
    )
    s_ref, vcorr_ref = tc.ref_kda.chunk_h_kda(
        k.float(), u_ref, w_ref, g_cs, CHUNK, tc.cu_seqlens_list
    )

    s_npu = torch.zeros_like(s_ref, device=dev, dtype=torch.float16)
    vcorr_npu = torch.zeros(1, tc.T, H, V_DIM, device=dev, dtype=torch.float16)
    cu = (
        torch.tensor(tc.cu_seqlens_list, dtype=torch.int32, device=dev)
        if tc.cu_seqlens_list
        else None
    )
    N_seq = len(tc.cu_seqlens_list) - 1 if tc.cu_seqlens_list else 1
    stream = torch.npu.current_stream()._as_parameter_

    torch.npu.synchronize()
    run_chunk_h_kda(
        k.half().to(dev),
        w_ref.half().to(dev),
        u_ref.half().to(dev),
        g_cs.half().to(dev),
        s_npu,
        vcorr_npu,
        stream=stream,
        chunk_size=CHUNK,
        cu_seqlens=cu,
        batch_size_override=N_seq,
    )
    torch.npu.synchronize()

    return ACCURACY.stats_ok(s_npu.cpu(), s_ref) and ACCURACY.stats_ok(
        vcorr_npu.cpu(), vcorr_ref
    )
