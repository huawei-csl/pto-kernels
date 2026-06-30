# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------
"""Unit tests for the kda_gate_cumsum NPU kernel.

Ported and adapted from
  huawei-csl/megagdn-pto @ f10b9f2 tests/test_kda_single_kernels.py
  (test_gate_cumsum stage).

Mathematical operation (per chunk of C tokens, per head h, per key-dim d):
  g_sum[t, h, d] = Σ_{i=0}^{t} g[i, h, d]    for t = 0 .. valid-1

Differs from pto_chunk_cumsum (GDN) in two ways:
  - input dtype is fp16 (not fp32) — accumulated in fp32 inside the kernel
  - gate tensor is 3D [T, HV, D] (per-dim gate) instead of 2D [T, H]
"""

import random

import pytest
import torch

from pto_kernels import pto_kda_gate_cumsum

# Compile-time constants — must match GDN_H, GDN_D, GDN_C baked into the kernel binary.
GDN_H = 16  # number of heads
GDN_D = 128  # key/gate dimension per head
GDN_C = 128  # chunk size in tokens

random.seed(42)
torch.manual_seed(42)


def ref_kda_gate_cumsum(
    g: torch.Tensor,  # [T, HV, D] fp16
    cu_seqlens=None,
    batch_size: int = 1,
    seq_len: int = 0,
) -> torch.Tensor:
    """CPU reference: per-chunk prefix sum (reset at chunk/sequence boundaries).

    Casts g from fp16 → fp32 before accumulating (matching kernel behaviour)
    so reference and kernel use identical precision.

    Returns [T, HV, D] float32.
    """
    g_f32 = g.float()
    out = torch.zeros_like(g_f32)

    if cu_seqlens is None:
        for b in range(batch_size):
            bos = b * seq_len
            for j in range(0, seq_len, GDN_C):
                s = bos + j
                e = min(s + GDN_C, bos + seq_len)
                out[s:e] = g_f32[s:e].cumsum(dim=0)
    else:
        cu = cu_seqlens.tolist() if hasattr(cu_seqlens, "tolist") else list(cu_seqlens)
        for i in range(len(cu) - 1):
            bos, eos = cu[i], cu[i + 1]
            for j in range(0, eos - bos, GDN_C):
                s = bos + j
                e = min(s + GDN_C, eos)
                out[s:e] = g_f32[s:e].cumsum(dim=0)

    return out


def _cu_from_seqlens(seqlens: list) -> list:
    cu = [0]
    for s in seqlens:
        cu.append(cu[-1] + s)
    return cu


def _rand_cu_aligned(n_seq: int, total: int, chunk_size: int) -> list:
    rng = random.Random(42)
    if n_seq == 1:
        aligned_total = ((total + chunk_size - 1) // chunk_size) * chunk_size
        return [0, aligned_total]
    raw = sorted(rng.sample(range(1, total), n_seq - 1))
    aligned = [0]
    for v in raw:
        v_aligned = ((v + chunk_size - 1) // chunk_size) * chunk_size
        aligned.append(max(v_aligned, aligned[-1] + chunk_size))
    total_aligned = max(total, aligned[-1] + chunk_size)
    total_aligned = ((total_aligned + chunk_size - 1) // chunk_size) * chunk_size
    aligned.append(total_aligned)
    return aligned


@pytest.mark.parametrize("T", [128, 256, 384, 512, 1024])
def test_kda_gate_cumsum_fixed(npu_device, T: int):
    torch.manual_seed(42)
    # Production-like gate magnitude: values in (-1, 0) per dim.
    g_cpu = -torch.rand(T, GDN_H, GDN_D, dtype=torch.float16)

    result = pto_kda_gate_cumsum(g_cpu.to(npu_device), batch_size=1, seq_len=T).cpu()
    torch.npu.synchronize()

    expected = ref_kda_gate_cumsum(g_cpu, batch_size=1, seq_len=T)
    torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-4)


_VARLEN_SEQLENS = [
    [128],
    [256],
    [384],
    [512],
    [256, 256],
    [128, 256],
    [384, 128],
    [128, 128, 128],
    [256, 128, 384],
]


@pytest.mark.parametrize(
    "seqlens",
    _VARLEN_SEQLENS,
    ids=[str(s) for s in _VARLEN_SEQLENS],
)
def test_kda_gate_cumsum_varlen(npu_device, seqlens: list):
    cu_list = _cu_from_seqlens(seqlens)
    T = cu_list[-1]
    N_seq = len(seqlens)

    torch.manual_seed(42)
    g_cpu = -torch.rand(T, GDN_H, GDN_D, dtype=torch.float16)
    cu_npu = torch.tensor(cu_list, dtype=torch.int32).to(npu_device)

    result = pto_kda_gate_cumsum(
        g_cpu.to(npu_device), batch_size=N_seq, seq_len=0, cu_seqlens=cu_npu
    ).cpu()
    torch.npu.synchronize()

    expected = ref_kda_gate_cumsum(g_cpu, cu_seqlens=cu_list)
    torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# Boundary-heavy varlen tests (lengths that straddle chunk boundaries)
# ---------------------------------------------------------------------------


_BOUNDARY_SEQLENS = [
    [1, 63, 64, 65, 127, 128, 129, 447],
    [1, 17, 31, 32, 33, 95, 127, 128, 129, 191, 192, 193, 367],
]


@pytest.mark.parametrize(
    "seqlens",
    _BOUNDARY_SEQLENS,
    ids=[f"boundary_{i}" for i in range(len(_BOUNDARY_SEQLENS))],
)
def test_kda_gate_cumsum_boundary(npu_device, seqlens: list):
    cu_list = _cu_from_seqlens(seqlens)
    T = cu_list[-1]
    N_seq = len(seqlens)

    torch.manual_seed(42)
    g_cpu = -torch.rand(T, GDN_H, GDN_D, dtype=torch.float16)
    cu_npu = torch.tensor(cu_list, dtype=torch.int32).to(npu_device)

    result = pto_kda_gate_cumsum(
        g_cpu.to(npu_device), batch_size=N_seq, seq_len=0, cu_seqlens=cu_npu
    ).cpu()
    torch.npu.synchronize()

    expected = ref_kda_gate_cumsum(g_cpu, cu_seqlens=cu_list)
    torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# Randomly-generated chunk-aligned varlen tests (seed=42)
# ---------------------------------------------------------------------------


_RAND_VARLEN_PARAMS = [(3, 768), (7, 1792), (10, 2560)]
_RAND_VARLEN_CU = [
    _rand_cu_aligned(n, total, GDN_C) for n, total in _RAND_VARLEN_PARAMS
]


@pytest.mark.parametrize(
    "cu_list",
    _RAND_VARLEN_CU,
    ids=[
        f"rand_{n}seq_T{cu[-1]}"
        for (n, _), cu in zip(_RAND_VARLEN_PARAMS, _RAND_VARLEN_CU)
    ],
)
def test_kda_gate_cumsum_random_varlen(npu_device, cu_list: list):
    T = cu_list[-1]
    N_seq = len(cu_list) - 1

    torch.manual_seed(42)
    g_cpu = -torch.rand(T, GDN_H, GDN_D, dtype=torch.float16)
    cu_npu = torch.tensor(cu_list, dtype=torch.int32).to(npu_device)

    result = pto_kda_gate_cumsum(
        g_cpu.to(npu_device), batch_size=N_seq, seq_len=0, cu_seqlens=cu_npu
    ).cpu()
    torch.npu.synchronize()

    expected = ref_kda_gate_cumsum(g_cpu, cu_seqlens=cu_list)
    torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-4)
