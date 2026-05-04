# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------
#
# Tests for pto_chunk_cumsum — chunked prefix-sum of gate values G along the
# time dimension, independently per head h:
#   g_sum[t, h] = Σ_{i=0}^{t} g[i, h]   for t = 0 .. valid-1
#
# Reference test cases adapted from:
# https://github.com/huawei-csl/megagdn-pto/blob/dev/tests/test_single_kernels.py
# (test_cumsum function)

import random

import pytest
import torch

from pto_kernels import pto_chunk_cumsum

# Compile-time constants — must match GDN_H and GDN_C baked into the kernel binary.
GDN_H = 16  # number of attention heads
GDN_C = 128  # chunk size in tokens

random.seed(42)
torch.manual_seed(42)


# ---------------------------------------------------------------------------
# CPU float32 reference
# ---------------------------------------------------------------------------


def ref_chunk_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    cu_seqlens=None,
    batch_size: int = 1,
    seq_len: int = 0,
) -> torch.Tensor:
    """Chunk-local cumulative sum of gate values (CPU float32 reference).

    g : [total_tokens, H] float32
    returns [total_tokens, H] float32 with the same chunk boundaries as the kernel.
    """
    out = torch.zeros_like(g, dtype=torch.float32)
    gf = g.float()
    if cu_seqlens is None:
        for b in range(batch_size):
            bos = b * seq_len
            for j in range(0, seq_len, chunk_size):
                s = bos + j
                e = min(s + chunk_size, bos + seq_len)
                out[s:e] = gf[s:e].cumsum(dim=0)
    else:
        cu = cu_seqlens.tolist() if hasattr(cu_seqlens, "tolist") else list(cu_seqlens)
        for i in range(len(cu) - 1):
            bos, eos = cu[i], cu[i + 1]
            for j in range(0, eos - bos, chunk_size):
                s = bos + j
                e = min(s + chunk_size, eos)
                out[s:e] = gf[s:e].cumsum(dim=0)
    return out


# ---------------------------------------------------------------------------
# Helpers for building cu_seqlens
# ---------------------------------------------------------------------------


def _cu_from_seqlens(seqlens: list[int]) -> list[int]:
    cu = [0]
    for s in seqlens:
        cu.append(cu[-1] + s)
    return cu


def _rand_cu_aligned(n_seq: int, total: int, chunk_size: int) -> list[int]:
    """Random cu_seqlens with all boundaries aligned to chunk_size (seed=42)."""
    rng = random.Random(42)
    if n_seq == 1:
        aligned_total = ((total + chunk_size - 1) // chunk_size) * chunk_size
        return [0, aligned_total]
    raw = sorted(rng.sample(range(1, total), n_seq - 1))
    aligned: list[int] = [0]
    for v in raw:
        v_aligned = ((v + chunk_size - 1) // chunk_size) * chunk_size
        aligned.append(max(v_aligned, aligned[-1] + chunk_size))
    total_aligned = max(total, aligned[-1] + chunk_size)
    total_aligned = ((total_aligned + chunk_size - 1) // chunk_size) * chunk_size
    aligned.append(total_aligned)
    return aligned


# ---------------------------------------------------------------------------
# Fixed-length sequence tests
# (reference: TestCase with cu_seqlens_list=None, T in [128,256,385,512,1024])
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("T", [128, 256, 385, 512, 1024])
@pytest.mark.parametrize("H", [GDN_H])
def test_chunk_cumsum_fixed_length(T: int, H: int):
    """Single batch, fixed sequence length T — exercises the cu_seqlens==nullptr path."""
    torch.manual_seed(42)
    g_cpu = torch.randn(T, H, dtype=torch.float32)
    g_npu = g_cpu.npu()

    result = pto_chunk_cumsum(g_npu, batch_size=1, seq_len=T).cpu()
    torch.npu.synchronize()

    expected = ref_chunk_cumsum(g_cpu, chunk_size=GDN_C, batch_size=1, seq_len=T)
    torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# Variable-length (varlen) sequence tests
# (reference: seqlens lists from build_test_cases())
# ---------------------------------------------------------------------------

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
@pytest.mark.parametrize("H", [GDN_H])
def test_chunk_cumsum_varlen(seqlens: list[int], H: int):
    """Variable-length sequences via cu_seqlens — exercises the non-null cu_seqlens path."""
    cu_list = _cu_from_seqlens(seqlens)
    total_tokens = cu_list[-1]
    N_seq = len(seqlens)

    torch.manual_seed(42)
    g_cpu = torch.randn(total_tokens, H, dtype=torch.float32)
    g_npu = g_cpu.npu()
    cu_npu = torch.tensor(cu_list, dtype=torch.int32).npu()

    result = pto_chunk_cumsum(
        g_npu, batch_size=N_seq, seq_len=0, cu_seqlens=cu_npu
    ).cpu()
    torch.npu.synchronize()

    expected = ref_chunk_cumsum(g_cpu, chunk_size=GDN_C, cu_seqlens=cu_list)
    torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# Boundary-heavy varlen tests
# (reference: seqlens with 1-token seqs and values crossing chunk boundaries)
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
@pytest.mark.parametrize("H", [GDN_H])
def test_chunk_cumsum_boundary(seqlens: list[int], H: int):
    """Sequences with lengths that straddle chunk boundaries and include single-token seqs."""
    cu_list = _cu_from_seqlens(seqlens)
    total_tokens = cu_list[-1]
    N_seq = len(seqlens)

    torch.manual_seed(42)
    g_cpu = torch.randn(total_tokens, H, dtype=torch.float32)
    g_npu = g_cpu.npu()
    cu_npu = torch.tensor(cu_list, dtype=torch.int32).npu()

    result = pto_chunk_cumsum(
        g_npu, batch_size=N_seq, seq_len=0, cu_seqlens=cu_npu
    ).cpu()
    torch.npu.synchronize()

    expected = ref_chunk_cumsum(g_cpu, chunk_size=GDN_C, cu_seqlens=cu_list)
    torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# Randomly-generated chunk-aligned varlen tests (seed=42, mirrors reference)
# ---------------------------------------------------------------------------

_RAND_VARLEN_PARAMS = [
    (3, 768),
    (7, 1792),
    (10, 2560),
]
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
@pytest.mark.parametrize("H", [GDN_H])
def test_chunk_cumsum_random_varlen(cu_list: list[int], H: int):
    """Randomly-generated chunk-aligned varlen shapes (seed=42, 3/7/10 sequences)."""
    total_tokens = cu_list[-1]
    N_seq = len(cu_list) - 1

    torch.manual_seed(42)
    g_cpu = torch.randn(total_tokens, H, dtype=torch.float32)
    g_npu = g_cpu.npu()
    cu_npu = torch.tensor(cu_list, dtype=torch.int32).npu()

    result = pto_chunk_cumsum(
        g_npu, batch_size=N_seq, seq_len=0, cu_seqlens=cu_npu
    ).cpu()
    torch.npu.synchronize()

    expected = ref_chunk_cumsum(g_cpu, chunk_size=GDN_C, cu_seqlens=cu_list)
    torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-4)
