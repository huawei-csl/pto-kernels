"""
Correctness tests for the doubly-stochastic Sinkhorn normalization kernel.

Reference: DeepSeek MHC sinkhorn_normalize_ref
  x = x.softmax(-1) + eps
  x = x / (x.sum(-2, keepdim=True) + eps)
  for _ in range(repeat - 1):
      x = x / (x.sum(-1, keepdim=True) + eps)
      x = x / (x.sum(-2, keepdim=True) + eps)
"""

from pathlib import Path

import pytest
import torch
import torch_npu  # noqa

from jit_util_sinkhorn import jit_compile

DTYPE = torch.float16
KERNEL_CPP = Path(__file__).resolve().parent / "kernel_sinkhorn.cpp"


def sinkhorn_ref(x: torch.Tensor, repeat: int = 10, eps: float = 1e-6) -> torch.Tensor:
    """Pure-PyTorch reference (fp32 internal)."""
    x = x.float()
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x.to(torch.float16)


TEST_SHAPES = [
    (1, 4),
    (1, 8),
    (1, 16),
    (1, 32),
    (1, 64),
    (1, 128),
    (4, 4),
    (4, 16),
    (8, 8),
    (16, 16),
    (32, 4),
    (64, 8),
]
TEST_REPEATS = [1, 5, 10]
TEST_SEEDS = [0, 42]
TEST_CASES = [
    (N, K, repeat, seed)
    for N, K in TEST_SHAPES
    for repeat in TEST_REPEATS
    for seed in TEST_SEEDS
]


@pytest.fixture(scope="session")
def sinkhorn_kernel(npu_device):
    return jit_compile(str(KERNEL_CPP), verbose=True, device=npu_device)


@pytest.mark.parametrize("N,K,repeat,seed", TEST_CASES)
def test_sinkhorn_ds_matches_reference(sinkhorn_kernel, npu_device, N, K, repeat, seed):
    torch.manual_seed(seed)
    x = torch.randn(N, K, K, device=npu_device, dtype=DTYPE)
    out = torch.empty_like(x)

    sinkhorn_kernel(x, out, repeat=repeat, eps=1e-6)
    torch.npu.synchronize()

    ref = sinkhorn_ref(x.cpu(), repeat=repeat, eps=1e-6)

    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-5)


def test_output_is_doubly_stochastic(sinkhorn_kernel, npu_device):
    """After enough iterations, rows and columns should approximately sum to 1/K."""
    torch.manual_seed(123)
    K = 8
    x = torch.randn(4, K, K, device=npu_device, dtype=DTYPE)
    out = torch.empty_like(x)

    sinkhorn_kernel(x, out, repeat=20, eps=1e-6)
    torch.npu.synchronize()

    out_f = out.float()
    row_sums = out_f.sum(dim=-1)  # (4, K)
    col_sums = out_f.sum(dim=-2)  # (4, K)

    # All row sums should be approximately equal
    assert row_sums.std(dim=-1).max() < 0.05, f"Row sums not uniform: {row_sums}"
    # All col sums should be approximately equal
    assert col_sums.std(dim=-1).max() < 0.05, f"Col sums not uniform: {col_sums}"
