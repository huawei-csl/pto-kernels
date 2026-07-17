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


# Dispatch paths in kernel_sinkhorn.cpp::dispatchByK:
#   K=4  — sinkhornSmallBatch    (N <  2048) | sinkhornFastPath    (N >= 2048)
#   K=8  — sinkhornSmallBatch    (N <  1024) | sinkhornFastPath    (N >= 1024)
#   K=16 — sinkhornSmallBatch    (N <   512) | sinkhornFastPath    (N >=  512)
#   K=32 — sinkhornSmallBatch    (N <   256) | sinkhornStridedTree (N >=  256)
#   K=64 — sinkhornSmallBatch    (N <   128) | sinkhornStridedTree (N >=  128)
#   K ∈ (0, 16], K ∉ {4,8,16} — sinkhornStridedTree<TILE_COLS=16>
#   K ∈ (16, 32], K ≠ 32      — sinkhornStridedTree<TILE_COLS=32>
#   K ∈ (32, 64], K ≠ 64      — sinkhornStridedTree<TILE_COLS=64>
#   K ∈ (64, 128]             — sinkhornPerMatrixFp32
#
# Each shape below targets one of those paths; shapes marked "boundary"
# sit on a dispatch threshold where the path flips.
DISPATCH_SHAPES = [
    # --- K=4 smallBatch (N < 2048) ---
    (1, 4),
    (4, 4),
    (32, 4),
    (100, 4),
    (1000, 4),
    (2047, 4),  # boundary: last smallBatch N
    # --- K=4 fastPath (N >= 2048) ---
    (2048, 4),  # boundary: first fastPath N
    (2049, 4),
    (4096, 4),
    (8192, 4),
    # --- K=8 smallBatch (N < 1024) ---
    (1, 8),
    (8, 8),
    (64, 8),
    (500, 8),
    (1023, 8),  # boundary
    # --- K=8 fastPath (N >= 1024) ---
    (1024, 8),  # boundary
    (1025, 8),
    (2048, 8),
    # --- K=16 smallBatch (N < 512) ---
    (1, 16),
    (16, 16),
    (256, 16),
    (511, 16),  # boundary
    # --- K=16 fastPath (N >= 512) ---
    (512, 16),  # boundary
    (513, 16),
    (1024, 16),
    # --- K=32 smallBatch (N < 256) then stridedTree<32> ---
    (1, 32),
    (64, 32),
    (255, 32),  # boundary
    (256, 32),  # boundary → stridedTree
    (512, 32),
    # --- K=64 smallBatch (N < 128) then stridedTree<64> ---
    (1, 64),
    (32, 64),
    (127, 64),  # boundary
    (128, 64),  # boundary → stridedTree
    (256, 64),
    # --- Odd/other K ∈ (0, 16] → stridedTree<16> ---
    (1, 5),
    (8, 7),
    (16, 12),
    (4, 13),
    (32, 15),
    # --- K ∈ (16, 32], K ≠ 32 → stridedTree<32> ---
    (1, 17),
    (64, 20),
    (32, 24),
    (8, 30),
    # --- K ∈ (32, 64], K ≠ 64 → stridedTree<64> ---
    (1, 33),
    (16, 48),
    (8, 50),
    (4, 60),
    # --- K ∈ (64, 128] → fp32 fallback ---
    (1, 65),
    (2, 80),
    (4, 96),
    (2, 100),
    (8, 128),
]
# One (repeat, seed) per shape — cheap, broad dispatch-path coverage.
DISPATCH_CASES = [(batch, K, 10, 0) for (batch, K) in DISPATCH_SHAPES]

# Dense (repeat × seed) coverage for representative shapes — catches
# numerical regressions independent of dispatch path.
DENSE_SHAPES = [
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
DENSE_CASES = [
    (batch, K, repeat, seed)
    for (batch, K) in DENSE_SHAPES
    for repeat in (1, 5, 10)
    for seed in (0, 42)
]

# Dedup (DISPATCH and DENSE overlap on the original small shapes).
TEST_CASES = sorted(set(DISPATCH_CASES + DENSE_CASES))


@pytest.fixture(scope="session")
def sinkhorn_kernel(npu_device):
    return jit_compile(str(KERNEL_CPP), verbose=True, device=npu_device)


@pytest.mark.parametrize("batch,K,repeat,seed", TEST_CASES)
def test_sinkhorn_ds_matches_reference(
    sinkhorn_kernel, npu_device, batch, K, repeat, seed
):
    torch.manual_seed(seed)
    x = torch.randn(batch, K, K, device=npu_device, dtype=DTYPE)
    out = torch.empty_like(x)

    sinkhorn_kernel(x, out, repeat=repeat, eps=1e-6)
    torch.npu.synchronize()

    ref = sinkhorn_ref(x.cpu(), repeat=repeat, eps=1e-6)

    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-5)


# Doubly-stochastic check across one representative shape per dispatch path.
DOUBLY_STOCHASTIC_SHAPES = [
    # smallBatch K ∈ {4,8,16,32,64}
    (4, 4),
    (4, 8),
    (4, 16),
    (4, 32),
    (4, 64),
    # fastPath K ∈ {4,8,16}
    (2048, 4),
    (1024, 8),
    (512, 16),
    # stridedTree (odd / non-{4,8,16,32,64} K)
    (4, 7),
    (4, 20),
    (4, 48),
    # stridedTree at large N for K ∈ {32, 64}
    (256, 32),
    (128, 64),
    # fp32 fallback
    (4, 96),
    (4, 128),
]


@pytest.mark.parametrize("batch,K", DOUBLY_STOCHASTIC_SHAPES)
def test_output_is_doubly_stochastic(sinkhorn_kernel, npu_device, batch, K):
    """After enough iterations, rows and columns should approximately sum to 1/K."""
    torch.manual_seed(123)
    x = torch.randn(batch, K, K, device=npu_device, dtype=DTYPE)
    out = torch.empty_like(x)

    sinkhorn_kernel(x, out, repeat=20, eps=1e-6)
    torch.npu.synchronize()

    out_f = out.float()
    row_sums = out_f.sum(dim=-1)  # (batch, K)
    col_sums = out_f.sum(dim=-2)  # (batch, K)

    # All row sums should be approximately equal
    assert row_sums.std(dim=-1).max() < 0.05, f"Row sums not uniform: {row_sums}"
    # All col sums should be approximately equal
    assert col_sums.std(dim=-1).max() < 0.05, f"Col sums not uniform: {col_sums}"
