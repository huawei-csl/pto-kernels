# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------

import pytest
import torch

from pto_kernels import pto_histogram
from pto_kernels import pto_histogram_op  # noqa: F401 (used in error tests)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ref_histogram(
    x_cpu: torch.Tensor, bins: int, range_min: float, range_max: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute histogram on CPU using torch.histogram as ground truth."""
    return torch.histogram(
        x_cpu.float(),
        bins=bins,
        range=(range_min, range_max),
    )


# ---------------------------------------------------------------------------
# Basic correctness tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_tiles", [1, 4, 16, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=str)
def test_pto_histogram_uniform(num_tiles: int, dtype: torch.dtype):
    """Values drawn from a uniform distribution; verify counts sum to numel."""
    tile_len = 64
    n = num_tiles * tile_len
    torch.manual_seed(42)
    x = torch.rand(n, dtype=dtype)
    range_min, range_max = 0.0, 1.0

    x_npu = x.npu()
    result = pto_histogram(x_npu, bins=10, range=(range_min, range_max))

    assert result.hist.shape == (10,)
    assert result.bin_edges.shape == (11,)
    # Total counts must equal numel
    assert int(result.hist.sum().cpu()) == n


@pytest.mark.parametrize("n_bins", [8, 16, 64, 100, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=str)
def test_pto_histogram_correctness(n_bins: int, dtype: torch.dtype):
    """Counts must match torch.histogram (CPU ground truth)."""
    tile_len = 64
    n = 8 * tile_len  # 512 elements
    torch.manual_seed(0)
    x = torch.randn(n, dtype=dtype)

    range_min = float(x.min().item()) - 0.01
    range_max = float(x.max().item()) + 0.01

    # NPU result
    x_npu = x.npu()
    result = pto_histogram(x_npu, bins=n_bins, range=(range_min, range_max))
    hist_npu = result.hist.cpu().float()

    # CPU ground truth
    ref_hist, ref_edges = _ref_histogram(x, n_bins, range_min, range_max)

    assert torch.equal(hist_npu, ref_hist), (
        f"Histogram mismatch for n_bins={n_bins}, dtype={dtype}:\n"
        f"  NPU  : {hist_npu}\n"
        f"  CPU  : {ref_hist}"
    )
    assert torch.allclose(result.bin_edges.cpu(), ref_edges, atol=1e-5)


# ---------------------------------------------------------------------------
# API compatibility tests (mirrors torch.histogram interface)
# ---------------------------------------------------------------------------


def test_pto_histogram_default_range():
    """When range is None, the range is derived from the data."""
    tile_len = 64
    x = torch.linspace(-1.0, 1.0, steps=tile_len, dtype=torch.float32)
    x_npu = x.npu()
    result = pto_histogram(x_npu, bins=10)
    assert result.hist.shape == (10,)
    assert int(result.hist.sum().cpu()) == tile_len


def test_pto_histogram_bins_tensor():
    """bins can be a 1-D Tensor of monotonically increasing bin edges."""
    tile_len = 64
    x = torch.rand(tile_len, dtype=torch.float32)
    bin_edges = torch.linspace(0.0, 1.0, steps=11)  # 10 bins

    x_npu = x.npu()
    result = pto_histogram(x_npu, bins=bin_edges)

    assert result.hist.shape == (10,)
    assert result.bin_edges.shape == (11,)
    assert int(result.hist.sum().cpu()) == tile_len


def test_pto_histogram_density():
    """density=True: integral of histogram over range should equal 1."""
    tile_len = 64
    torch.manual_seed(7)
    x = torch.rand(tile_len, dtype=torch.float32)
    x_npu = x.npu()

    result = pto_histogram(x_npu, bins=10, range=(0.0, 1.0), density=True)

    # Integral = sum(hist * bin_width)
    bin_widths = result.bin_edges[1:] - result.bin_edges[:-1]
    integral = float((result.hist.cpu() * bin_widths.cpu()).sum())
    assert abs(integral - 1.0) < 1e-4, f"density integral {integral} != 1.0"


def test_pto_histogram_named_tuple_fields():
    """Return value is a named tuple with .hist and .bin_edges attributes."""
    x = torch.rand(64, dtype=torch.float32).npu()
    result = pto_histogram(x, bins=8, range=(0.0, 1.0))

    # Both attribute access and index unpack should work
    hist, bin_edges = result
    assert torch.equal(hist, result.hist)
    assert torch.equal(bin_edges, result.bin_edges)


def test_pto_histogram_non_multiple_of_tile():
    """Inputs whose length is not a multiple of 64 are handled transparently."""
    # 100 elements – not a multiple of 64
    x = torch.rand(100, dtype=torch.float32)
    x_npu = x.npu()
    result = pto_histogram(x_npu, bins=10, range=(0.0, 1.0))
    # All 100 real elements should be counted
    assert int(result.hist.sum().cpu()) == 100


def test_pto_histogram_multidimensional_input():
    """Multi-dimensional input is flattened automatically."""
    x = torch.rand(4, 64, dtype=torch.float32)  # shape [4, 64] = 256 elements
    x_npu = x.npu()
    result = pto_histogram(x_npu, bins=10, range=(0.0, 1.0))
    assert int(result.hist.sum().cpu()) == x.numel()


# ---------------------------------------------------------------------------
# Error-handling tests
# ---------------------------------------------------------------------------


def test_pto_histogram_weight_raises():
    """Passing weight= raises NotImplementedError."""
    x = torch.rand(64, dtype=torch.float32).npu()
    w = torch.ones(64, dtype=torch.float32).npu()
    with pytest.raises(NotImplementedError, match="weight"):
        pto_histogram(x, bins=10, range=(0.0, 1.0), weight=w)


def test_pto_histogram_invalid_range():
    """range_min >= range_max must raise RuntimeError."""
    x = torch.rand(64, dtype=torch.float32).npu()
    with pytest.raises(RuntimeError):
        pto_histogram_op(x, 10, 1.0, 0.0)


def test_pto_histogram_invalid_bins():
    """bins outside [1, 1024] must raise RuntimeError."""
    x = torch.rand(64, dtype=torch.float32).npu()
    with pytest.raises(RuntimeError):
        pto_histogram_op(x, 0, 0.0, 1.0)
    with pytest.raises(RuntimeError):
        pto_histogram_op(x, 2000, 0.0, 1.0)
