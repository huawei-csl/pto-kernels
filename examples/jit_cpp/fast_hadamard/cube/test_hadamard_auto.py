"""Correctness and routing for the size-dispatching entry point.

The required grid is `bench_common.BENCH_HIDDEN_DIMS` = [128 .. 16384], which is
what `standard/test_hadamard.py` also covers; this suite runs that grid plus the
two sizes only one back end reaches (64, one-sided only; 32768, two-sided only).
"""

import pytest
import torch

from cube.jit_util_hadamard_auto import (
    MAX_AUTO_N,
    MIN_AUTO_N,
    ONESIDED_BATCH_MIN,
    ONESIDED_MAX_N,
    TWOSIDED_MIN_N,
    pick_implementation,
    supported_auto_sizes,
)
from bench_common import BENCH_HIDDEN_DIMS

# The grid every other suite in this example is held to.
REQUIRED_HIDDEN_DIMS = list(BENCH_HIDDEN_DIMS)
# Plus the ends of the dispatcher's own range, which the vector kernel cannot
# reach: 64 is below its grid and 32768 above its hard UB-tile limit.
EXTRA_HIDDEN_DIMS = [64, 32768]
ALL_HIDDEN_DIMS = sorted(set(REQUIRED_HIDDEN_DIMS + EXTRA_HIDDEN_DIMS))

TEST_BATCHES = [1, 7, 22, 65]
DTYPE = torch.float16

# Same basis as cube/test_hadamard_twosided.py: both back ends accumulate in fp32
# and round to fp16 only a couple of times, so neither reproduces a staged-fp16
# reference bit for bit and both are compared against an exact fp64 transform.
REL_RMS_TOL = 1e-3
MAX_ELEMENT_TOL = 1e-2


def fwht_f64(x):
    """Exact natural-order (Sylvester) transform, on the host in float64.

    The NPU has no float64 and silently falls back to float32.
    """
    a = x.cpu().double().clone()
    b, n = a.shape
    h = 1
    while h < n:
        a = a.reshape(b, n // (2 * h), 2, h)
        u = a[:, :, 0, :].clone()
        v = a[:, :, 1, :].clone()
        a[:, :, 0, :] = u + v
        a[:, :, 1, :] = u - v
        a = a.reshape(b, n)
        h *= 2
    return a


def _assert_matches_reference(y, ref):
    got = y.cpu().double()
    assert torch.isfinite(got).all()
    scale = (ref**2).mean().sqrt()
    rel_rms = ((got - ref) ** 2).mean().sqrt() / scale
    max_err = (got - ref).abs().max() / ref.abs().max()
    assert rel_rms < REL_RMS_TOL, f"rel_rms {rel_rms:.3e}"
    assert max_err < MAX_ELEMENT_TOL, f"max {max_err:.3e}"


# --- routing -----------------------------------------------------------------


@pytest.mark.parametrize("n", ALL_HIDDEN_DIMS)
def test_routing_is_the_measured_crossover(n):
    expected = "onesided" if n < TWOSIDED_MIN_N else "twosided"
    assert pick_implementation(n) == expected


@pytest.mark.parametrize(
    "batch,expected",
    [
        (1, "twosided"),
        (16, "twosided"),
        (1024, "twosided"),
        (ONESIDED_BATCH_MIN[256] - 1, "twosided"),
        (ONESIDED_BATCH_MIN[256], "onesided"),
        (4 * ONESIDED_BATCH_MIN[256], "onesided"),
    ],
)
def test_n256_routing_follows_batch(batch, expected):
    """N=256 is the one size where batch moves the choice: both kernels hit the
    memory roofline there, and past the floor the one-sided store pattern is
    the better of the two by 1.03-1.04x. Below it the split wins by up to
    1.29x."""
    assert pick_implementation(256, batch) == expected


@pytest.mark.parametrize("n", [64, 128])
def test_batch_does_not_move_routing_below_the_split(n):
    """Below N=256 there is no R * C with both factors 16-aligned, so the
    one-sided kernel is the only one that runs at any batch."""
    for batch in (1, 16, 1024, 16384, 1 << 20):
        assert pick_implementation(n, batch) == "onesided"


@pytest.mark.parametrize("n", [512, 1024, 16384, 32768])
def test_batch_does_not_move_routing_above_256(n):
    """Everywhere above N=256 the split wins at every batch, and from 1024 up
    it is the only kernel that runs at all."""
    for batch in (1, 16, 1024, 16384, 1 << 20):
        assert pick_implementation(n, batch) == "twosided"


def test_onesided_floors_are_inside_the_overlap():
    """A floor at an N only one kernel reaches would be unreachable, and at an
    N the one-sided kernel refuses it would route to a ValueError."""
    for n in ONESIDED_BATCH_MIN:
        assert TWOSIDED_MIN_N <= n <= ONESIDED_MAX_N


def test_required_grid_is_inside_the_supported_range():
    """The dispatcher must cover every size the example's own grid asks for."""
    assert min(REQUIRED_HIDDEN_DIMS) >= MIN_AUTO_N
    assert max(REQUIRED_HIDDEN_DIMS) <= MAX_AUTO_N
    for n in REQUIRED_HIDDEN_DIMS:
        assert pick_implementation(n) in ("onesided", "twosided")


def test_supported_sizes_are_the_union_of_the_two_back_ends():
    assert supported_auto_sizes() == [
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
    ]


@pytest.mark.parametrize("n", [32, MIN_AUTO_N // 2, MAX_AUTO_N * 2])
def test_out_of_range_n_rejected(n):
    with pytest.raises(ValueError, match="n must be in"):
        pick_implementation(n)


@pytest.mark.parametrize("n", [96, 1000, 3000])
def test_non_power_of_two_rejected(n):
    with pytest.raises(ValueError, match="power of two"):
        pick_implementation(n)


# --- correctness -------------------------------------------------------------


@pytest.mark.parametrize("n", ALL_HIDDEN_DIMS)
@pytest.mark.parametrize("batch", TEST_BATCHES)
def test_matches_exact_transform(auto_hadamard_kernel, npu_device, batch, n):
    torch.manual_seed(batch * 1000 + n)
    x = torch.randn(batch, n, device=npu_device, dtype=DTYPE)
    y = torch.empty_like(x)
    auto_hadamard_kernel(x, y)
    torch.npu.synchronize()
    _assert_matches_reference(y, fwht_f64(x))


@pytest.mark.parametrize("n", ALL_HIDDEN_DIMS)
def test_dispatch_is_bit_identical_to_calling_the_back_end_directly(
    auto_hadamard_kernel,
    twosided_hadamard_kernel,
    onesided_hadamard_kernel,
    npu_device,
    n,
):
    """The dispatcher must add nothing but the choice.

    Same kernel, same arguments, so the outputs have to agree exactly -- not
    within a tolerance.
    """
    torch.manual_seed(n)
    x = torch.randn(64, n, device=npu_device, dtype=DTYPE)
    y_auto = torch.empty_like(x)
    y_direct = torch.empty_like(x)

    auto_hadamard_kernel(x, y_auto)
    backend = (
        onesided_hadamard_kernel
        if pick_implementation(n) == "onesided"
        else twosided_hadamard_kernel
    )
    backend(x, y_direct)
    torch.npu.synchronize()
    assert torch.equal(y_auto, y_direct)


def test_input_is_not_modified(auto_hadamard_kernel, npu_device):
    """Out of place, unlike the in-place vector kernel it replaces."""
    torch.manual_seed(0)
    x = torch.randn(32, 1024, device=npu_device, dtype=DTYPE)
    before = x.clone()
    auto_hadamard_kernel(x, torch.empty_like(x))
    torch.npu.synchronize()
    assert torch.equal(x, before)


def test_1d_input_rejected(auto_hadamard_kernel, npu_device):
    x = torch.randn(1024, device=npu_device, dtype=DTYPE)
    with pytest.raises(ValueError, match="2D"):
        auto_hadamard_kernel(x, torch.empty_like(x))


def test_only_the_needed_back_end_is_compiled(auto_hadamard_kernel):
    """Compilation is on demand, so the set of built kernels tracks use."""
    assert set(auto_hadamard_kernel.compiled) <= {"onesided", "twosided"}
    auto_hadamard_kernel.kernel_for("onesided")
    auto_hadamard_kernel.kernel_for("twosided")
    assert set(auto_hadamard_kernel.compiled) == {"onesided", "twosided"}
