import math
import pathlib
import re

import pytest
import torch

# The wrapper's own tables rather than anything recomputed here: a test that
# re-derived S would silently start probing the wrong boundaries the next time
# the kernel's case table moves.
from cube.jit_util_hadamard_twosided import (
    MAX_TWOSIDED_N,
    MIN_TWOSIDED_N,
    _SAMPLES_PER_TILE,
    _TILES_MID,
    _TILES_TALL,
    hadamard_factors,
    hadamard_split,
    twosided_block_dim_schedule,
    validate_twosided_hadamard_args,
)

# A small spread, run against every N below; the two tests after it straddle
# the tile and tier boundaries, derived from the tables.
TEST_BATCHES = [1, 7, 22, 65]
# Every size in FAST_HADAMARD_TWOSIDED_CASES.
TEST_HIDDEN_DIMS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
TEST_SEEDS = [0, 1]
# Three sizes rather than all eight, because every boundary batch below is a
# device launch: 256 and 1024 carry C == 16 and so the whole-tile reinterpret,
# 4096 the general turnaround. Their tall tiles are 32, 8 and 2 samples, so the
# sweep also crosses the S == 2 edge where the third tier collapses.
TILE_BOUNDARY_HIDDEN_DIMS = [256, 1024, 4096]
# 31/32/33 and 64/65 straddle N=256's tall tile and twice it, 47 leaves the
# last tile masked, and 1-3 are below its shortest tier, where the standalone
# short-span path runs instead of the ring.
TILE_BOUNDARY_BATCHES = [1, 2, 3, 31, 32, 33, 47, 64, 65]
DTYPE = torch.float16

# The kernel rounds to fp16 only twice, so it is compared against an exact
# float64 transform rather than the staged reference. Worst case over the grid
# below is 3.0e-4 relative RMS and 2.4e-3 for the worst element, both relative
# to the output RMS, so these tolerances keep 3-4x of margin.
REL_RMS_TOL = 1e-3
MAX_ELEMENT_TOL = 1e-2


def fwht_f64(x):
    """Exact natural-order (Sylvester) Walsh-Hadamard transform in float64.

    Everything runs on the host: the NPU has no float64 and silently falls
    back to float32 for it.
    """
    a = x.cpu().double().clone()
    b, n = a.shape
    h = 1
    while h < n:
        a = a.reshape(b, n // (2 * h), 2, h)
        u = a[:, :, 0, :].clone()
        v = a[:, :, 1, :].clone()
        a = torch.stack((u + v, u - v), dim=2).reshape(b, n)
        h *= 2
    return a


def hadamard_ref_inplace(x):
    """Reference FHT with the same stage layout as the PTO vector kernel."""
    x = x.clone()
    n = x.shape[-1]
    n_half = n // 2
    log2_n = int(math.log2(n))

    for _ in range(log2_n):
        even = x[..., 0::2].clone()
        odd = x[..., 1::2].clone()
        x[..., :n_half] = even + odd
        x[..., n_half:] = even - odd
    return x


def relative_errors(actual, exact):
    """(relative RMS, worst element) of `actual`, both scaled by the exact RMS."""
    rms = exact.pow(2).mean().sqrt()
    diff = actual - exact
    return (
        (diff.pow(2).mean().sqrt() / rms).item(),
        (diff.abs().max() / rms).item(),
    )


def run_twosided_hadamard(twosided_hadamard_kernel, x):
    y = torch.empty_like(x)
    twosided_hadamard_kernel(x, y)
    torch.npu.synchronize()
    return y


def assert_matches_exact(y, x_cpu, context):
    actual = y.cpu().double()
    assert torch.isfinite(actual).all(), f"non-finite output for {context}"
    if actual.numel() == 0:
        return

    exact = fwht_f64(x_cpu)
    rel_rms, max_element = relative_errors(actual, exact)
    assert (
        rel_rms <= REL_RMS_TOL
    ), f"relative RMS {rel_rms:.3e} exceeds {REL_RMS_TOL:.1e} for {context}"
    assert max_element <= MAX_ELEMENT_TOL, (
        f"worst element {max_element:.3e} exceeds {MAX_ELEMENT_TOL:.1e} "
        f"for {context}"
    )


def check_shape(twosided_hadamard_kernel, npu_device, batch, n, seed=0):
    torch.manual_seed(seed)
    x_cpu = torch.randn(batch, n, dtype=DTYPE)
    y = run_twosided_hadamard(twosided_hadamard_kernel, x_cpu.to(npu_device))
    assert_matches_exact(y, x_cpu, f"seed={seed}, batch={batch}, N={n}")


@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize("batch", TEST_BATCHES)
@pytest.mark.parametrize("n", TEST_HIDDEN_DIMS)
def test_twosided_hadamard_correctness(
    twosided_hadamard_kernel, npu_device, seed, batch, n
):
    check_shape(twosided_hadamard_kernel, npu_device, batch, n, seed=seed)


@pytest.mark.parametrize("n", TILE_BOUNDARY_HIDDEN_DIMS)
def test_sample_tile_boundaries(twosided_hadamard_kernel, npu_device, n):
    """The kernel halves its sample tile on the way down, so straddle the sizes."""
    for batch in TILE_BOUNDARY_BATCHES:
        check_shape(twosided_hadamard_kernel, npu_device, batch, n)


@pytest.mark.parametrize("n", TILE_BOUNDARY_HIDDEN_DIMS)
def test_tile_tier_boundaries(twosided_hadamard_kernel, npu_device, n):
    """Cross both tile-height thresholds at every size that has them, including
    the batch below each, where the shorter tile carries a masked remainder."""
    tall = _SAMPLES_PER_TILE[n]
    mid = max(1, tall // 2)
    edges = {_TILES_TALL * tall, _TILES_MID * mid}
    for edge in edges:
        for batch in (edge - 1, edge, edge + 1):
            if batch > 0:
                check_shape(twosided_hadamard_kernel, npu_device, batch, n)


def test_tile_table_mirrors_the_kernel():
    """The host sizes block_dim for the tile the kernel will run, and a
    disagreement is silent, so compare against the kernel source directly."""
    src = (
        pathlib.Path(__file__).resolve().parent / "fast_hadamard_twosided.cpp"
    ).read_text()

    def define(name):
        m = re.search(rf"^#define {name} (\d+)$", src, re.M)
        assert m, f"{name} is not a plain integer #define any more"
        return int(m.group(1))

    assert define("PTO_TS_TILES_TALL") == _TILES_TALL
    assert define("PTO_TS_TILES_MID") == _TILES_MID

    rows = re.findall(r"^  X\((\d+), (\d+), (\d+), ([A-Za-z0-9_]+)\)", src, re.M)
    assert len(rows) == len(_SAMPLES_PER_TILE), "case table size changed"
    for n, rows_, cols, samples in rows:
        n = int(n)
        s = int(samples) if samples.isdigit() else define(samples)
        assert int(rows_) * int(cols) == n, f"R * C != N for N={n}"
        # The host sizes the OPERAND BUFFERS from hadamard_split, so a split
        # the kernel is not running hands it a wrong-shaped H: a correctness
        # gate, not a schedule gate.
        assert hadamard_split(n) == (int(rows_), int(cols)), (
            f"N={n}: kernel splits as {(int(rows_), int(cols))}, host "
            f"hadamard_split says {hadamard_split(n)}"
        )
        assert _SAMPLES_PER_TILE[n] == s, (
            f"N={n}: kernel tile is {s} samples, host table says "
            f"{_SAMPLES_PER_TILE[n]}"
        )


def test_shared_factor_sizes_have_identical_operand_buffers(npu_device):
    """When R == C, `Layout::kShareHf` has the kernel read ONE GM buffer and
    feed both L0A and L0B from it, so the two host buffers must hold identical
    bytes. Nothing in `_factor_buffer` enforces that: give either side its own
    convention (a transpose, a different NZ image, a sign) and the kernel
    silently multiplies by the wrong matrix on the side it no longer reads.
    """
    src = (
        pathlib.Path(__file__).resolve().parent / "fast_hadamard_twosided.cpp"
    ).read_text()
    assert "kShareHf = (R == C)" in src, (
        "Layout::kShareHf changed shape -- re-derive which sizes share a "
        "buffer before trusting this test"
    )

    rows = re.findall(r"^  X\((\d+), (\d+), (\d+), ([A-Za-z0-9_]+)\)", src, re.M)
    shared = [int(n) for n, r, c, _ in rows if int(r) == int(c)]
    assert shared, "no R == C size left in the case table"

    for n in shared:
        h_left, h_right = hadamard_factors(n, npu_device)
        assert torch.equal(h_left, h_right), (
            f"N={n}: R == C, so the kernel reads h_left for both sides, but "
            "the two operand buffers do not match"
        )


def test_batch_partition_boundaries(twosided_hadamard_kernel, npu_device):
    """The launch splits whole tiles over the cores, so cross the point where
    cores start running out of work (and the point where there is none)."""
    n = 1024
    cores = twosided_hadamard_kernel.block_dim
    batches = sorted({0, 1, cores - 1, cores, cores + 1, 2 * cores, 2 * cores + 1})

    for batch in batches:
        check_shape(twosided_hadamard_kernel, npu_device, batch, n)


@pytest.mark.parametrize("n", [256, 4096])
def test_input_is_not_modified(twosided_hadamard_kernel, npu_device, n):
    # 9 is a whole tile plus a masked remainder at 256, nine whole tiles at 4096.
    torch.manual_seed(0)
    x_cpu = torch.randn(9, n, dtype=DTYPE)
    x = x_cpu.to(npu_device)

    run_twosided_hadamard(twosided_hadamard_kernel, x)

    assert torch.equal(x.cpu(), x_cpu), f"kernel modified its input for N={n}"


@pytest.mark.parametrize("n", [1024, 4096])
def test_more_accurate_than_staged_fp16(twosided_hadamard_kernel, npu_device, n):
    """Rounding twice instead of log2(N) times is the point of the cube path."""
    torch.manual_seed(0)
    x_cpu = torch.randn(16, n, dtype=DTYPE)
    exact = fwht_f64(x_cpu)

    y = run_twosided_hadamard(twosided_hadamard_kernel, x_cpu.to(npu_device))

    twosided_rel_rms, _ = relative_errors(y.cpu().double(), exact)
    staged_rel_rms, _ = relative_errors(hadamard_ref_inplace(x_cpu).double(), exact)
    assert twosided_rel_rms <= staged_rel_rms, (
        f"cube relative RMS {twosided_rel_rms:.3e} is worse than the staged fp16 "
        f"reference {staged_rel_rms:.3e} for N={n}"
    )


# --- argument rejection ---------------------------------------------------


@pytest.mark.parametrize("n", [MIN_TWOSIDED_N // 2, MAX_TWOSIDED_N * 2])
def test_out_of_range_n_rejected(npu_device, n):
    x = torch.randn(2, n, device=npu_device, dtype=DTYPE)
    y = torch.empty_like(x)

    with pytest.raises(ValueError, match="n must be in"):
        validate_twosided_hadamard_args(x, y)


def test_non_power_of_two_n_rejected(npu_device):
    x = torch.randn(2, 192, device=npu_device, dtype=DTYPE)
    y = torch.empty_like(x)

    with pytest.raises(ValueError, match="power of two"):
        validate_twosided_hadamard_args(x, y)


def test_wrong_dtype_rejected(npu_device):
    x = torch.randn(2, 1024, device=npu_device, dtype=torch.float32)
    y = torch.empty_like(x)

    with pytest.raises(TypeError, match="torch.float16"):
        validate_twosided_hadamard_args(x, y)


def test_noncontiguous_input_rejected(npu_device):
    x = torch.randn(1024, 5, device=npu_device, dtype=DTYPE).transpose(0, 1)
    y = torch.empty(5, 1024, device=npu_device, dtype=DTYPE)
    assert x.shape == y.shape
    assert not x.is_contiguous()

    with pytest.raises(ValueError, match="contiguous"):
        validate_twosided_hadamard_args(x, y)


def test_shape_mismatch_rejected(npu_device):
    x = torch.randn(2, 1024, device=npu_device, dtype=DTYPE)
    y = torch.empty(3, 1024, device=npu_device, dtype=DTYPE)

    with pytest.raises(ValueError, match="matching shapes"):
        validate_twosided_hadamard_args(x, y)


# --- block_dim schedule ---------------------------------------------------
#
# block_dim is a per-launch argument and the best value moves with the shape:
# too many cores leaves each one a tile run too short to fill its pipes. It must
# not change the result, which is what these pin down.


def test_block_dim_schedule_in_range():
    for n in TEST_HIDDEN_DIMS:
        for batch in (1, 7, 64, 256, 1024, 8192, 65536):
            bd = twosided_block_dim_schedule(batch, n, 24)
            assert 1 <= bd <= 24, (batch, n, bd)


def test_block_dim_schedule_grows_with_batch():
    """More tiles justify more cores, so the schedule must be monotone."""
    for n in TEST_HIDDEN_DIMS:
        seq = [
            twosided_block_dim_schedule(b, n, 24)
            for b in (1, 64, 256, 1024, 4096, 8192, 65536)
        ]
        assert seq == sorted(seq), (n, seq)


def test_block_dim_schedule_respects_cap():
    assert twosided_block_dim_schedule(1 << 20, 256, 4) == 4
    assert twosided_block_dim_schedule(1 << 20, 256, 1) == 1


@pytest.mark.parametrize("n", [256, 1024, 4096])
def test_block_dim_does_not_change_result(twosided_hadamard_kernel, npu_device, n):
    """The gate that makes the schedule free of correctness risk: every
    block_dim, including the scheduled one, must give bit-identical output.

    It moves the per-core range boundaries, and a range shorter than one tile
    drops that core into runShortSpan with a smaller tile -- so this also tests
    that a sample gets the same answer whatever tile covers it.
    """
    torch.manual_seed(7)
    batch = 300  # 300 % 16 == 12, so the per-core tails differ at 256 and 1024
    x = torch.randn(batch, n, device=npu_device, dtype=DTYPE)

    reference = torch.empty_like(x)
    twosided_hadamard_kernel(x, reference, block_dim=24)
    torch.npu.synchronize()

    scheduled = twosided_block_dim_schedule(batch, n, 24)
    for block_dim in sorted({1, 2, 3, 5, 8, 13, 24, scheduled}):
        y = torch.full_like(x, float("nan"))
        twosided_hadamard_kernel(x, y, block_dim=block_dim)
        torch.npu.synchronize()
        assert torch.equal(y, reference), f"block_dim={block_dim} changed the result"
