import pytest
import torch

# The wrapper's own tables rather than TILE_ELEMS // N: S is a per-N override
# in the kernel (256 and 512 run SMALLER tiles than the capacity formula
# gives), so a test that recomputed it would probe the wrong boundaries the
# next time the kernel's case table moves.
from cube.jit_util_hadamard_onesided import (
    MAX_ONESIDED_N,
    MIN_ONESIDED_BATCH,
    MIN_ONESIDED_N,
    _SAMPLES_PER_TILE,
    _SAMPLES_PER_TILE_SHALLOW,
    clear_operand_cache,
    hadamard_full,
    hadamard_matrix,
    hadamard_operands,
    onesided_block_dim_schedule,
    validate_onesided_hadamard_args,
)

# The one-sided product costs n*n MACs per sample however H is factored, so its
# case table stops at 512 on compute grounds; larger belongs to cube/.
TEST_HIDDEN_DIMS = [64, 128, 256, 512]
# Straddles both boundaries that matter: the S-sample tile (256/128/64/64
# across the grid) and the 16-sample tail the kernel mops up with a masked
# tile.
TEST_BATCHES = [1, 3, 15, 16, 17, 31, 32, 33, 48, 63, 64, 65, 127, 128, 255, 300, 1000]
TEST_SEEDS = [0, 1]
SHORT_BATCHES = list(range(1, MIN_ONESIDED_BATCH))
MAX_BLOCK_DIM = 24
DTYPE = torch.float16

# Exactly representable in fp16 and far outside randn's range, so a row the
# kernel or the padding path must not touch is recognisable byte for byte.
SENTINEL = -7.0

# The one-sided kernel accumulates in fp32 and rounds to fp16 exactly ONCE, so
# it reproduces no staged fp16 reference bit for bit and is compared against an
# exact float64 transform. Worst case over the grid below is 2.43e-4 relative
# RMS (at batch=1, N=64, where a single sample makes the RMS noisy) and 1.88e-3
# for the worst element, both relative to the output RMS; these tolerances keep
# 4-5x of margin and match the two-sided suite's.
REL_RMS_TOL = 1e-3
MAX_ELEMENT_TOL = 1e-2

# test_more_accurate_than_two_sided compares two measured error figures, so it
# allows a few percent of slack rather than asserting a strict ordering on
# numbers that move with the input.
ACCURACY_SLACK = 1.05


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


def tile_sizes(n: int) -> tuple[int, ...]:
    """Every S this N can run, as the wrapper mirrors the kernel's dispatch.

    Two for the resident sizes (the pick depends on batch), one otherwise. The
    boundary tests straddle all of them, so they stay aimed at a real tile edge
    whichever side of the crossover the batch falls.

    Not `.get(n, 1)` like the schedule's own lookup: a size missing from the
    table would quietly turn every boundary below into batch=1, so fail loudly
    instead.
    """
    assert n in _SAMPLES_PER_TILE, (
        f"N={n} is in the test grid but not in the kernel's tile table "
        f"{sorted(_SAMPLES_PER_TILE)}"
    )
    return tuple(sorted({_SAMPLES_PER_TILE[n], _SAMPLES_PER_TILE_SHALLOW[n]}))


def relative_errors(actual, exact):
    """(relative RMS, worst element) of `actual`, both scaled by the exact RMS."""
    rms = exact.pow(2).mean().sqrt()
    diff = actual - exact
    return (
        (diff.pow(2).mean().sqrt() / rms).item(),
        (diff.abs().max() / rms).item(),
    )


def run_onesided_hadamard(onesided_hadamard_kernel, x, **kwargs):
    y = torch.full_like(x, SENTINEL)
    onesided_hadamard_kernel(x, y, **kwargs)
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


def check_shape(onesided_hadamard_kernel, npu_device, batch, n, seed=0):
    torch.manual_seed(seed)
    x_cpu = torch.randn(batch, n, dtype=DTYPE)
    y = run_onesided_hadamard(onesided_hadamard_kernel, x_cpu.to(npu_device))
    assert_matches_exact(y, x_cpu, f"seed={seed}, batch={batch}, N={n}")


@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize("batch", TEST_BATCHES)
@pytest.mark.parametrize("n", TEST_HIDDEN_DIMS)
def test_onesided_hadamard_correctness(
    onesided_hadamard_kernel, npu_device, seed, batch, n
):
    check_shape(onesided_hadamard_kernel, npu_device, batch, n, seed=seed)


@pytest.mark.parametrize("n", TEST_HIDDEN_DIMS)
def test_sample_tile_boundaries(onesided_hadamard_kernel, npu_device, n):
    """The kernel halves its sample tile on the way down to 16, so straddle
    every level it can stop at for this N."""
    batches = sorted(
        {
            b
            for s in tile_sizes(n)
            for step in (s, s // 2, s // 4, 16)
            for b in (step - 1, step, step + 1, 2 * step, 2 * step + 1)
            if b >= 1
        }
    )
    for batch in batches:
        check_shape(onesided_hadamard_kernel, npu_device, batch, n)


@pytest.mark.parametrize("n", TEST_HIDDEN_DIMS)
def test_unaligned_tail(onesided_hadamard_kernel, npu_device, n):
    """A core whose span is not a whole number of tiles covers the remainder
    with a SHORT last ring tile: a full-size tile whose load and store carry a
    row count of 1..S-1. Nothing outside the span may be read or written, and
    the rows past the row count must not leak into the output.

    The batches below straddle a tile boundary rather than merely a 16-row one,
    so the short tile really is the last of several and has a predecessor to
    overlap; a span shorter than one tile takes the standalone path instead."""
    batches = sorted(
        {
            b
            for s in tile_sizes(n)
            for b in (17, 23, 31, 33, 47, s - 1, s + 1, s + 15, 2 * s + 3, 999)
            if b >= MIN_ONESIDED_BATCH and b % 16 != 0
        }
    )
    assert batches, f"no tail batches for N={n}"
    for batch in batches:
        check_shape(onesided_hadamard_kernel, npu_device, batch, n)


@pytest.mark.parametrize("n", TEST_HIDDEN_DIMS)
def test_batch_partition_boundaries(onesided_hadamard_kernel, npu_device, n):
    """The launch splits whole tiles over the cores, so cross the point where
    cores start running out of work (and the point where there is none)."""
    cores = onesided_hadamard_kernel.block_dim
    batches = sorted(
        {0, 1}
        | {
            b
            for s in tile_sizes(n)
            for b in (s, s * (cores - 1), s * cores, s * cores + 1, s * cores + 17)
        }
    )
    for batch in batches:
        check_shape(onesided_hadamard_kernel, npu_device, batch, n)


@pytest.mark.parametrize("n", TEST_HIDDEN_DIMS)
def test_short_batch(onesided_hadamard_kernel, npu_device, n):
    """batch 1..15 leaves the tile loop with no whole tile to run, so the rows
    go straight to the smallest masked tile -- the same path any batch's last
    1..15 rows take. The result must be right, and the rows past `batch` in the
    caller's buffer must be left alone: that containment is the whole reason
    the store is narrowed rather than the input padded."""
    rows = MIN_ONESIDED_BATCH + 4
    torch.manual_seed(11)
    x_cpu = torch.randn(rows, n, dtype=DTYPE)
    x = x_cpu.to(npu_device)

    for batch in SHORT_BATCHES:
        y = torch.full((rows, n), SENTINEL, device=npu_device, dtype=DTYPE)
        x_in, y_out = x[:batch], y[:batch]
        assert x_in.is_contiguous() and y_out.is_contiguous()

        onesided_hadamard_kernel(x_in, y_out)
        torch.npu.synchronize()

        assert_matches_exact(y_out, x_cpu[:batch], f"padded batch={batch}, N={n}")
        untouched = y[batch:].cpu()
        assert torch.equal(
            untouched, torch.full_like(untouched, SENTINEL)
        ), f"padding path wrote past batch={batch} for N={n}"


@pytest.mark.parametrize("n", [64, 512])
def test_short_batch_stale_l1_tail(onesided_hadamard_kernel, npu_device, n):
    """A masked tile loads only `rows` rows into the L1 X slot, so the rest of
    the slot still holds whatever the previous call left there and the cube
    folds it into output rows that are never stored. Walk the batch down so
    every call inherits a longer stale tail than it writes."""
    for batch in (15, 1, 9, 2, 14, 3):
        check_shape(onesided_hadamard_kernel, npu_device, batch, n, seed=batch)


@pytest.mark.parametrize("n", [64, 512])
def test_input_is_not_modified(onesided_hadamard_kernel, npu_device, n):
    # 9 is all partial tile; 100 is whole tiles plus a 4-row partial tile.
    for batch in (9, 100):
        torch.manual_seed(0)
        x_cpu = torch.randn(batch, n, dtype=DTYPE)
        x = x_cpu.to(npu_device)

        run_onesided_hadamard(onesided_hadamard_kernel, x)

        assert torch.equal(
            x.cpu(), x_cpu
        ), f"kernel modified its input for batch={batch}, N={n}"


@pytest.mark.parametrize("n", [256, 512])
def test_more_accurate_than_two_sided(request, onesided_hadamard_kernel, npu_device, n):
    """Rounding once instead of twice is the point of the one-sided path.

    The two-sided kernel needs N = R * C with both factors 16-aligned, so this
    only has a comparison arm at N >= 256.
    """
    try:
        twosided_hadamard_kernel = request.getfixturevalue("twosided_hadamard_kernel")
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"two-sided kernel unavailable: {exc}")

    torch.manual_seed(0)
    x_cpu = torch.randn(128, n, dtype=DTYPE)
    x = x_cpu.to(npu_device)
    exact = fwht_f64(x_cpu)

    onesided_rel_rms, _ = relative_errors(
        run_onesided_hadamard(onesided_hadamard_kernel, x).cpu().double(), exact
    )

    twosided_y = torch.empty_like(x)
    twosided_hadamard_kernel(x, twosided_y)
    torch.npu.synchronize()
    twosided_rel_rms, _ = relative_errors(twosided_y.cpu().double(), exact)

    assert onesided_rel_rms <= twosided_rel_rms * ACCURACY_SLACK, (
        f"one-sided relative RMS {onesided_rel_rms:.3e} is worse than the "
        f"two-sided kernel's {twosided_rel_rms:.3e} for N={n}"
    )


def test_hadamard_full_is_cached_and_orthogonal(npu_device):
    """H is handed to the kernel as the right operand with no transpose, which
    only gives the natural-order transform because H is symmetric."""
    h = hadamard_full(256, torch.device(npu_device))
    assert hadamard_full(256, torch.device(npu_device)) is h, "H is not cached"
    assert h.dtype == DTYPE and h.is_contiguous()

    h64 = hadamard_matrix(64, "cpu", dtype=torch.float64)
    assert torch.equal(h64, h64.T), "H must be symmetric"
    assert torch.equal(h64 @ h64, 64 * torch.eye(64, dtype=torch.float64))


@pytest.mark.parametrize("n", TEST_HIDDEN_DIMS)
def test_operand_buffer_layout(npu_device, n):
    """The kernel reads H_n at offset 0 and +-H_M at offset n*n.

    Those offsets are hard-coded in loadHadamard, so this is the host half of
    that contract. A change here does not fail loudly, it feeds the matmul a
    wrong-sized operand.
    """
    m, b = hadamard_operands(n)
    assert m * b == n and m == min(64, n), f"unexpected split M={m} B={b}"

    h = hadamard_full(n, torch.device(npu_device)).cpu()
    assert h.numel() == n * n + 2 * m * m, "operand buffer is the wrong size"

    h_n = hadamard_matrix(n, "cpu")
    h_m = hadamard_matrix(m, "cpu")
    assert torch.equal(h[: n * n].view(n, n), h_n), "H_n is not at offset 0"
    assert torch.equal(
        h[n * n : n * n + m * m].view(m, m), h_m
    ), "+H_M is not at offset n*n"
    assert torch.equal(h[n * n + m * m :].view(m, m), -h_m), "-H_M does not follow +H_M"


@pytest.mark.parametrize("n", TEST_HIDDEN_DIMS)
def test_kronecker_block_identity(n):
    """The identity the blocked form computes, checked exactly in fp64.

        Y[:, jM:(j+1)M] = sum_i X[:, iM:(i+1)M] @ (H_B[i,j] * H_M)

    and H_B[i,j] == (-1)**popcount(i & j), which is why the kernel never builds
    or reads H_B at all.
    """
    m, b = hadamard_operands(n)
    h_m = hadamard_matrix(m, "cpu", dtype=torch.float64)
    h_b = hadamard_matrix(b, "cpu", dtype=torch.float64)

    for i in range(b):
        for j in range(b):
            assert h_b[i, j] == (-1) ** bin(i & j).count(
                "1"
            ), f"sign rule wrong at ({i},{j}) for B={b}"

    # Small integers, so every product and partial sum is exact in fp64 and the
    # identity can be asserted bitwise. With randn it holds only to a few ULP,
    # because the two sides accumulate in different orders -- which would make
    # the test about rounding rather than about the decomposition.
    torch.manual_seed(0)
    x = torch.randint(-8, 9, (5, n)).to(torch.float64)
    want = x @ hadamard_matrix(n, "cpu", dtype=torch.float64)

    got = torch.zeros_like(want)
    for j in range(b):
        for i in range(b):
            got[:, j * m : (j + 1) * m] += x[:, i * m : (i + 1) * m] @ (h_b[i, j] * h_m)
    assert torch.equal(got, want), "block-Kronecker decomposition is not exact"


# --- argument rejection ---------------------------------------------------


@pytest.mark.parametrize("n", [MIN_ONESIDED_N // 2, MAX_ONESIDED_N * 2])
def test_out_of_range_n_rejected(npu_device, n):
    x = torch.randn(32, n, device=npu_device, dtype=DTYPE)
    y = torch.empty_like(x)

    with pytest.raises(ValueError, match="n must be in"):
        validate_onesided_hadamard_args(x, y)


def test_non_power_of_two_n_rejected(npu_device):
    x = torch.randn(32, 192, device=npu_device, dtype=DTYPE)
    y = torch.empty_like(x)

    with pytest.raises(ValueError, match="power of two"):
        validate_onesided_hadamard_args(x, y)


def test_wrong_dtype_rejected(npu_device):
    x = torch.randn(32, 256, device=npu_device, dtype=torch.float32)
    y = torch.empty_like(x)

    with pytest.raises(TypeError, match="torch.float16"):
        validate_onesided_hadamard_args(x, y)


def test_noncontiguous_input_rejected(npu_device):
    x = torch.randn(256, 32, device=npu_device, dtype=DTYPE).transpose(0, 1)
    y = torch.empty(32, 256, device=npu_device, dtype=DTYPE)
    assert x.shape == y.shape
    assert not x.is_contiguous()

    with pytest.raises(ValueError, match="contiguous"):
        validate_onesided_hadamard_args(x, y)


def test_shape_mismatch_rejected(npu_device):
    x = torch.randn(32, 256, device=npu_device, dtype=DTYPE)
    y = torch.empty(33, 256, device=npu_device, dtype=DTYPE)

    with pytest.raises(ValueError, match="matching shapes"):
        validate_onesided_hadamard_args(x, y)


# --- block_dim schedule ---------------------------------------------------
#
# block_dim is a per-launch argument and the best value moves with the shape:
# too many cores leaves each one a tile run too short to fill its pipes. It must
# not change the result, which is what these pin down.


def test_block_dim_schedule_in_range():
    for n in TEST_HIDDEN_DIMS:
        for batch in (1, 7, 64, 256, 1024, 8192, 65536):
            bd = onesided_block_dim_schedule(batch, n, MAX_BLOCK_DIM)
            assert 1 <= bd <= MAX_BLOCK_DIM, (batch, n, bd)


def test_block_dim_schedule_grows_with_batch():
    """More tiles justify more cores, so the schedule must be monotone."""
    for n in TEST_HIDDEN_DIMS:
        seq = [
            onesided_block_dim_schedule(b, n, MAX_BLOCK_DIM)
            for b in (1, 64, 256, 1024, 4096, 8192, 65536)
        ]
        assert seq == sorted(seq), (n, seq)


def test_block_dim_schedule_respects_cap():
    assert onesided_block_dim_schedule(1 << 20, 256, 4) == 4
    assert onesided_block_dim_schedule(1 << 20, 256, 1) == 1


@pytest.mark.parametrize("n", TEST_HIDDEN_DIMS)
def test_block_dim_does_not_change_result(onesided_hadamard_kernel, npu_device, n):
    """The gate that makes the schedule free of correctness risk: every
    block_dim, including the scheduled one, must give bit-identical output.

    block_dim moves both the per-core range boundaries and which tile size each
    sample lands in, and the backed-up tail makes neighbouring ranges overlap,
    so this is the test that the overlap really is byte-identical.
    """
    torch.manual_seed(7)
    batch = 300  # 300 % 16 == 12, so no tile size divides it on any core
    assert batch % 16 != 0 and all(batch % s != 0 for s in tile_sizes(n))
    x = torch.randn(batch, n, device=npu_device, dtype=DTYPE)

    reference = torch.empty_like(x)
    onesided_hadamard_kernel(x, reference, block_dim=MAX_BLOCK_DIM)
    torch.npu.synchronize()

    scheduled = onesided_block_dim_schedule(batch, n, MAX_BLOCK_DIM)
    for block_dim in sorted({1, 2, 3, 5, 8, 13, MAX_BLOCK_DIM, scheduled}):
        y = torch.full_like(x, float("nan"))
        onesided_hadamard_kernel(x, y, block_dim=block_dim)
        torch.npu.synchronize()
        assert torch.equal(y, reference), f"block_dim={block_dim} changed the result"


def test_first_call_for_a_size_is_ordered_after_h_is_built(
    onesided_hadamard_kernel, npu_device
):
    """The very first call for a given n must not race the H buffer.

    hadamard_full enqueues torch ops on a cache miss, and the raw ctypes launch
    is only ordered after them because resolve_stream_ptr drains torch_npu's
    task queue first. Evaluated in the wrong order, the kernel reads H as zeros
    and returns Y = X * 0, on the first call for each n only.

    Clearing the cache is what makes this a regression test rather than a
    repeat of the correctness sweep: without it every n is already warm.
    """
    for n in TEST_HIDDEN_DIMS:
        clear_operand_cache()
        torch.manual_seed(0)
        x_cpu = torch.randn(1, n, dtype=DTYPE)
        y = run_onesided_hadamard(onesided_hadamard_kernel, x_cpu.to(npu_device))
        assert y.abs().sum().item() != 0.0, (
            f"N={n}: output is all zeros on the first call, so the kernel read "
            "H before it was written"
        )
        assert_matches_exact(y, x_cpu, f"first call, batch=1, N={n}")
