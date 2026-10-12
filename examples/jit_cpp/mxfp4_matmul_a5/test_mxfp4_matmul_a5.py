"""Correctness for the A5 MXFP4 matmul.

The reference is the operation's own definition, from PTO's CPU implementation
(``pto/cpu/TMatmul.hpp``):

    acc(i, j) = sum_k a(i, k) * b(k, j) * aScale(i, k / 32) * bScale(k / 32, j)

with a and b E2M1 nibbles and the scales E8M0, a biased power of two. Operands
are built from nibble CODES rather than from random floats, deliberately: every
value is then exactly representable, so the only rounding in the whole path is
the bf16 output cast and a mismatch means the kernel is wrong rather than
differently rounded. There is no tolerance to tune.

Two structural gates carry more weight than the error figure, because no layout
mistake survives either: uniform operands under a 2^0 scale must return exactly
K, and a one-hot activation row must recover the matching weight row.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from jit_util_mxfp4_matmul_a5 import (  # noqa
    MX_BLOCK,
    compile_kernel,
    kernel_shape,
    load_matmul,
    tile_plan,
)

# E2M1: sign in bit 3, magnitude in bits 0..2.
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)
E8M0_BIAS = 127
# bf16 carries about three decimal digits, so this is the storage limit of the
# output, not an arithmetic tolerance.
REL_TOLERANCE = 5e-3

# (K, N) pairs. Non-square is included because K and N are independent template
# arguments and a square-only matrix would let a transposed stride pass.
SHAPES = [(512, 512), (1024, 1024), (1024, 512), (512, 1024), (2048, 1024)]


def decode(codes: np.ndarray) -> np.ndarray:
    """Nibble codes -> the exact float values they denote."""
    magnitude = E2M1[codes & 0x7]
    return np.where(codes >= 8, -magnitude, magnitude)


def pack(codes: np.ndarray) -> np.ndarray:
    """Two nibbles per byte, low nibble first."""
    low, high = codes[..., 0::2], codes[..., 1::2]
    return (low | (high << 4)).astype(np.uint8)


def scaled(codes: np.ndarray, exponents: np.ndarray, axis: int) -> np.ndarray:
    """Apply one E8M0 scale per MX_BLOCK elements along ``axis``."""
    factors = np.ldexp(1.0, exponents.astype(np.int32) - E8M0_BIAS)
    return decode(codes) * np.repeat(factors, MX_BLOCK, axis=axis)


def fail_without_vendor(why: str):
    """A comparison with nothing to compare against proves nothing."""
    if os.environ.get("PTO_ALLOW_NO_VENDOR") == "1":
        pytest.skip(f"{why} (PTO_ALLOW_NO_VENDOR=1)")
    pytest.fail(f"{why}: no reference to compare against, so this proves nothing")


def to_device(array: np.ndarray):
    """Host-side generation only.

    Device-side RNG runs on the vector cores, so on a part with one failing
    vector core it raises at the next synchronize and looks exactly like a
    fault in whatever was launched most recently.
    """
    return torch.from_numpy(np.ascontiguousarray(array)).npu()


@pytest.fixture(scope="module", autouse=True)
def device():
    torch.npu.set_device(0)


def build(k: int, n: int):
    so = compile_kernel(k, n, verbose=False)
    return so, load_matmul(so, k, n)


def operands(m: int, k: int, n: int, seed: int, exponent_span=(120, 136)):
    """Random nibble codes plus their exact host product.

    B is ``Layout::DN``: stored (N, K) row-major, that is transposed, and its
    scales likewise. Weights are prepared offline so the transpose is free, but
    feeding row-major B computes something else and looks plausible doing it.
    """
    rng = np.random.default_rng(seed)
    low, high = exponent_span
    a_codes = rng.integers(0, 16, size=(m, k), dtype=np.uint8)
    b_codes = rng.integers(0, 16, size=(k, n), dtype=np.uint8)
    a_exp = rng.integers(low, high, size=(m, k // MX_BLOCK), dtype=np.uint8)
    b_exp = rng.integers(low, high, size=(k // MX_BLOCK, n), dtype=np.uint8)
    want = scaled(a_codes, a_exp, axis=1) @ scaled(b_codes, b_exp, axis=0)
    packed = (
        to_device(pack(a_codes)),
        to_device(a_exp),
        to_device(pack(b_codes.T)),
        to_device(b_exp.T),
    )
    return packed, want


def relative_error(got: np.ndarray, want: np.ndarray) -> float:
    denominator = max(np.abs(want).mean(), 1e-30)
    return float(np.abs(got - want).mean() / denominator)


def run(matmul, packed, m: int):
    out = matmul(*packed, m=m)
    torch.npu.synchronize()
    got = out.float().cpu().numpy().astype(np.float64)
    unwritten = int(np.isnan(got).sum())
    return got, unwritten


@pytest.mark.parametrize("k,n", SHAPES)
def test_matches_exact_reference(k, n):
    so, matmul = build(k, n)
    plan = tile_plan(so)
    m = plan(256)[0]
    packed, want = operands(m, k, n, seed=7)
    got, unwritten = run(matmul, packed, m)
    assert unwritten == 0, f"{unwritten} output elements were never written"
    assert relative_error(got, want) < REL_TOLERANCE


# M values chosen to land on DIFFERENT tile branches, not merely to be spread
# out: the kernel picks a tile from M, so a set of widths that all take one
# branch reports five passes for one code path.
@pytest.mark.parametrize("m", [16, 32, 128, 256, 512, 4096])
def test_matches_exact_reference_across_tile_branches(m):
    k = n = 1024
    so, matmul = build(k, n)
    m_run, _, _ = tile_plan(so)(m)
    packed, want = operands(m_run, k, n, seed=11)
    got, unwritten = run(matmul, packed, m_run)
    assert unwritten == 0, f"{unwritten} output elements were never written"
    assert relative_error(got, want) < REL_TOLERANCE


def test_tile_branches_are_actually_covered():
    """Pin that the parametrization above spans the tile picks it claims to."""
    so, _ = build(1024, 1024)
    plan = tile_plan(so)
    tiles = {plan(m)[1] for m in (16, 32, 128, 256, 512, 4096)}
    # Exactly the three the kernel has -- TINY, SMALL, BIG. Asserting ">= 2"
    # would pass while one whole branch went untested, which is the hole this
    # test exists to close.
    assert len(tiles) == 3, f"tile branches covered: {sorted(tiles)}"


def test_uniform_operands_return_exactly_k():
    """1.0 is exact in E2M1 under a 2^0 scale, so the sum must be exactly K."""
    k = n = 512
    so, matmul = build(k, n)
    m = tile_plan(so)(128)[0]
    ones = np.full((m, k), 2, dtype=np.uint8)  # code 2 == 1.0
    weights = np.full((k, n), 2, dtype=np.uint8)
    unit = np.full((m, k // MX_BLOCK), E8M0_BIAS, dtype=np.uint8)
    weight_unit = np.full((k // MX_BLOCK, n), E8M0_BIAS, dtype=np.uint8)
    packed = (
        to_device(pack(ones)),
        to_device(unit),
        to_device(pack(weights.T)),
        to_device(weight_unit.T),
    )
    got, unwritten = run(matmul, packed, m)
    assert unwritten == 0
    assert np.all(got == float(k)), f"expected exactly {k}, got {np.unique(got)}"


def test_one_hot_activation_recovers_the_weight_row():
    """The sharpest layout probe: a wrong stride returns a different row."""
    k = n = 512
    so, matmul = build(k, n)
    m = tile_plan(so)(128)[0]
    rng = np.random.default_rng(3)
    b_codes = rng.integers(0, 16, size=(k, n), dtype=np.uint8)
    a_codes = np.zeros((m, k), dtype=np.uint8)
    picks = [(row, (row * 37) % k) for row in range(0, m, max(1, m // 24))]
    for row, column in picks:
        a_codes[row, column] = 2  # 1.0
    unit = np.full((m, k // MX_BLOCK), E8M0_BIAS, dtype=np.uint8)
    weight_unit = np.full((k // MX_BLOCK, n), E8M0_BIAS, dtype=np.uint8)
    packed = (
        to_device(pack(a_codes)),
        to_device(unit),
        to_device(pack(b_codes.T)),
        to_device(weight_unit.T),
    )
    got, unwritten = run(matmul, packed, m)
    assert unwritten == 0
    for row, column in picks:
        expected = decode(b_codes[column])
        assert np.array_equal(got[row], expected), f"row {row} took the wrong B row"


def test_matches_the_vendor_on_the_vendors_own_operands():
    """Feed both kernels one quantization and require the same answer.

    ``npu_dynamic_mx_quant`` emits scales shaped (rows, K/64, 2) -- pairs of
    E8M0 bytes -- which is this kernel's (rows, K/32) layout reshaped, so the
    same bytes drive both. That makes this a layout test as much as an
    arithmetic one: if the pairing along K were the other way round, the two
    results would differ.
    """
    if not hasattr(torch_npu, "npu_quant_matmul") or not hasattr(
        torch_npu, "npu_dynamic_mx_quant"
    ):
        fail_without_vendor("torch_npu has no MXFP4 matmul")
    k = n = 1024
    so, matmul = build(k, n)
    m = tile_plan(so)(256)[0]
    rng = np.random.default_rng(29)
    activations = to_device(
        np.asarray(rng.normal(0, 1, (m, k)), dtype=np.float32)
    ).bfloat16()
    weights = to_device(
        np.asarray(rng.normal(0, 0.05, (k, n)), dtype=np.float32)
    ).bfloat16()

    a_packed, a_scale = torch_npu.npu_dynamic_mx_quant(activations)
    b_packed, b_scale = torch_npu.npu_dynamic_mx_quant(weights.t().contiguous())
    fp4, e8m0 = torch.float4_e2m1fn_x2, torch.float8_e8m0fnu
    reference = torch_npu.npu_quant_matmul(
        a_packed.view(fp4),
        b_packed.view(fp4).t(),
        b_scale.view(e8m0).transpose(0, 1),
        pertoken_scale=a_scale.view(e8m0),
        output_dtype=torch.bfloat16,
        group_sizes=[1, 1, 32],
    )
    torch.npu.synchronize()

    ours = matmul(
        a_packed.view(torch.uint8).reshape(m, k // 2),
        a_scale.view(torch.uint8).reshape(m, k // MX_BLOCK),
        b_packed.view(torch.uint8).reshape(n, k // 2),
        b_scale.view(torch.uint8).reshape(n, k // MX_BLOCK),
        m=m,
    )
    torch.npu.synchronize()
    got = ours.float().cpu().numpy().astype(np.float64)
    want = reference.float().cpu().numpy().astype(np.float64)
    assert int(np.isnan(got).sum()) == 0
    assert relative_error(got, want) < REL_TOLERANCE


def test_build_rejects_shapes_the_kernel_would_silently_decline():
    """The launcher answers a shape it was not built for by writing nothing."""
    with pytest.raises(ValueError, match="whole L1 slabs"):
        compile_kernel(256, 512, verbose=False)
    with pytest.raises(ValueError, match="multiple of 64"):
        compile_kernel(512, 96, verbose=False)


def test_load_rejects_a_mismatched_binary():
    so = compile_kernel(512, 512, verbose=False)
    with pytest.raises(ValueError, match="was built for"):
        load_matmul(so, 1024, 1024)


def test_row_count_is_rounded_up_and_the_padding_is_visible():
    """An M off the tile grid is rounded UP, which the caller can see.

    Rounding is the intended behaviour -- the kernel launches on whole tiles --
    so the contract is that the returned buffer has m_round(m) rows, not m. A
    caller who supplies its own output buffer at the unrounded height is told,
    rather than having the extra rows written past the end of it.
    """
    k = n = 512
    so, matmul = build(k, n)
    m_tile = kernel_shape(so)["m_tile"]
    off_grid = m_tile + 1
    m_run = tile_plan(so)(off_grid)[0]
    assert m_run > off_grid and m_run % m_tile == 0

    packed, _ = operands(m_run, k, n, seed=5)
    out = matmul(*packed, m=off_grid)
    torch.npu.synchronize()
    assert tuple(out.shape) == (m_run, n)

    too_short = torch.empty((off_grid, n), dtype=torch.bfloat16, device="npu")
    with pytest.raises(ValueError, match="out must be"):
        matmul(*packed, out=too_short, m=off_grid)


def test_launcher_rejects_a_row_count_it_cannot_serve():
    k = n = 512
    so, matmul = build(k, n)
    m = tile_plan(so)(128)[0]
    packed, _ = operands(m, k, n, seed=5)
    with pytest.raises(ValueError, match="positive"):
        matmul(*packed, m=0)
    with pytest.raises(ValueError, match="at most"):
        matmul(*packed, m=kernel_shape(so)["m_max"] + 1)


def test_output_buffer_shape_is_checked():
    k = n = 512
    so, matmul = build(k, n)
    m = tile_plan(so)(128)[0]
    packed, _ = operands(m, k, n, seed=13)
    wrong = torch.empty((m, n // 2), dtype=torch.bfloat16, device="npu")
    with pytest.raises(ValueError, match="out must be"):
        matmul(*packed, out=wrong, m=m)
