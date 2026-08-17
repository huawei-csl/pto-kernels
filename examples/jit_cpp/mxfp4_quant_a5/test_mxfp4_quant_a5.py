# pylint: disable=wrong-import-position  # imports are guarded by importorskip
"""Correctness for mxfp4_quant_a5, bit-exact against torch_npu.npu_dynamic_mx_quant.

Runs repeat (PTO_DEVICE_REPEATS, default 5, floored at 1).
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")

from jit_util_mxfp4_a5 import (  # noqa: E402
    K,
    TILE_GRAIN,
    TILE_ELEMS,
    MX_BLOCK,
    SUPPORTED_K,
    build_and_load,
    compile_kernel,
    kernel_rows_for,
    load_quantizer,
    row_quantum,
    rows_for,
)

_REFERENCE = (
    Path(__file__).resolve().parents[3] / ".skills/testing-pto-kernels/reference"
)
if _REFERENCE.is_dir():
    sys.path.insert(0, str(_REFERENCE))
    import pto_demo_utils as demo  # noqa: E402
else:  # pragma: no cover - only on a partial checkout
    demo = None

VENDOR_DST_TYPE = 296  # torch_npu.float4_e2m1fn_x2
# E2M1 magnitude grid by 3-bit field, used only by the quality report
E2M1_GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)


def repeats() -> int:
    """Floored at 1: a repeat count of 0 would make every comparison loop below
    iterate zero times, so 25 tests would pass having asserted nothing."""
    return max(1, int(demo.device_repeats() if demo else 5))


def synchronize() -> None:
    (demo.synchronize_device if demo else torch.npu.synchronize)()


def fail_without_vendor(why: str):
    """Fail, not skip, when the reference is missing.

    PTO_ALLOW_NO_VENDOR=1 downgrades to a skip, for a machine that lacks the
    operator.
    """
    if os.environ.get("PTO_ALLOW_NO_VENDOR") == "1":
        pytest.skip(f"{why} (PTO_ALLOW_NO_VENDOR=1)")
    pytest.fail(f"{why}: no reference to compare against, so this proves nothing")


def vendor_quantize(tensor):
    """(nibbles, scales) from the CANN operator, reshaped to match ours."""
    fn = getattr(torch_npu, "npu_dynamic_mx_quant", None)
    if fn is None:
        fail_without_vendor("torch_npu.npu_dynamic_mx_quant is missing")
    try:
        nibbles, scales = fn(tensor, dst_type=VENDOR_DST_TYPE)
    except Exception as exc:  # pragma: no cover - op signature drift
        fail_without_vendor(f"vendor op rejected the call: {type(exc).__name__}: {exc}")
    synchronize()
    batch, k = tensor.shape
    blocks = k // MX_BLOCK
    # The vendor lays its scales out as (batch, k/64, 2), so a k that is an ODD
    # multiple of 32 -- 96 is the shipped one -- has no whole number of pairs and
    # it emits a padded ceil(blocks/2)*2 columns. Ours is the tight spec layout,
    # so compare against the leading `blocks` and check the padding is only that.
    wide = scales.cpu().numpy().reshape(batch, -1)
    assert wide.shape[1] in (blocks, -(-blocks // 2) * 2), (
        f"unexpected vendor scale width {wide.shape[1]} for k={k} "
        f"(blocks={blocks}); the vendor scale layout changed"
    )
    return nibbles.cpu().numpy().reshape(batch, k // 2), wide[:, :blocks]


def make_bf16(batch, k, seed):
    """Random bf16, rounded once on the host so both sides see the same values."""
    gen = torch.Generator().manual_seed(seed)
    values = torch.randn(batch, k, generator=gen, dtype=torch.float32)
    return values.to(torch.bfloat16).npu()


def run_and_compare(kernel, tensor, label):
    want_nibbles, want_scales = vendor_quantize(tensor)
    for attempt in range(repeats()):
        nibbles, scales = kernel(tensor)
        synchronize()
        got_nibbles, got_scales = nibbles.cpu().numpy(), scales.cpu().numpy()
        for what, got, want in (
            ("scale", got_scales, want_scales),
            ("nibble", got_nibbles, want_nibbles),
        ):
            assert np.array_equal(got, want), (
                f"{label}: {what} differs from the vendor on attempt {attempt} "
                f"({int((got != want).sum())} of {want.size})"
            )


@pytest.fixture(scope="module")
def default_quantizer():
    return build_and_load(verbose=False)


# 1000 and 4097 do not fill a whole number of tiles, so they exercise the kernel's
# partial-tile tail; 65536 is more logical work than physical cores. None of
# these reaches the host padding path -- row_quantum is 1 at this K. See
# test_host_padding_path for that.
@pytest.mark.parametrize("batch", [1, 7, 33, 64, 128, 1000, 4097, 12345, 65536])
def test_matches_vendor(default_quantizer, batch):
    run_and_compare(default_quantizer, make_bf16(batch, K, batch), f"batch={batch}")


@pytest.mark.parametrize("k", [k for k in SUPPORTED_K if row_quantum(k) > 1])
def test_host_padding_path(k):
    """A batch the wrapper must round up: the alloc/copy/synchronize path, which
    nothing else here reaches and where ordering against torch's copy matters."""
    quantum = row_quantum(k)
    assert quantum > 1, f"k={k} never pads; this test needs a narrower width"
    batch = 2 * rows_for(k) + quantum - 1
    assert batch % quantum, "a multiple would skip the padding path entirely"
    kernel = build_and_load(k=k, verbose=False)
    run_and_compare(kernel, make_bf16(batch, k, batch), f"k={k} padded")


@pytest.mark.parametrize("k", SUPPORTED_K)
def test_matches_vendor_at_row_width(k):
    kernel = build_and_load(k=k, verbose=False)
    run_and_compare(kernel, make_bf16(4 * rows_for(k), k, k), f"k={k}")


@pytest.mark.parametrize("k", [k for k in SUPPORTED_K if rows_for(k) > 1])
def test_partial_last_tile(k):
    """A batch that does NOT fill its last tile, so the kernel's tail runs.

    A multiple of row_quantum, so this is the kernel's tail, not the host's
    zero-fill.
    """
    rows, quantum = rows_for(k), row_quantum(k)
    # smallest multiple of quantum past three whole tiles; quantum does not
    # divide rows at every width (k=96 has rows=128, quantum=10), so nudge on
    batch = ((3 * rows) // quantum + 1) * quantum
    if batch % rows == 0:
        batch += quantum
    assert batch % rows, f"k={k}: batch {batch} fills whole tiles, no tail"
    assert batch % quantum == 0, f"k={k}: would pad, hiding the kernel tail"
    kernel = build_and_load(k=k, verbose=False)
    run_and_compare(kernel, make_bf16(batch, k, batch), f"k={k} tail")


def test_stream_pointer_follows_the_active_stream():
    """The launch must go to the CURRENT stream, not the first one seen."""
    from jit_util_mxfp4_a5 import current_stream_ptr

    default = current_stream_ptr().value
    side = torch.npu.Stream()
    with torch.npu.stream(side):
        assert current_stream_ptr().value != default, "stale cached stream"
    assert current_stream_ptr().value == default


def test_tiling_invariants():
    """Every derived shape the wrapper and kernel both depend on, in one place.

    row_quantum is a conservative bound: the 32-byte scale-row rule binds the
    tile type's compile-time Cols, not the runtime valid extent, so this does not
    assert that property of it.
    """
    for k in SUPPORTED_K:
        tile, quantum = rows_for(k) * k, row_quantum(k)
        assert tile % TILE_GRAIN == 0, f"k={k}: tile {tile} is not a whole grain"
        assert (tile // MX_BLOCK) % 32 == 0, f"k={k}: scale row is not legal DMA"
        assert tile <= TILE_ELEMS, f"k={k}: tile {tile} exceeds TILE_ELEMS"
        assert 1 <= quantum <= rows_for(k), f"k={k}: bad quantum {quantum}"
    assert rows_for(768) == 20, "768 must back off from the naive 21"
    assert rows_for(96) == 160 and rows_for(3584) == 4
    assert row_quantum(64) == 16 and row_quantum(4096) == 1


def test_nibble_order_is_pinned(default_quantizer):
    """One block of known codes, asserted exactly. No auto-fitting."""
    tensor = torch.zeros((rows_for(K), K), dtype=torch.bfloat16)
    tensor[0, 0], tensor[0, 1], tensor[0, 31] = 1.0, 2.0, 6.0
    nibbles, scales = default_quantizer(tensor.npu())
    synchronize()
    assert int(scales.cpu().numpy()[0, 0]) == 127, "amax=6.0 must give scale byte 127"
    assert int(nibbles.cpu().numpy()[0, 0]) == 0x42, "element 0 must be the low nibble"


ADVERSARIAL = {
    "clamp_window": [2.0**-15, 2.0**-14, 2.0**-13],
    "subnormal_amax": [2.0**-20, 2.0**-24],
    "clip_to_six_band": [6.5, 7.0, 7.9],
    "e2m1_midpoints": [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
    "all_zero": [0.0],
    "huge_outlier": [1024.0],
    "near_bf16_max": [3.0e38],
    "signed_zero": [-0.0],
}


@pytest.mark.parametrize("name", sorted(ADVERSARIAL))
def test_adversarial_blocks(default_quantizer, name):
    """Values random N(0,1) never reaches."""
    family = ADVERSARIAL[name]
    tensor = torch.zeros((rows_for(K), K), dtype=torch.bfloat16)
    for index, value in enumerate(family):
        block_start = (index % (K // MX_BLOCK)) * MX_BLOCK
        tensor[0, block_start : block_start + MX_BLOCK] = value
        if len(family) > 1:
            tensor[0, block_start + 1] = value / 2.0
    run_and_compare(default_quantizer, tensor.npu(), name)


def test_output_is_nontrivial(default_quantizer):
    """Two different inputs must give two different outputs."""
    outs = []
    for seed in (11, 12):
        nibbles, scales = default_quantizer(make_bf16(rows_for(K), K, seed))
        synchronize()
        outs.append((nibbles.cpu().numpy().copy(), scales.cpu().numpy().copy()))
    for what, a, b in (
        ("q", outs[0][0], outs[1][0]),
        ("scale", outs[0][1], outs[1][1]),
    ):
        assert not np.array_equal(a, b), f"{what} same for both inputs: did not run"


def test_quantization_quality(default_quantizer):
    """Relative RMSE and R-squared, not max error alone."""
    tensor = make_bf16(1024, K, 13)
    nibbles, scales = default_quantizer(tensor)
    synchronize()
    packed, rows, nblk = nibbles.cpu().numpy(), tensor.shape[0], K // MX_BLOCK
    codes = np.empty((rows, K), dtype=np.uint8)
    codes[:, 0::2], codes[:, 1::2] = packed & 0x0F, packed >> 4
    mag = E2M1_GRID[codes & 0x07]
    signed = np.where((codes & 0x08) != 0, -mag, mag)
    scale = np.exp2(scales.cpu().numpy().astype(np.float64) - 127.0)
    recon = (signed.reshape((rows, nblk, MX_BLOCK)) * scale[:, :, None]).reshape(
        rows, K
    )
    original = tensor.float().cpu().numpy().astype(np.float64)
    mse = float(np.mean((recon - original) ** 2))
    rmse_rel = mse**0.5 / (float(np.sqrt(np.mean(original**2))) or 1.0)
    r2 = 1.0 - mse / float(np.var(original))
    print(f"\n  MXFP4 quality: rmse/rms={rmse_rel:.4f}  R^2={r2:.4f}")
    # MXFP4 keeps 3 magnitude bits over a 32-element block, so ~0.1 on N(0,1).
    # These bounds catch a broken kernel, not a subtle one.
    assert rmse_rel < 0.25, f"relative RMSE {rmse_rel:.4f} too high for MXFP4"
    assert r2 > 0.9, f"R^2 {r2:.4f} too low for MXFP4"


# 32 and 160 are multiples of MX_BLOCK with no instantiation; 4864 is a real model
# width no row count can tile at this TILE_ELEMS; 16384 exceeds it outright.
@pytest.mark.parametrize("bad_k", [0, 32, 160, 4864, 16384])
def test_unsupported_row_width_is_rejected(bad_k):
    with pytest.raises(ValueError):
        build_and_load(k=bad_k, verbose=False)
    with pytest.raises(ValueError):
        load_quantizer(compile_kernel(verbose=False), k=bad_k)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_wrong_dtype_is_rejected(default_quantizer, dtype):
    tensor = torch.zeros((rows_for(K), K), dtype=dtype).npu()
    with pytest.raises(AssertionError, match="bfloat16"):
        default_quantizer(tensor)


def test_non_contiguous_is_rejected(default_quantizer):
    wide = torch.zeros((rows_for(K), 2 * K), dtype=torch.bfloat16).npu()
    view = wide[:, :K]
    assert not view.is_contiguous(), "test needs a genuinely strided view"
    with pytest.raises(AssertionError, match="contiguous"):
        default_quantizer(view)
    default_quantizer(view.contiguous())  # same data, accepted


def test_rows_for_matches_kernel():
    """The host rows_for() must agree with the kernel's RowsFor<K>."""
    query = kernel_rows_for(compile_kernel(verbose=False))
    mismatched = {
        k: (rows_for(k), query(k)) for k in SUPPORTED_K if rows_for(k) != query(k)
    }
    assert not mismatched, f"host/kernel tiling disagree: {mismatched}"
