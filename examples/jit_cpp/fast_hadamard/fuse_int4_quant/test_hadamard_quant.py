import math
from pathlib import Path

import pytest
import torch

from fuse_int4_quant.jit_util_hadamard_quant import jit_compile

TEST_BATCHES = [1, 7, 65]
TEST_HIDDEN_DIMS = [128, 1024, 16384]
TEST_SCALES = [0.5, 1.0, 2.0]
TEST_SEEDS = [0, 1]
DTYPE = torch.float16


def hadamard_ref_inplace(x):
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


def quantize_int4_values_ref(x, scale, q_offset=0.0):
    scale_half = torch.tensor(scale, device=x.device, dtype=DTYPE)
    offset_half = torch.tensor(q_offset, device=x.device, dtype=DTYPE)
    q = torch.round((x * scale_half + offset_half).float()).to(torch.int32)
    q = torch.clamp(q, -8, 7)
    return q.to(torch.int8)


def quantize_int4_ref(x, scale, q_offset=0.0):
    q = quantize_int4_values_ref(x, scale, q_offset=q_offset).to(torch.int32)
    low = torch.bitwise_and(q[:, 0::2], 0xF)
    high = torch.bitwise_and(q[:, 1::2], 0xF)
    packed = torch.bitwise_or(low, torch.bitwise_left_shift(high, 4))
    packed = packed.to(torch.int16)
    packed = torch.where(packed >= 128, packed - 256, packed)
    return packed.to(torch.int8)


def unpack_int4_packed(packed):
    if packed.dtype != torch.int8:
        raise TypeError("packed must use torch.int8 packed-byte storage.")
    if packed.dim() != 2:
        raise ValueError("packed must be a 2D tensor.")

    packed_u8 = torch.bitwise_and(packed.to(torch.int16), 0xFF)
    low = torch.bitwise_and(packed_u8, 0xF)
    high = torch.bitwise_and(torch.bitwise_right_shift(packed_u8, 4), 0xF)

    low = torch.where(low >= 8, low - 16, low).to(torch.int8)
    high = torch.where(high >= 8, high - 16, high).to(torch.int8)

    unpacked = torch.empty(
        packed.shape[0],
        packed.shape[1] * 2,
        device=packed.device,
        dtype=torch.int8,
    )
    unpacked[:, 0::2] = low
    unpacked[:, 1::2] = high
    return unpacked


def fused_ref(x, scale, q_offset=0.0):
    return quantize_int4_ref(hadamard_ref_inplace(x), scale, q_offset=q_offset)


def run_fused(fused_kernel, x, y, scale, log2_n=None, **kwargs):
    n = x.shape[-1]
    if log2_n is None:
        log2_n = int(math.log2(n))
    fused_kernel(x, y, x.shape[0], n, log2_n, scale, **kwargs)
    torch.npu.synchronize()


@pytest.fixture(scope="session")
def hadamard_quant_kernel(npu_device):
    base = Path(__file__).resolve().parent
    src = base / "fast_hadamard_quant.cpp"
    return jit_compile(str(src), verbose=True, device=npu_device)


@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize("scale", TEST_SCALES)
@pytest.mark.parametrize("batch", TEST_BATCHES)
@pytest.mark.parametrize("n", TEST_HIDDEN_DIMS)
def test_fast_hadamard_quant_correctness(
    hadamard_quant_kernel, npu_device, seed, scale, batch, n
):
    torch.manual_seed(seed)
    x = torch.randn(batch, n, device=npu_device, dtype=DTYPE)
    y = torch.empty(batch, n // 2, device=npu_device, dtype=torch.int8)

    y_ref = fused_ref(x, scale)

    run_fused(hadamard_quant_kernel, x, y, scale)

    assert torch.equal(
        y, y_ref
    ), f"Mismatch for seed={seed}, batch={batch}, N={n}, scale={scale}"


def test_fast_hadamard_quant_with_offset(hadamard_quant_kernel, npu_device):
    x = torch.randn(7, 1024, device=npu_device, dtype=DTYPE)
    y = torch.empty(7, 512, device=npu_device, dtype=torch.int8)
    y_ref = fused_ref(x, 1.25, q_offset=1.5)

    run_fused(hadamard_quant_kernel, x, y, 1.25, q_offset=1.5)

    assert torch.equal(y, y_ref)


def test_fast_hadamard_quant_grouped_scales(hadamard_quant_kernel, npu_device):
    batch = 5
    n = 1024
    group_size = 128
    groups = n // group_size
    x = torch.randn(batch, n, device=npu_device, dtype=DTYPE)
    y = torch.empty(batch, n // 2, device=npu_device, dtype=torch.int8)
    q_scales = torch.linspace(
        0.5, 1.5, groups, device=npu_device, dtype=DTYPE
    ).contiguous()

    x_ref = hadamard_ref_inplace(x)
    y_parts = []
    for g in range(groups):
        beg = g * group_size
        end = beg + group_size
        y_parts.append(quantize_int4_ref(x_ref[:, beg:end], float(q_scales[g].item())))
    y_ref = torch.cat(y_parts, dim=1)

    run_fused(
        hadamard_quant_kernel,
        x,
        y,
        1.0,
        group_size=group_size,
        q_scales=q_scales,
    )

    assert torch.equal(y, y_ref)


def test_fast_hadamard_quant_grouped_scale_and_offsets(
    hadamard_quant_kernel, npu_device
):
    batch = 3
    n = 1024
    group_size = 256
    groups = n // group_size
    x = torch.randn(batch, n, device=npu_device, dtype=DTYPE)
    y = torch.empty(batch, n // 2, device=npu_device, dtype=torch.int8)
    q_scales = torch.linspace(
        0.75, 1.25, groups, device=npu_device, dtype=DTYPE
    ).contiguous()
    q_offsets = torch.linspace(
        -1.0, 1.0, groups, device=npu_device, dtype=DTYPE
    ).contiguous()

    x_ref = hadamard_ref_inplace(x)
    y_parts = []
    for g in range(groups):
        beg = g * group_size
        end = beg + group_size
        y_parts.append(
            quantize_int4_ref(
                x_ref[:, beg:end],
                float(q_scales[g].item()),
                q_offset=float(q_offsets[g].item()),
            )
        )
    y_ref = torch.cat(y_parts, dim=1)

    run_fused(
        hadamard_quant_kernel,
        x,
        y,
        1.0,
        group_size=group_size,
        q_scales=q_scales,
        q_offsets=q_offsets,
    )

    assert torch.equal(y, y_ref)


def test_fast_hadamard_quant_packs_signed_int4_nibbles(
    hadamard_quant_kernel, npu_device
):
    torch.manual_seed(0)
    scale = 1.25
    q_offset = 1.5
    x = torch.randn(2, 128, device=npu_device, dtype=DTYPE)
    y = torch.empty(2, 64, device=npu_device, dtype=torch.int8)

    run_fused(hadamard_quant_kernel, x, y, scale, q_offset=q_offset)

    expected_int4 = quantize_int4_values_ref(
        hadamard_ref_inplace(x),
        scale,
        q_offset=q_offset,
    )

    assert y.dtype == torch.int8
    assert y.shape == (x.shape[0], x.shape[1] // 2)
    assert torch.equal(unpack_int4_packed(y), expected_int4)


def test_fast_hadamard_quant_does_not_mutate_input(hadamard_quant_kernel, npu_device):
    x = torch.randn(7, 1024, device=npu_device, dtype=DTYPE)
    x_before = x.clone()
    y = torch.empty(7, 512, device=npu_device, dtype=torch.int8)

    run_fused(hadamard_quant_kernel, x, y, 1.0)

    assert torch.equal(x, x_before)


@pytest.mark.parametrize("n", [1, 3, 257, 16385])
def test_fast_hadamard_quant_rejects_unsupported_n(
    hadamard_quant_kernel, npu_device, n
):
    x = torch.randn(2, n, device=npu_device, dtype=DTYPE)
    y = torch.empty(2, max(1, n // 2), device=npu_device, dtype=torch.int8)

    with pytest.raises(ValueError, match="n must"):
        hadamard_quant_kernel(x, y, scale=1.0)
