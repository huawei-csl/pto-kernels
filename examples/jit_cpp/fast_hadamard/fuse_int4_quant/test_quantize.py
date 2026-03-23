from pathlib import Path

import pytest
import torch

from fuse_int4_quant.jit_util_quantize import jit_compile

TEST_BATCHES = [1, 7, 65]
TEST_HIDDEN_DIMS = [128, 1024, 16384]
TEST_SCALES = [0.5, 1.0, 2.0]
TEST_SEEDS = [0, 1]
DTYPE = torch.float16


def quantize_int4_values_ref(x, scale):
    scale_half = torch.tensor(scale, device=x.device, dtype=DTYPE)
    q = torch.round((x * scale_half).float()).to(torch.int32)
    q = torch.clamp(q, -8, 7)
    return q.to(torch.int8)


def quantize_int4_ref(x, scale):
    q = quantize_int4_values_ref(x, scale).to(torch.int32)
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


def assert_quantize_matches_ref(quantize_kernel, x, scale):
    y = torch.empty(x.shape[0], x.shape[1] // 2, device=x.device, dtype=torch.int8)
    y_ref = quantize_int4_ref(x, scale)

    quantize_kernel(x, y, scale)
    torch.npu.synchronize()

    assert torch.equal(y, y_ref)


@pytest.fixture(scope="session")
def quantize_kernel(npu_device):
    base = Path(__file__).resolve().parent
    src = base / "quantize.cpp"
    return jit_compile(str(src), verbose=True, device=npu_device)


@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize("scale", TEST_SCALES)
@pytest.mark.parametrize("batch", TEST_BATCHES)
@pytest.mark.parametrize("n", TEST_HIDDEN_DIMS)
def test_quantize_correctness(quantize_kernel, npu_device, seed, scale, batch, n):
    torch.manual_seed(seed)
    x = torch.randn(batch, n, device=npu_device, dtype=DTYPE)
    assert_quantize_matches_ref(quantize_kernel, x, scale)


@pytest.mark.parametrize("scale", [0.0, -0.5, -1.0, -2.0])
def test_quantize_zero_and_negative_scales(quantize_kernel, npu_device, scale):
    x = torch.tensor(
        [[-12.0, -3.5, -1.0, 0.0, 1.0, 3.5, 12.0, 15.5]],
        device=npu_device,
        dtype=DTYPE,
    )
    assert_quantize_matches_ref(quantize_kernel, x, scale)


def test_quantize_saturates(quantize_kernel, npu_device):
    x = torch.tensor(
        [[-300.0, -9.2, -7.4, -0.6, 0.4, 6.8, 7.4, 300.0]],
        device=npu_device,
        dtype=DTYPE,
    )
    assert_quantize_matches_ref(quantize_kernel, x, 1.0)


def test_quantize_packs_signed_int4_nibbles(quantize_kernel, npu_device):
    torch.manual_seed(0)
    scale = 1.25
    x = torch.randn(2, 128, device=npu_device, dtype=DTYPE)
    y = torch.empty(2, 64, device=npu_device, dtype=torch.int8)

    quantize_kernel(x, y, scale)
    torch.npu.synchronize()

    expected_int4 = quantize_int4_values_ref(x, scale)

    assert y.dtype == torch.int8
    assert y.shape == (x.shape[0], x.shape[1] // 2)
    assert torch.equal(unpack_int4_packed(y), expected_int4)


def test_quantize_rejects_odd_last_dim(quantize_kernel, npu_device):
    x = torch.randn(2, 7, device=npu_device, dtype=DTYPE)
    y = torch.empty(2, 3, device=npu_device, dtype=torch.int8)

    with pytest.raises(ValueError, match="positive even"):
        quantize_kernel(x, y, 1.0)
