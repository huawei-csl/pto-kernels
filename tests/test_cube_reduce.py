# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved. See LICENSE in the root of the repository.

import pytest
import torch

from pto_kernels import pto_cube_reduce


def check_sum(x, *, exact=False):
    dtype = torch.float32 if x.dtype != torch.int8 else torch.int32
    # Use a wider CPU reference for fp32 cancellation and precision checks.
    reference_dtype = torch.float64 if x.dtype == torch.float32 else dtype
    expected = x.cpu().reshape(-1).sum(dtype=reference_dtype).to(dtype).reshape(1)
    actual = pto_cube_reduce(x)
    assert actual.shape == (1,)
    assert actual.dtype == dtype
    assert actual.device == x.device
    tolerance = 0.01 if dtype == torch.float32 and not exact else 0
    torch.testing.assert_close(actual.cpu(), expected, atol=tolerance, rtol=tolerance)
    return actual


def make_input(shape, dtype, device):
    if dtype == torch.int8:
        return torch.randint(-3, 3, shape, dtype=dtype, device=device)
    return (0.1 * torch.randn(shape, device=device)).to(dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.int8])
@pytest.mark.parametrize(
    "shape",
    [
        (),
        (0,),
        (2, 0, 3),
        (0, 7, 11),
        (1,),
        (13,),
        (1000,),
        (1023,),
        (1024,),
        (1025,),
        (4095,),
        (4096,),
        (4097,),
        (16383,),
        (16384,),
        (16385,),
        (327679,),
        (327680,),
        (327681,),
        (1, 7, 1, 11, 1),
        (1000003,),
        (16 * 1024 * 1024 + 7,),
        (7, 11, 13),
        (4, 5, 16384),
    ],
)
def test_cube_reduce(shape, dtype, npu_device):
    check_sum(make_input(shape, dtype, npu_device))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.int8])
@pytest.mark.parametrize("tiles", [21, 39, 41, 61])
@pytest.mark.parametrize("tail", [0, 1, 16383])
def test_cube_reduce_uneven_blocks(dtype, tiles, tail, npu_device):
    # Vary tile values to expose omitted or duplicated block ranges exactly.
    length = tiles * 128 * 128 + tail
    x = (torch.arange(length, dtype=torch.int32) // 16384 % 7 - 3).to(dtype)
    check_sum(x.to(npu_device), exact=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.int8])
def test_cube_reduce_views(dtype, npu_device):
    base = make_input((655362,), dtype, npu_device)
    check_sum(base[1:327681])  # contiguous with a nonzero storage offset
    check_sum(base[1::2])
    check_sum(base[:327680].reshape(20, 16384).t())


@pytest.mark.parametrize("value", [-128, 127])
def test_cube_reduce_int8_extremes(value, npu_device):
    check_sum(torch.full((327681,), value, dtype=torch.int8, device=npu_device))


@pytest.mark.parametrize(
    "dtype",
    [
        torch.bool,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bfloat16,
        torch.float64,
    ],
)
@pytest.mark.parametrize("shape", [(13,), (0,)])
def test_cube_reduce_rejects_dtype(dtype, shape, npu_device):
    with pytest.raises(RuntimeError, match="float16, float32 and int8"):
        pto_cube_reduce(torch.ones(shape, dtype=dtype, device=npu_device))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.int8])
@pytest.mark.parametrize("shape", [(), (0,), (13,)])
def test_cube_reduce_rejects_cpu(dtype, shape):
    with pytest.raises(RuntimeError, match="must be on NPU"):
        pto_cube_reduce(torch.ones(shape, dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.int8])
@pytest.mark.parametrize("length", [1, 1023, 1024, 4095, 4096, 16383, 16384, 16385])
def test_cube_reduce_offset_excludes_surrounding_storage(dtype, length, npu_device):
    # Poison both sides of a view, including where tail padding would be read.
    base = torch.full((length + 65,), 17, dtype=dtype, device=npu_device)
    x = base[1 : length + 1]
    x.fill_(1)
    assert x.is_contiguous() and x.storage_offset() == 1
    check_sum(x, exact=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.int8])
@pytest.mark.parametrize(
    "layout", ["strided", "permuted", "broadcast", "channels_last", "empty"]
)
def test_cube_reduce_noncontiguous_layouts(dtype, layout, npu_device):
    base = make_input((4, 7, 11, 13), dtype, npu_device)
    if layout == "strided":
        x = base[1:, 1::2, ::2, 1::3]
    elif layout == "permuted":
        x = base.permute(2, 0, 3, 1)
    elif layout == "broadcast":
        x = base[0, 0, :1, :].expand(17, 23, 13)
        assert 0 in x.stride()
    elif layout == "channels_last":
        # Construct NHWC storage using operations supported on the NPU.
        x = base.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2)
        assert x.is_contiguous(memory_format=torch.channels_last)
    else:
        x = base[:, :, :0, :].transpose(0, 3)
    if x.numel():
        assert not x.is_contiguous()
    check_sum(x)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.int8])
@pytest.mark.parametrize("length", [1025, 4097, 16385, 327681])
def test_cube_reduce_tile_edge_impulses(dtype, length, npu_device):
    # Exercise row boundaries, tile boundaries, and the final padded element.
    positions = sorted(
        {0, 31, 32, 63, 64, 127, 128, 1023, 1024, 4095, 4096, 16383, 16384, length - 1}
    )
    positions = [index for index in positions if index < length]
    x = torch.zeros(length, dtype=dtype)
    x[positions] = torch.arange(1, len(positions) + 1).to(dtype)
    check_sum(x.to(npu_device), exact=True)


@pytest.mark.parametrize(
    "length,value",
    [
        (327681, 0),
        (327681, 1),
        (327681, -1),
        (327681, 32768),
        (327681, -32768),
        (16385, 65504),
        (16385, -65504),
    ],
)
def test_cube_reduce_fp16_wide_accumulation(length, value, npu_device):
    # Finite inputs may sum beyond fp16's range; accumulation must stay fp32.
    # These lengths/values also keep intermediate fp32 sums exactly representable
    # up to the final rounding, allowing a strict comparison with the CPU.
    x = torch.full((length,), value, dtype=torch.float16, device=npu_device)
    check_sum(x, exact=True)


@pytest.mark.parametrize("length", [1025, 4097, 16385, 327681])
def test_cube_reduce_fp16_cancellation(length, npu_device):
    # All values and intermediate sums are exactly representable in fp32.
    x = torch.empty(length, dtype=torch.float16)
    x[0:-1:2] = 32
    x[1:-1:2] = -32
    x[-1] = 0.5
    check_sum(x.to(npu_device), exact=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.int8])
def test_cube_reduce_repeated_calls_preserve_input_and_outputs(dtype, npu_device):
    x = torch.full((327681,), 1, dtype=dtype, device=npu_device)
    original = x.clone()
    first = check_sum(x, exact=True)
    check_sum(x, exact=True)
    torch.testing.assert_close(x, original, atol=0, rtol=0)
    x.zero_()
    second = check_sum(x, exact=True)
    assert first.data_ptr() != second.data_ptr()
    assert first.cpu().item() == 327681


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.int8])
def test_cube_reduce_uses_current_stream(dtype, npu_device):
    stream = torch.npu.Stream(device=npu_device)
    # Queue the producer, two reductions, and a consumer on the same stream.
    # Avoid host copies between launches so synchronization bugs stay visible.
    with torch.npu.stream(stream):
        x = torch.full((327681,), 1, dtype=dtype, device=npu_device)
        first = pto_cube_reduce(x)
        x.fill_(2)
        second = pto_cube_reduce(x)
        combined = first + second
    stream.synchronize()
    assert first.cpu().item() == 327681
    assert second.cpu().item() == 655362
    assert combined.cpu().item() == 983043


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.int8])
def test_cube_reduce_rejects_length_over_uint32(dtype, npu_device):
    # A zero-stride view reaches the length guard without a huge allocation.
    x = torch.ones(1, dtype=dtype, device=npu_device).expand(2**32)
    with pytest.raises(RuntimeError, match="input length exceeds uint32 range"):
        pto_cube_reduce(x)


@pytest.mark.parametrize("length", [1025, 4097, 16385, 327681])
@pytest.mark.parametrize("value", [1.0000001192092896, 1e-8, 1e10])
def test_cube_reduce_fp32_range_and_precision(length, value, npu_device):
    # One nonzero operand makes the sum exact, exposing reduced input precision
    # and covering values outside fp16's finite range.
    x = torch.zeros(length, dtype=torch.float32)
    x[-1] = value
    check_sum(x.to(npu_device), exact=True)


@pytest.mark.parametrize("length", [1025, 4097, 16385, 327681])
def test_cube_reduce_fp32_cancellation(length, npu_device):
    x = torch.empty(length, dtype=torch.float32)
    x[0:-1:2] = 32
    x[1:-1:2] = -32
    x[-1] = 0.5000000596046448
    check_sum(x.to(npu_device), exact=True)
