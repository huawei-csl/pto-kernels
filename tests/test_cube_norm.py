# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved. See LICENSE in the root of the repository.

import numpy as np
import pytest
import torch

from pto_kernels import pto_cube_norm


# Largest int8 input run_cube_norm accepts: every entry of the shared int32
# workspace tile sums numel / size products of at most 128 * 128, and past
# that the accumulator wraps. Mirrors the host's bound at size 64.
INT8_MAX_NUMEL = 64 * ((2**31 - 1) // (128 * 128))


# Tile size pick of run_cube_norm: the largest tile the input fills. int8 has
# no 16 x 16 L1 tile, so it stops at 32 and leaves shorter inputs to the tail.
def tile_size(numel, dtype=np.float16):
    if numel >= 128 * 128:
        return 64
    if dtype == np.int8:
        return 32
    return 32 if numel >= 64 * 64 else 16


def numpy_cube_norm(x, blocks=24):
    """NumPy model of the kernel, tiling and fp32 accumulation included.

    1. The flat input of `n` elements is cut into `n // (S*S)` contiguous S x S
       tiles A_0, A_1, ... This is the same thing as viewing x as the
       (n/S) x S matrix A and taking its S x S row blocks; the Frobenius norm
       is invariant under transposition, so reading the tiles in the
       DMA-friendly contiguous order computes the norm of the S x (n/S) matrix
       of the task statement just as well.
    2. Each block of the grid owns a contiguous range of tiles and accumulates
       F = sum_i A_i^T @ A_i in its L0C accumulator, in fp32.
    3. The Vector stage gathers and sums the diagonals of every block's F.
       Because trace(A^T A) is the sum of the squares of A's entries, that is
       the squared norm of everything the Cube stage saw.
    4. Elements past the last whole tile (fewer than S*S of them) never reach
       the Cube unit; the Vector stage squares and sums them directly.
    5. sqrt of the total is the result.
    """
    flat = np.ascontiguousarray(x).reshape(-1)
    size = tile_size(flat.size, flat.dtype)
    # int8 operands multiply into an int32 accumulator, which the host keeps
    # the input short enough to leave exact; int64 here models that exactly.
    accumulate_in = np.int64 if flat.dtype == np.int8 else np.float32
    num_tiles = flat.size // (size * size)
    # The grid is sized so that every launched block gets at least one tile.
    blocks = max(1, min(num_tiles, blocks))

    # --- Cube stage: one S x S accumulator per block. ---
    squared_norm = np.float32(0)
    for block in range(blocks):
        begin = block * (num_tiles // blocks) + min(block, num_tiles % blocks)
        count = num_tiles // blocks + (block < num_tiles % blocks)
        accumulator = np.zeros((size, size), dtype=accumulate_in)
        for index in range(begin, begin + count):
            tile = flat[index * size * size : (index + 1) * size * size]
            # fp16 and int8 operands both widen into the accumulator on chip.
            tile = tile.reshape(size, size).astype(accumulate_in)
            accumulator += tile.T @ tile
        # --- Vector stage: the diagonal of each block's result. ---
        squared_norm += np.trace(accumulator).astype(np.float32)

    # --- Vector stage: the ragged tail, squared and summed as it is. ---
    tail = flat[num_tiles * size * size :].astype(np.float32)
    squared_norm += (tail * tail).sum(dtype=np.float32)
    return np.sqrt(squared_norm)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize(
    "numel", [0, 1, 1023, 1024, 4097, 16384, 16385, 327681, 1000003]
)
def test_numpy_model_matches_numpy_norm(numel, dtype):
    # The model is only a reference if it is a norm in the first place; this
    # pins it to np.linalg.norm before the kernel is compared against it.
    rng = np.random.default_rng(0)
    x = (0.1 * rng.standard_normal(numel)).astype(dtype)
    expected = np.linalg.norm(x.astype(np.float64))
    assert numpy_cube_norm(x) == pytest.approx(expected, rel=1e-3, abs=1e-6)


def check_norm(x, *, tolerance=1e-3):
    # A float64 CPU reference: the device sums squares in a different order,
    # and fp16 inputs are exact in fp32, so only the ordering shows up here.
    expected = x.cpu().reshape(-1).to(torch.float64).norm().to(torch.float32)
    actual = pto_cube_norm(x)
    assert actual.shape == (1,)
    assert actual.dtype == torch.float32
    assert actual.device == x.device
    torch.testing.assert_close(
        actual.cpu(),
        expected.reshape(1),
        atol=tolerance * max(expected.item(), 1.0),
        rtol=tolerance,
    )
    return actual


def make_input(shape, dtype, device):
    if dtype == torch.int8:
        # Full int8 range, so the int32 accumulator is exercised as hard as
        # the host's length bound allows.
        return torch.randint(-128, 128, shape, dtype=torch.int8).to(device)
    # The 0.1 scale keeps squares of fp16 inputs well inside fp32's range.
    return (0.1 * torch.randn(shape, device=device)).to(dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize(
    "shape",
    [
        (),
        (0,),
        (2, 0, 3),
        (1,),
        (13,),
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
        (1024, 1024),
    ],
)
def test_cube_norm(shape, dtype, npu_device):
    check_norm(make_input(shape, dtype, npu_device))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("tiles", [21, 39, 41, 61])
@pytest.mark.parametrize("tail", [0, 1, 16383])
def test_cube_norm_uneven_blocks(dtype, tiles, tail, npu_device):
    # Vary tile values to expose omitted or duplicated block ranges exactly.
    length = tiles * 128 * 128 + tail
    x = (torch.arange(length, dtype=torch.int32) // 16384 % 7 - 3).to(dtype)
    check_norm(x.to(npu_device))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_cube_norm_matches_numpy_model(dtype, npu_device):
    # The kernel's tiling, its fp32 accumulation and its tail handling are all
    # modelled by numpy_cube_norm above; both must agree closely.
    for length in (1023, 16385, 327681, 1000003):
        x = make_input((length,), dtype, npu_device)
        expected = numpy_cube_norm(x.cpu().numpy())
        actual = pto_cube_norm(x).cpu().item()
        assert abs(actual - expected) <= 1e-3 * max(abs(expected), 1.0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_cube_norm_views(dtype, npu_device):
    base = make_input((655362,), dtype, npu_device)
    check_norm(base[1:327681])  # contiguous with a nonzero storage offset
    check_norm(base[1::2])
    check_norm(base[:327680].reshape(20, 16384).t())


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize(
    "layout", ["strided", "permuted", "broadcast", "channels_last", "empty"]
)
def test_cube_norm_noncontiguous_layouts(dtype, layout, npu_device):
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
    check_norm(x)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("length", [1, 1023, 1024, 4095, 4096, 16383, 16384, 16385])
def test_cube_norm_offset_excludes_surrounding_storage(dtype, length, npu_device):
    # Poison both sides of a view, including where tail padding would be read.
    base = torch.full((length + 65,), 17, dtype=dtype, device=npu_device)
    x = base[1 : length + 1]
    x.fill_(1)
    assert x.is_contiguous() and x.storage_offset() == 1
    # Every element is one, so the norm is sqrt(length) exactly.
    check_norm(x, tolerance=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("length", [1025, 4097, 16385, 327681])
def test_cube_norm_tile_edge_impulses(dtype, length, npu_device):
    # Exercise row boundaries, tile boundaries, and the final padded element.
    positions = sorted(
        {0, 31, 32, 63, 64, 127, 128, 1023, 1024, 4095, 4096, 16383, 16384, length - 1}
    )
    positions = [index for index in positions if index < length]
    x = torch.zeros(length, dtype=dtype)
    x[positions] = torch.arange(1, len(positions) + 1).to(dtype)
    check_norm(x.to(npu_device), tolerance=1e-6)


@pytest.mark.parametrize("length", [1025, 16385, 327681])
@pytest.mark.parametrize("value", [1.0, 1024.0, 65504.0])
def test_cube_norm_fp16_wide_accumulation(length, value, npu_device):
    # The sum of squares leaves fp16's range long before the norm does, so the
    # accumulation has to stay fp32 all the way to the square root.
    x = torch.full((length,), value, dtype=torch.float16, device=npu_device)
    check_norm(x, tolerance=1e-5)


@pytest.mark.parametrize("length", [1025, 16385, 327681])
@pytest.mark.parametrize("value", [1e-8, 1e10, 1.0000001192092896])
def test_cube_norm_fp32_range_and_precision(length, value, npu_device):
    # One nonzero operand makes the norm |value| exactly, which covers values
    # outside fp16's finite range and reduced input precision.
    x = torch.zeros(length, dtype=torch.float32)
    x[-1] = value
    actual = pto_cube_norm(x.to(npu_device)).cpu().item()
    assert actual == pytest.approx(value, rel=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_cube_norm_zero_input_is_zero(dtype, npu_device):
    x = torch.zeros(327681, dtype=dtype, device=npu_device)
    assert pto_cube_norm(x).cpu().item() == 0.0


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
def test_cube_norm_rejects_dtype(dtype, shape, npu_device):
    with pytest.raises(RuntimeError, match="float16, float32 and int8"):
        pto_cube_norm(torch.ones(shape, dtype=dtype, device=npu_device))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(), (0,), (13,)])
def test_cube_norm_rejects_cpu(dtype, shape):
    with pytest.raises(RuntimeError, match="must be on NPU"):
        pto_cube_norm(torch.ones(shape, dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_cube_norm_rejects_length_over_uint32(dtype, npu_device):
    # A zero-stride view reaches the length guard without a huge allocation.
    x = torch.ones(1, dtype=dtype, device=npu_device).expand(2**32)
    with pytest.raises(RuntimeError, match="input length exceeds uint32 range"):
        pto_cube_norm(x)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_cube_norm_repeated_calls_preserve_input_and_outputs(dtype, npu_device):
    x = torch.full((327681,), 1, dtype=dtype, device=npu_device)
    original = x.clone()
    first = pto_cube_norm(x)
    check_norm(x, tolerance=1e-6)
    torch.testing.assert_close(x, original, atol=0, rtol=0)
    x.zero_()
    second = pto_cube_norm(x)
    assert first.data_ptr() != second.data_ptr()
    assert first.cpu().item() == pytest.approx(327681**0.5, rel=1e-6)
    assert second.cpu().item() == 0.0


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_cube_norm_uses_current_stream(dtype, npu_device):
    stream = torch.npu.Stream(device=npu_device)
    # Queue the producer, two norms, and a consumer on the same stream.
    # Avoid host copies between launches so synchronization bugs stay visible.
    with torch.npu.stream(stream):
        x = torch.full((327681,), 1, dtype=dtype, device=npu_device)
        first = pto_cube_norm(x)
        x.fill_(2)
        second = pto_cube_norm(x)
        combined = first + second
    stream.synchronize()
    assert first.cpu().item() == pytest.approx(327681**0.5, rel=1e-6)
    assert second.cpu().item() == pytest.approx(2 * 327681**0.5, rel=1e-6)
    assert combined.cpu().item() == pytest.approx(3 * 327681**0.5, rel=1e-6)


# --- int8 ---------------------------------------------------------------
#
# int8 operands multiply into an int32 L0C rather than an fp32 one, the
# workspace the grid shares is int32, and the Vector stage widens both that
# tile's diagonal and the ragged tail (via half) before the square root. The
# result is an fp32 norm, as it is for the floating-point inputs.


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (0,),
        (2, 0, 3),
        (1,),
        (13,),
        (1023,),  # shorter than one 32 x 32 tile: tail path only
        (1024,),
        (1025,),
        (4095,),
        (4096,),
        (16383,),  # still below the 128 * 128 cut, so size 32
        (16384,),  # first length that uses size 64
        (16385,),
        (327679,),
        (327680,),
        (327681,),
        (1, 7, 1, 11, 1),
        (1000003,),
        (7, 11, 13),
        (4, 5, 16384),
        (1024, 1024),
        (INT8_MAX_NUMEL,),  # the longest int8 input the host accepts
    ],
)
def test_cube_norm_int8(shape, npu_device):
    check_norm(make_input(shape, torch.int8, npu_device))


@pytest.mark.parametrize("value", [-128, -1, 1, 127])
@pytest.mark.parametrize("length", [1023, 1025, 16385, 327681])
def test_cube_norm_int8_constant(value, length, npu_device):
    # Every element the same makes the norm |value| * sqrt(length) exactly,
    # and -128 is the worst case for the int32 accumulator.
    x = torch.full((length,), value, dtype=torch.int8, device=npu_device)
    actual = pto_cube_norm(x).cpu().item()
    assert actual == pytest.approx(abs(value) * length**0.5, rel=1e-6)


def test_cube_norm_int8_longest_input_is_exact(npu_device):
    # The host's length bound is derived from this exact case: the whole
    # input at -128, which puts every int32 workspace entry at its limit.
    x = torch.full((INT8_MAX_NUMEL,), -128, dtype=torch.int8, device=npu_device)
    actual = pto_cube_norm(x).cpu().item()
    assert actual == pytest.approx(128 * INT8_MAX_NUMEL**0.5, rel=1e-6)


@pytest.mark.parametrize("tiles", [21, 39, 41, 61])
@pytest.mark.parametrize("tail", [0, 1, 16383])
def test_cube_norm_int8_uneven_blocks(tiles, tail, npu_device):
    # Vary tile values to expose omitted or duplicated block ranges exactly.
    length = tiles * 128 * 128 + tail
    x = (torch.arange(length, dtype=torch.int32) // 16384 % 7 - 3).to(torch.int8)
    check_norm(x.to(npu_device))


def test_cube_norm_int8_matches_numpy_model(npu_device):
    for length in (1023, 16385, 327681, 1000003):
        x = make_input((length,), torch.int8, npu_device)
        expected = numpy_cube_norm(x.cpu().numpy())
        actual = pto_cube_norm(x).cpu().item()
        assert abs(actual - expected) <= 1e-5 * max(abs(expected), 1.0)


def test_cube_norm_int8_views(npu_device):
    base = make_input((655362,), torch.int8, npu_device)
    check_norm(base[1:327681])  # contiguous with a nonzero storage offset
    check_norm(base[1::2])
    check_norm(base[:327680].reshape(20, 16384).t())


@pytest.mark.parametrize("length", [1, 1023, 1024, 4095, 4096, 16383, 16384, 16385])
def test_cube_norm_int8_offset_excludes_surrounding_storage(length, npu_device):
    # Poison both sides of a view, including where tail padding would be read.
    base = torch.full((length + 65,), 17, dtype=torch.int8, device=npu_device)
    x = base[1 : length + 1]
    x.fill_(1)
    assert x.is_contiguous() and x.storage_offset() == 1
    # Every element is one, so the norm is sqrt(length) exactly.
    check_norm(x, tolerance=1e-6)


@pytest.mark.parametrize("length", [1025, 16385, 327681])
def test_cube_norm_int8_tile_edge_impulses(length, npu_device):
    # Exercise row boundaries, tile boundaries, and the final padded element.
    positions = sorted(
        {0, 31, 32, 63, 64, 127, 128, 1023, 1024, 4095, 4096, 16383, 16384, length - 1}
    )
    positions = [index for index in positions if index < length]
    x = torch.zeros(length, dtype=torch.int8)
    x[positions] = torch.arange(1, len(positions) + 1).to(torch.int8)
    check_norm(x.to(npu_device), tolerance=1e-6)


def test_cube_norm_int8_zero_input_is_zero(npu_device):
    x = torch.zeros(327681, dtype=torch.int8, device=npu_device)
    assert pto_cube_norm(x).cpu().item() == 0.0


@pytest.mark.parametrize("over", [1, 10**6])
def test_cube_norm_int8_rejects_length_over_accumulator_bound(over, npu_device):
    # A zero-stride view reaches the length guard without a huge allocation.
    x = torch.ones(1, dtype=torch.int8, device=npu_device).expand(INT8_MAX_NUMEL + over)
    with pytest.raises(RuntimeError, match="int8 input length"):
        pto_cube_norm(x)
