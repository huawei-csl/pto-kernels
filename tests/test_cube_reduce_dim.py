# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved. See LICENSE in the root of the repository.

import pytest
import torch

from pto_kernels import pto_cube_reduce_dim

DTYPES = [torch.float16, torch.float32, torch.int8]


# The tile load carries its source row pitch in a 16-bit field, so a reduction
# whose tiles are cut across a longer axis is rejected on the host instead.
MAX_ROW_PITCH = 65535


def out_dtype(dtype):
    return torch.int32 if dtype == torch.int8 else torch.float32


def row_pitch(shape, dim):
    """The input extent the kernel uses as its tile row pitch."""
    axis = dim % len(shape)
    inner = 1
    for size in shape[axis + 1 :]:
        inner *= size
    return shape[axis] if inner == 1 else inner


def check_sum(x, dim, *, exact=False):
    if row_pitch(tuple(x.shape), dim) > MAX_ROW_PITCH:
        with pytest.raises(RuntimeError, match="row pitch"):
            pto_cube_reduce_dim(x, dim)
        return None
    dtype = out_dtype(x.dtype)
    # A wider CPU reference keeps the comparison honest for fp32 cancellation.
    reference_dtype = torch.float64 if x.dtype == torch.float32 else dtype
    expected = x.cpu().sum(dim=dim, dtype=reference_dtype).to(dtype)
    actual = pto_cube_reduce_dim(x, dim)
    assert actual.shape == expected.shape
    assert actual.dtype == dtype
    assert actual.device == x.device
    tolerance = 0.01 if dtype == torch.float32 and not exact else 0
    torch.testing.assert_close(actual.cpu(), expected, atol=tolerance, rtol=tolerance)
    return actual


def make_input(shape, dtype, device):
    if dtype == torch.int8:
        return torch.randint(-3, 3, shape, dtype=dtype, device=device)
    return (0.1 * torch.randn(shape, device=device)).to(dtype)


def exact_pattern(shape, dtype):
    # Values vary across the tensor so a dropped or doubled range changes some
    # output element. They stay in [-3, 3], exact in every supported dtype.
    length = 1
    for size in shape:
        length *= size
    flat = (torch.arange(length, dtype=torch.int32) // 251 % 7 - 3).to(dtype)
    return flat.reshape(shape)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dim", [0, 1, 2, -1, -2, -3])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 1),
        (1, 1, 5),
        (1, 5, 1),
        (5, 1, 1),
        (2, 3, 4),
        (7, 11, 13),
        (17, 19, 23),
        (1, 1, 4096),
        (1, 4096, 1),
        (4096, 1, 1),
        (3, 128, 128),
        (3, 127, 129),
        (2, 256, 256),
        (64, 64, 64),
        (1, 1000, 1000),
        (128, 5, 777),
    ],
)
def test_reduce_dim(shape, dim, dtype, npu_device):
    check_sum(make_input(shape, dtype, npu_device), dim)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize(
    "shape",
    [
        # The cube path tiles the kept inner axis and the reduced axis in
        # squares, so sizes one short of, exactly at, and one past a tile
        # boundary are the cases where padding and the ragged edge matter.
        (3, 31, 31),
        (3, 32, 32),
        (3, 33, 33),
        (3, 63, 63),
        (3, 64, 64),
        (3, 65, 65),
        (3, 127, 127),
        (3, 128, 128),
        (3, 129, 129),
        (3, 128, 1),
        (3, 1, 128),
        (3, 128, 127),
        (3, 127, 128),
        (3, 256, 129),
        (3, 129, 256),
        # The tile is held at a constant volume, so its width ranges over
        # 32..512 and its height over 512..32; these cover the ragged edge at
        # the wide end as the shapes above cover it at the narrow end.
        (3, 255, 257),
        (3, 256, 256),
        (3, 257, 255),
        (2, 511, 513),
        (2, 512, 512),
        (2, 513, 511),
        (2, 33, 1025),
        (2, 1025, 33),
        # Exercise an outer extent that does not divide the core count.
        (41, 37, 67),
        (61, 7, 129),
    ],
)
def test_reduce_dim_tile_boundaries(shape, dim, dtype, npu_device):
    check_sum(exact_pattern(shape, dtype).to(npu_device), dim, exact=True)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize(
    "shape", [(0, 3, 4), (3, 0, 4), (3, 4, 0), (0, 0, 0), (1, 0, 1)]
)
def test_reduce_dim_empty(shape, dim, dtype, npu_device):
    # Reducing a zero-length axis yields zeros; a zero-length kept axis yields
    # an empty result. Both must match torch.sum exactly.
    check_sum(torch.ones(shape, dtype=dtype, device=npu_device), dim, exact=True)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dim", [0, 1, 2])
def test_reduce_dim_impulses(dtype, dim, npu_device):
    # A single nonzero per slice localizes any tiling or indexing error.
    shape = (5, 130, 131)
    x = torch.zeros(shape, dtype=dtype)
    x[0, 0, 0] = 1
    x[4, 129, 130] = 2
    x[2, 64, 65] = 3
    x[1, 128, 1] = -1
    x[3, 1, 128] = -2
    check_sum(x.to(npu_device), dim, exact=True)


@pytest.mark.parametrize("dim", [0, 1, 2])
def test_reduce_dim_fp16_wide_accumulation(dim, npu_device):
    # A slice may sum beyond fp16's range; accumulation must stay fp32.
    x = torch.full((40, 40, 40), 1000, dtype=torch.float16, device=npu_device)
    check_sum(x, dim, exact=True)


@pytest.mark.parametrize("dim", [0, 1, 2])
def test_reduce_dim_fp16_cancellation(dim, npu_device):
    x = torch.empty((8, 129, 130), dtype=torch.float16)
    flat = x.reshape(-1)
    flat[0::2] = 32
    flat[1::2] = -32
    check_sum(x.to(npu_device), dim, exact=True)


@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("value", [-128, 127])
def test_reduce_dim_int8_extremes(dim, value, npu_device):
    x = torch.full((17, 130, 131), value, dtype=torch.int8, device=npu_device)
    check_sum(x, dim, exact=True)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dim", [0, 1, 2])
def test_reduce_dim_accepts_views(dtype, dim, npu_device):
    # A contiguous view with a nonzero storage offset, and a non-contiguous
    # view that the host must materialize before launching.
    base = make_input((6, 130, 131), dtype, npu_device)
    offset_view = base[1:5]
    assert offset_view.is_contiguous() and offset_view.storage_offset() != 0
    check_sum(offset_view, dim)
    permuted = base.permute(2, 0, 1)
    assert not permuted.is_contiguous()
    check_sum(permuted, dim)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dim", [0, 1, 2])
def test_reduce_dim_excludes_surrounding_storage(dtype, dim, npu_device):
    # Poison the storage around a view so any over-read past a slice, a tile,
    # or the end of the tensor changes the result.
    shape = (4, 129, 130)
    numel = shape[0] * shape[1] * shape[2]
    base = torch.full((numel + 2048,), 17, dtype=dtype, device=npu_device)
    x = base[1 : numel + 1].reshape(shape)
    x.fill_(1)
    assert x.is_contiguous() and x.storage_offset() == 1
    check_sum(x, dim, exact=True)


@pytest.mark.parametrize("dtype", DTYPES)
def test_reduce_dim_repeated_calls_preserve_input(dtype, npu_device):
    x = torch.full((5, 130, 131), 1, dtype=dtype, device=npu_device)
    original = x.clone()
    first = check_sum(x, 1, exact=True)
    second = check_sum(x, 1, exact=True)
    torch.testing.assert_close(x, original, atol=0, rtol=0)
    assert first.data_ptr() != second.data_ptr()


@pytest.mark.parametrize("dtype", DTYPES)
def test_reduce_dim_uses_current_stream(dtype, npu_device):
    stream = torch.npu.Stream(device=npu_device)
    with torch.npu.stream(stream):
        x = torch.full((5, 130, 131), 1, dtype=dtype, device=npu_device)
        first = pto_cube_reduce_dim(x, 1)
        x.fill_(2)
        second = pto_cube_reduce_dim(x, 1)
    stream.synchronize()
    assert first.cpu().flatten()[0].item() == 130
    assert second.cpu().flatten()[0].item() == 260


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(2, 3, 4, 5, 6, 7), (2,) * 7, (1, 1, 1, 1, 1, 1)])
def test_reduce_dim_rejects_rank_over_five(dtype, shape, npu_device):
    x = torch.ones(shape, dtype=dtype, device=npu_device)
    with pytest.raises(RuntimeError, match="rank 1 to 5"):
        pto_cube_reduce_dim(x, 0)


@pytest.mark.parametrize("dim", [3, -4, 100])
def test_reduce_dim_rejects_out_of_range_dim(dim, npu_device):
    x = torch.ones((2, 3, 4), dtype=torch.float16, device=npu_device)
    with pytest.raises(RuntimeError, match="dim"):
        pto_cube_reduce_dim(x, dim)


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
def test_reduce_dim_rejects_dtype(dtype, npu_device):
    x = torch.ones((2, 3, 4), dtype=dtype, device=npu_device)
    with pytest.raises(RuntimeError, match="float16, float32 and int8"):
        pto_cube_reduce_dim(x, 0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_reduce_dim_rejects_cpu(dtype):
    x = torch.ones((2, 3, 4), dtype=dtype)
    with pytest.raises(RuntimeError, match="must be on NPU"):
        pto_cube_reduce_dim(x, 0)


@pytest.mark.parametrize(
    "shape,dim",
    [
        ((2, 256, 256), 0),  # inner folds to 65536, one past the limit
        ((1, 1000, 1000), 0),  # inner folds to 1000000
        ((2, 3, 100000), 2),  # the reduced axis is itself the pitch
    ],
)
def test_reduce_dim_rejects_oversized_row_pitch(shape, dim, npu_device):
    x = torch.ones(shape, dtype=torch.float16, device=npu_device)
    assert row_pitch(shape, dim) > MAX_ROW_PITCH
    with pytest.raises(RuntimeError, match="row pitch"):
        pto_cube_reduce_dim(x, dim)


@pytest.mark.parametrize(
    "shape,dim",
    [((2, 255, 257), 0), ((2, 3, 65535), 2), ((2, 65535, 3), 1)],
)
def test_reduce_dim_accepts_pitch_at_the_limit(shape, dim, npu_device):
    # The largest pitch the load supports must still go through the kernel.
    x = torch.ones(shape, dtype=torch.float16, device=npu_device)
    assert row_pitch(shape, dim) <= MAX_ROW_PITCH
    check_sum(x, dim, exact=True)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "shape",
    [
        # Rank reaches the kernel only as the three folded extents, so these
        # check that the fold itself is right at every rank: each shape is
        # reduced along every one of its axes and compared with torch.sum.
        (37,),
        (1,),
        (13, 17),
        (1, 129),
        (129, 1),
        (5, 7, 11),
        (3, 4, 5, 6),
        (2, 3, 4, 5, 6),
        (1, 3, 1, 5, 1),
        (2, 1, 64, 1, 3),
        (7, 1, 1, 1, 1),
        (1, 1, 1, 1, 7),
        (4, 4, 4, 4, 4),
        (2, 2, 130, 2, 2),
    ],
)
def test_reduce_dim_every_rank_and_axis(shape, dtype, npu_device):
    for dim in range(len(shape)):
        check_sum(exact_pattern(shape, dtype).to(npu_device), dim, exact=True)
        check_sum(
            exact_pattern(shape, dtype).to(npu_device),
            dim - len(shape),
            exact=True,
        )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(37,), (13, 17), (3, 4, 5, 6), (2, 3, 4, 5, 6)])
def test_reduce_dim_matches_a_folded_three_dimensional_input(shape, dtype, npu_device):
    # Folding the axes by hand must give the identical result, since that is
    # exactly what the host does before the kernel sees anything.
    x = exact_pattern(shape, dtype).to(npu_device)
    for dim in range(len(shape)):
        outer, inner = 1, 1
        for size in shape[:dim]:
            outer *= size
        for size in shape[dim + 1 :]:
            inner *= size
        folded = x.reshape(outer, shape[dim], inner)
        if row_pitch(tuple(x.shape), dim) > MAX_ROW_PITCH:
            continue
        torch.testing.assert_close(
            pto_cube_reduce_dim(x, dim).reshape(outer, inner).cpu(),
            pto_cube_reduce_dim(folded, 1).cpu(),
            atol=0,
            rtol=0,
        )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rank", [1, 2, 4, 5])
def test_reduce_dim_empty_at_every_rank(rank, dtype, npu_device):
    for zero_axis in range(rank):
        shape = tuple(0 if i == zero_axis else 3 for i in range(rank))
        for dim in range(rank):
            check_sum(
                torch.ones(shape, dtype=dtype, device=npu_device), dim, exact=True
            )


@pytest.mark.parametrize("dtype", DTYPES)
def test_reduce_dim_rank_one_is_a_full_reduction(dtype, npu_device):
    # A 1D input folds to outer = inner = 1, so the result is a scalar tensor
    # and the whole reduction lands on one work item.
    x = exact_pattern((4096,), dtype).to(npu_device)
    actual = pto_cube_reduce_dim(x, 0)
    assert actual.shape == ()
    torch.testing.assert_close(
        actual.cpu().reshape(1),
        x.cpu().sum(dtype=out_dtype(dtype)).reshape(1),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rank", [4, 5])
def test_reduce_dim_accepts_views_at_higher_rank(rank, dtype, npu_device):
    shape = (6, 5, 4, 3, 2)[:rank]
    base = make_input(shape, dtype, npu_device)
    offset_view = base[1:5]
    assert offset_view.is_contiguous() and offset_view.storage_offset() != 0
    for dim in range(rank):
        check_sum(offset_view, dim)
    permuted = base.permute(*reversed(range(rank)))
    assert not permuted.is_contiguous()
    for dim in range(rank):
        check_sum(permuted, dim)
