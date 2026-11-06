# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved. See LICENSE in the root of the repository.

import pytest
import torch

from pto_kernels import pto_cube_reduce


def check_sum(x):
    dtype = torch.float32 if x.dtype == torch.float16 else torch.int32
    # Compute the reference on the CPU independently of the NPU reduction.
    expected = x.cpu().reshape(-1).sum(dtype=dtype).reshape(1)
    actual = pto_cube_reduce(x)
    assert actual.shape == (1,)
    assert actual.dtype == dtype
    assert actual.device == x.device
    tolerance = 0.01 if dtype == torch.float32 else 0
    torch.testing.assert_close(actual.cpu(), expected, atol=tolerance, rtol=tolerance)


def make_input(shape, dtype, device):
    if dtype == torch.int8:
        return torch.randint(-3, 3, shape, dtype=dtype, device=device)
    return (0.1 * torch.randn(shape, device=device)).to(dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.int8])
@pytest.mark.parametrize(
    "shape",
    [
        (),
        (0,),
        (1,),
        (13,),
        (1000,),
        (1024,),
        (4096,),
        (16384,),
        (16385,),
        (327679,),
        (327680,),
        (327681,),
        (1000003,),
        (16 * 1024 * 1024 + 7,),
        (7, 11, 13),
        (4, 5, 16384),
    ],
)
def test_cube_reduce(shape, dtype, npu_device):
    check_sum(make_input(shape, dtype, npu_device))


@pytest.mark.parametrize("dtype", [torch.float16, torch.int8])
@pytest.mark.parametrize("tiles", [21, 39, 41, 61])
def test_cube_reduce_uneven_blocks(dtype, tiles, npu_device):
    check_sum(make_input((tiles * 128 * 128,), dtype, npu_device))


@pytest.mark.parametrize("dtype", [torch.float16, torch.int8])
def test_cube_reduce_views(dtype, npu_device):
    base = make_input((655362,), dtype, npu_device)
    check_sum(base[1:327681])  # contiguous with a nonzero storage offset
    check_sum(base[1::2])
    check_sum(base[:327680].reshape(20, 16384).t())


@pytest.mark.parametrize("value", [-128, 127])
def test_cube_reduce_int8_extremes(value, npu_device):
    check_sum(torch.full((327681,), value, dtype=torch.int8, device=npu_device))


def test_cube_reduce_rejects_dtype(npu_device):
    with pytest.raises(RuntimeError, match="float16 and int8"):
        pto_cube_reduce(torch.ones(13, device=npu_device))


def test_cube_reduce_rejects_cpu():
    with pytest.raises(RuntimeError, match="must be on NPU"):
        pto_cube_reduce(torch.ones(13, dtype=torch.float16))
