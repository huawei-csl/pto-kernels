# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------

import os
import random

import numpy as np
import pytest
import torch
from pto_kernels import pto_cube_reduce

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

NUM_BLOCKS = 20

NPU_DEVICE = os.environ.get("NPU_DEVICE", "npu:1")


def get_lengths(s: int, max_iters: int):
    for multiplier in range(1, max_iters):
        yield multiplier * NUM_BLOCKS * s * s


def _test_cube_reduce(vec_len: int, dtype: torch.dtype):
    if dtype == torch.float16:
        x = 0.1 * torch.randn(vec_len, dtype=dtype, device=NPU_DEVICE)
        out_dtype = torch.float32
    elif dtype == torch.int8:
        x = torch.randint(-3, 3, size=(vec_len,), dtype=torch.int8, device=NPU_DEVICE)
        out_dtype = torch.int32
    else:
        assert False, f"Unsupported dtype for cube_reduce. Got {dtype}."

    torch.npu.synchronize()
    expected = x.reshape(NUM_BLOCKS, -1).sum(dim=1, dtype=out_dtype).flatten()
    torch.npu.synchronize()
    actual = pto_cube_reduce(x, NUM_BLOCKS)
    torch.npu.synchronize()

    assert expected.dtype == actual.dtype
    assert expected.shape == actual.shape
    assert torch.allclose(
        actual, expected, atol=1e-0, rtol=1e-2
    ), f"Expected : {expected}. Actual: {actual}"


@pytest.mark.parametrize("vec_len", get_lengths(s=128, max_iters=16))
@pytest.mark.parametrize("offset", [0])  # TODO(anastasios): test unpadded cases
@pytest.mark.parametrize("dtype", [torch.int8, torch.float16], ids=str)
def test_cube_reduce(vec_len: int, offset: int, dtype: torch.dtype):
    _test_cube_reduce(vec_len + offset, dtype)
