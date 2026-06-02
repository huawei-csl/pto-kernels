# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------

import random

import numpy as np
import pytest
import torch

from pto_kernels import pto_cube_reduce

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

NUM_BLOCKS = 20


def get_lengths(s: int, max_iters: int):
    """Yield vec_len values that are divisible by NUM_BLOCKS * s * s."""
    for multiplier in range(1, max_iters):
        yield multiplier * NUM_BLOCKS * s * s


def _test_cube_reduce(vec_len: int):
    """
    Verify pto_cube_reduce against a reference PyTorch implementation.

    The kernel splits `x` into NUM_BLOCKS equal partitions and returns the
    sum of each partition as a float32 scalar.
    """
    x = 0.1 * torch.randn(vec_len, dtype=torch.float16).npu()

    torch.npu.synchronize()
    # Reference: reshape into NUM_BLOCKS rows, sum each row in float32.
    expected = x.reshape(NUM_BLOCKS, -1).sum(dim=1, dtype=torch.float32)
    torch.npu.synchronize()

    actual = pto_cube_reduce(x, NUM_BLOCKS)
    torch.npu.synchronize()

    assert actual.dtype == torch.float32, f"Expected float32, got {actual.dtype}"
    assert (
        actual.shape == expected.shape
    ), f"Shape mismatch: expected {expected.shape}, got {actual.shape}"
    assert torch.allclose(
        actual, expected, atol=1e-0, rtol=1e-2
    ), f"Expected: {expected}\nActual:   {actual}"


@pytest.mark.parametrize("vec_len", get_lengths(s=128, max_iters=16))
def test_cube_reduce_fp16(vec_len: int):
    """Block-wise sum reduction via Cube matmul, fp16 input → float32 output."""
    _test_cube_reduce(vec_len)
