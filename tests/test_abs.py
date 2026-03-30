# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------

import torch
from pto_kernels import pto_abs
import pytest


@pytest.mark.parametrize("num_blocks", [1, 2, 10, 20, 32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=str)
def test_pto_abs(num_blocks: int, dtype: torch.dtype):
    # FIXME: support only input length that are multiple of 64.
    tile_len = 64
    length = [num_blocks, tile_len]
    # Create random input tensors on CPU with float16 data type
    x = 2 * torch.rand(length, device="cpu", dtype=dtype) - 1
    # Copy the input tensor to NPU
    x_npu = x.npu()
    # Call the custom my_add operator
    output = pto_abs(x_npu).cpu()
    # Compute the expected result using standard torch.abs on CPU
    cpuout = torch.abs(x)

    # Validate the results
    assert torch.equal(output, cpuout)
