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


@pytest.mark.parametrize("num_blocks", [1, 2, 10, 20, 32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=str)
def test_pto_abs(num_blocks: int, dtype: torch.dtype):
    # Define the tensor size
    matrix_size = 64
    tile_len = matrix_size * matrix_size
    length = [num_blocks, tile_len]
    # Create random input tensors on CPU with float16 data type
    x = torch.rand(length, device="cpu", dtype=dtype)

    x_npu = x.npu()
    # Call the custom my_add operator
    output = pto_abs(x_npu).cpu()
    # Compute the expected result using standard addition
    cpuout = torch.abs(x)

    # Validate the results
    assert torch.equal(output, cpuout)
