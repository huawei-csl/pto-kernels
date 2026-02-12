# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
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
