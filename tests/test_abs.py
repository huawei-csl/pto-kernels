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

pytestmark = pytest.mark.npu


@pytest.mark.parametrize("size0", [1, 2, 3, 10, 20, 64, 128])
@pytest.mark.parametrize("size1", [1, 2, 3, 10, 20, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=str)
def test_pto_abs(size0: int, size1: int, dtype: torch.dtype):
    size = [size0, size1]
    # Create random input tensors on CPU
    x = 2 * torch.rand(size, device="cpu", dtype=dtype) - 1
    # Copy the input tensor to NPU
    x_npu = x.npu()
    # breakpoint()
    # Call the custom abs operator
    output = pto_abs(x_npu).cpu()
    # Compute the expected result using standard torch.abs on CPU
    cpuout = torch.abs(x)
    # Validate the results
    assert torch.allclose(output, cpuout)
