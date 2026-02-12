# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------

import torch
import pytest
import torch
from pto_kernels import pto_simple_matmul


@pytest.mark.parametrize("matrix_size", [16, 32, 64, 96, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=str)
def test_pto_simple_matmul(matrix_size: int, dtype: torch.dtype):
    m, k, n = matrix_size, matrix_size, matrix_size
    a = torch.rand((m, k), device="cpu", dtype=dtype)
    b = torch.rand((k, n), device="cpu", dtype=dtype)

    a_npu = a.npu()
    b_npu = b.npu()
    c_npu = pto_simple_matmul(a_npu, b_npu)

    ref = torch.matmul(a.float(), b.float())
    assert torch.allclose(c_npu.cpu(), ref)
