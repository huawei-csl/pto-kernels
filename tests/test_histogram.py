# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------

import torch
from pto_kernels import pto_histogram
import pytest


#@pytest.mark.parametrize("num_blocks", [1, 2, 10, 20, 32, 64])
#@pytest.mark.parametrize("bins", [2, 4, 16, 50, 100])
@pytest.mark.parametrize("num_blocks", [1])
@pytest.mark.parametrize("bins", [64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=str)
def test_pto_histogram(num_blocks: int, bins: int, dtype: torch.dtype):
    tile_len = 64
    length = [num_blocks * tile_len]
    
    x = torch.rand(length, device="cpu", dtype=dtype)
    x_npu = x.npu()

    y_npu = pto_histogram(x_npu, bins=bins).cpu()
    y_cpu = torch.histc(x, bins=bins)
    
    assert torch.equal(y_npu, y_cpu)
