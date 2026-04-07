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


@pytest.mark.parametrize("size", [1, 8, 50, 256, 1000])
@pytest.mark.parametrize("bins", [2, 4, 32, 100, 256])
# torch.float16 requires different boundary calculation to match torch implementeation
@pytest.mark.parametrize("dtype", [torch.float32], ids=str)
def test_pto_histogram(size: int, bins: int, dtype: torch.dtype):
    # Tile size is fixed in the kernel
    tile_size = 512
    num_cores = torch.npu.get_device_properties(0).vector_core_num
    num_tiles = num_cores * tile_size
    total_len = num_tiles * size

    x = torch.randint(high=bins, size=(total_len,), device="cpu", dtype=dtype)

    # Golden PyTorch implementation
    y_cpu = torch.histc(x, bins=bins).float()

    # NPU kernel execution, test to see if any race conditions occur across multiple runs
    x_npu = x.npu().contiguous()
    y_npu = []
    repeat_runs = 100
    for _ in range(repeat_runs):
        y_npu.append(pto_histogram(x_npu, bins=bins).cpu().float())

    torch.npu.synchronize()

    assert all(torch.equal(y_cpu, y) for y in y_npu)
