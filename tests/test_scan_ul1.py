# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------

import torch
import pytest
from pto_kernels import pto_scan_ul1

size = [16, 32, 64, 128]
matrix_size = [s * s for s in size]


@pytest.mark.parametrize("scan_size", matrix_size)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=str)
def test_pto_scan_ul1(scan_size: int, dtype: torch.dtype):
    a = torch.ones(scan_size, dtype=dtype)

    a_npu = a.npu()
    scan_npu = pto_scan_ul1(a_npu)

    ref = torch.cumsum(a.to(torch.float32), dim=0)

    assert torch.allclose(scan_npu.cpu(), ref)


if __name__ == "__main__":
    test_pto_scan_ul1(128 * 128, torch.float32)
