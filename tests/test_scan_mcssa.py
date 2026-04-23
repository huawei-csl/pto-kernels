# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------

import torch
import pytest
from pto_kernels import pto_scan_mcssa

# size = [32]
# matrix_size = [s * s for s in size]


@pytest.mark.parametrize("scan_size", [ 16 * 32])
@pytest.mark.parametrize("dtype", [torch.float32], ids=str)
def test_pto_scan_mcssa(scan_size: int, dtype: torch.dtype):
    a = torch.ones(scan_size, dtype=dtype)

    a_npu = a.npu()
    scan_npu = pto_scan_mcssa(a_npu)

    ref = torch.cumsum(a.to(torch.float32), dim=0)

    print("Scan result on NPU:", scan_npu.cpu())
    print("Reference result:", ref)

    assert torch.allclose(scan_npu.cpu(), ref)


if __name__ == "__main__":
    test_pto_scan_mcssa(128 * 128, torch.float32)
