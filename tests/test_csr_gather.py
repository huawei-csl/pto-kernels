# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------

import torch
from pto_kernels import pto_csr_gather
import pytest


def ref_csr_gather(
    values: torch.Tensor, indices: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    return values * x[indices]


x_size = [512, 1024, 2048, 16384, 32768, 40960]
v_size = [100, 128, 256, 600, 1024, 1200]
# Sweep only for v < x
sweep_sizes = [(x, v) for x in x_size for v in v_size if v < x]


@pytest.mark.parametrize("x_size, v_size", sweep_sizes)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=str)
def test_pto_csr_gather(x_size: int, v_size: int, dtype: torch.dtype):
    # Create random input tensors on CPU
    x = torch.rand((x_size,), device="cpu", dtype=dtype)
    values = torch.rand((v_size,), device="cpu", dtype=dtype)
    indices = torch.randint(0, x_size, (v_size,), device="cpu", dtype=torch.int32)
    # Copy the input tensors to NPU
    x_npu = x.npu()
    values_npu = values.npu()
    indices_npu = indices.npu()
    # Call the custom csr_gather operator
    output = pto_csr_gather(values_npu, indices_npu, x_npu).cpu()
    # Compute the expected result using a reference implementation on CPU
    cpuout = ref_csr_gather(values, indices, x)
    # Validate the results
    assert torch.allclose(output, cpuout)


if __name__ == "__main__":
    test_pto_csr_gather(32768, 16384, torch.float16)
