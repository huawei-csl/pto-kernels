import os
from pathlib import Path

import pytest
import torch

from copy_vs_hadamard.jit_util_copy_pto import jit_compile as jit_compile_copy_pto
from copy_vs_hadamard.jit_util_copy_raw_cce import (
    jit_compile as jit_compile_copy_raw_cce,
)
from copy_vs_hadamard.jit_util_copy_common import validate_copy_tensors

TEST_SHAPES = [(1, 1), (2, 32), (7, 128), (8, 2048)]


@pytest.fixture(scope="session")
def copy_pto_kernel(npu_device):
    base = Path(__file__).resolve().parent
    return jit_compile_copy_pto(
        str(base / "copy_pto.cpp"),
        verbose=True,
        device=npu_device,
    )


@pytest.fixture(scope="session")
def copy_raw_cce_kernel(npu_device, copy_pto_kernel):
    base = Path(__file__).resolve().parent
    return jit_compile_copy_raw_cce(
        str(base / "copy_raw_cce.cpp"),
        verbose=True,
        device=npu_device,
        block_dim=copy_pto_kernel.block_dim,
    )


@pytest.fixture(scope="session")
def copy_raw_cce_static_4096_kernel(npu_device, copy_pto_kernel):
    base = Path(__file__).resolve().parent
    return jit_compile_copy_raw_cce(
        str(base / "copy_raw_cce_static_4096_4096.cpp"),
        verbose=True,
        device=npu_device,
        block_dim=copy_pto_kernel.block_dim,
    )


def _run_copy(copy_kernel, x, y):
    copy_kernel(x, y, x.shape[0], x.shape[1])
    torch.npu.synchronize()


@pytest.mark.parametrize("batch,n", TEST_SHAPES)
def test_copy_pto_correctness(copy_pto_kernel, npu_device, batch, n):
    x = torch.randn(batch, n, device=npu_device, dtype=torch.float16)
    y = torch.empty_like(x)
    _run_copy(copy_pto_kernel, x, y)
    assert torch.equal(y, x)


@pytest.mark.parametrize("batch,n", TEST_SHAPES)
def test_copy_raw_cce_correctness(copy_raw_cce_kernel, npu_device, batch, n):
    x = torch.randn(batch, n, device=npu_device, dtype=torch.float16)
    y = torch.empty_like(x)
    _run_copy(copy_raw_cce_kernel, x, y)
    assert torch.equal(y, x)


def test_static_copy_wrapper_rejects_wrong_shape(npu_device):
    x = torch.randn(1, 1024, device=npu_device, dtype=torch.float16)
    y = torch.empty_like(x)
    with pytest.raises(ValueError, match="batch and n must match the input tensor shape."):
        validate_copy_tensors(x, y, batch=4096, n=4096)


@pytest.mark.skipif(
    os.environ.get("PTO_RUN_LARGE_COPY_STATIC_TEST") != "1",
    reason="set PTO_RUN_LARGE_COPY_STATIC_TEST=1 to run the 4096x4096 static copy test",
)
def test_static_copy_kernel_correctness(copy_raw_cce_static_4096_kernel, npu_device):
    x = torch.randn(4096, 4096, device=npu_device, dtype=torch.float16)
    y = torch.empty_like(x)
    _run_copy(copy_raw_cce_static_4096_kernel, x, y)
    assert torch.equal(y, x)
