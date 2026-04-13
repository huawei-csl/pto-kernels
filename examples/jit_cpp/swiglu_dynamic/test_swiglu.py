from pathlib import Path

import pytest
import torch
import torch_npu  # noqa

from jit_util_swiglu import jit_compile

DTYPE = torch.float16
KERNEL_CPP = Path(__file__).resolve().parent / "swiglu_dynamic.cpp"
TEST_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
TEST_HIDDEN_DIMS = [128, 1024, 4096, 6144, 16384]
TEST_SEEDS = [0, 1]


def swiglu_ref(x):
    a, b = torch.chunk(x.float(), 2, dim=-1)
    return (a * torch.sigmoid(a) * b).to(x.dtype)


@pytest.fixture(scope="session")
def swiglu_kernel(npu_device):
    return jit_compile(str(KERNEL_CPP), verbose=True, device=npu_device)


@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize("batch", TEST_BATCHES)
@pytest.mark.parametrize("n", TEST_HIDDEN_DIMS)
def test_swiglu_matches_reference_and_torch_npu(
    swiglu_kernel,
    npu_device,
    seed,
    batch,
    n,
):
    torch.manual_seed(seed)
    x = torch.randn(batch, n * 2, device=npu_device, dtype=DTYPE)
    y = torch.empty(batch, n, device=npu_device, dtype=DTYPE)

    swiglu_kernel(x, y)
    torch.npu.synchronize()

    ref = swiglu_ref(x.cpu())
    torch_npu_ref = torch_npu.npu_swiglu(x, dim=-1)

    torch.testing.assert_close(y.cpu(), ref, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(y, torch_npu_ref, rtol=3e-2, atol=3e-2)
