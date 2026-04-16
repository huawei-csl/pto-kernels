from pathlib import Path

import pytest
import torch
import torch_npu  # noqa

from jit_util_layernorm import jit_compile

DTYPE = torch.float16
KERNEL_CPP = Path(__file__).resolve().parent / "kernel_layernorm.cpp"
EPS = 1e-5

TEST_ROWS = [1, 2, 5, 8, 10, 16, 20, 32, 40, 64, 128, 256, 512, 1024, 2048, 4096]
TEST_HIDDEN_DIMS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
TEST_SEEDS = [0, 1]
TEST_LARGE_HIDDEN_MIN = 16384
TEST_LARGE_HIDDEN_MAX_ROWS = 64
TEST_CASES = [
    (rows, hidden)
    for rows in TEST_ROWS
    for hidden in TEST_HIDDEN_DIMS
    if hidden < TEST_LARGE_HIDDEN_MIN or rows <= TEST_LARGE_HIDDEN_MAX_ROWS
]


def layernorm_ref(x, gamma, beta, eps=EPS):
    x_fp = x.float()
    mean = x_fp.mean(dim=-1, keepdim=True)
    var = x_fp.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x_fp - mean) / (var + eps).sqrt()
    return (x_norm * gamma.float() + beta.float()).to(x.dtype)


@pytest.fixture(scope="session")
def layernorm_kernel(npu_device):
    return jit_compile(str(KERNEL_CPP), verbose=True, device=npu_device)


@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize("rows,hidden", TEST_CASES)
def test_layernorm_matches_reference(
    layernorm_kernel,
    npu_device,
    seed,
    rows,
    hidden,
):
    torch.manual_seed(seed)
    x = torch.randn(rows, hidden, device=npu_device, dtype=DTYPE)
    gamma = torch.randn(hidden, device=npu_device, dtype=DTYPE)
    beta = torch.randn(hidden, device=npu_device, dtype=DTYPE)
    y = torch.empty(rows, hidden, device=npu_device, dtype=DTYPE)

    layernorm_kernel(x, gamma, beta, y, eps=EPS)
    torch.npu.synchronize()

    ref = layernorm_ref(x.cpu(), gamma.cpu(), beta.cpu(), eps=EPS)
    torch.testing.assert_close(y.cpu(), ref, rtol=1e-2, atol=1e-2)
