import math
from pathlib import Path

import pytest
import torch

from fuse_int8_quant.jit_util_hadamard_quant import jit_compile

TEST_BATCHES = [1, 7, 65]
TEST_HIDDEN_DIMS = [128, 1024, 16384]
TEST_SCALES = [0.5, 1.0, 2.0]
TEST_SEEDS = [0, 1]
DTYPE = torch.float16


def hadamard_ref_inplace(x):
    x = x.clone()
    n = x.shape[-1]
    n_half = n // 2
    log2_n = int(math.log2(n))

    for _ in range(log2_n):
        even = x[..., 0::2].clone()
        odd = x[..., 1::2].clone()
        x[..., :n_half] = even + odd
        x[..., n_half:] = even - odd
    return x


def quantize_ref(x, scale):
    scale_half = torch.tensor(scale, device=x.device, dtype=DTYPE)
    y = torch.round((x * scale_half).float())
    y = torch.clamp(y, -128, 127)
    return y.to(torch.int8)


def fused_ref(x, scale):
    return quantize_ref(hadamard_ref_inplace(x), scale)


def run_fused(fused_kernel, x, y, scale, log2_n=None):
    n = x.shape[-1]
    if log2_n is None:
        log2_n = int(math.log2(n))
    fused_kernel(x, y, x.shape[0], n, log2_n, scale)
    torch.npu.synchronize()


@pytest.fixture(scope="session")
def hadamard_quant_kernel(npu_device):
    base = Path(__file__).resolve().parent
    src = base / "fast_hadamard_quant.cpp"
    return jit_compile(str(src), verbose=True, device=npu_device)


@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize("scale", TEST_SCALES)
@pytest.mark.parametrize("batch", TEST_BATCHES)
@pytest.mark.parametrize("n", TEST_HIDDEN_DIMS)
def test_fast_hadamard_quant_correctness(
    hadamard_quant_kernel, npu_device, seed, scale, batch, n
):
    torch.manual_seed(seed)
    x = torch.randn(batch, n, device=npu_device, dtype=DTYPE)
    y = torch.empty(batch, n, device=npu_device, dtype=torch.int8)

    y_ref = fused_ref(x, scale)

    run_fused(hadamard_quant_kernel, x, y, scale)

    assert torch.equal(
        y, y_ref
    ), f"Mismatch for seed={seed}, batch={batch}, N={n}, scale={scale}"


def test_fast_hadamard_quant_does_not_mutate_input(hadamard_quant_kernel, npu_device):
    x = torch.randn(7, 1024, device=npu_device, dtype=DTYPE)
    x_before = x.clone()
    y = torch.empty_like(x, dtype=torch.int8)

    run_fused(hadamard_quant_kernel, x, y, 1.0)

    assert torch.equal(x, x_before)


@pytest.mark.parametrize("n", [3, 257, 16385])
def test_fast_hadamard_quant_rejects_unsupported_n(
    hadamard_quant_kernel, npu_device, n
):
    x = torch.randn(2, n, device=npu_device, dtype=DTYPE)
    y = torch.empty(2, n, device=npu_device, dtype=torch.int8)

    with pytest.raises(ValueError, match="n must"):
        hadamard_quant_kernel(x, y, scale=1.0)


def test_fast_hadamard_quant_rejects_incorrect_log2_n(
    hadamard_quant_kernel, npu_device
):
    x = torch.randn(2, 1024, device=npu_device, dtype=DTYPE)
    y = torch.empty(2, 1024, device=npu_device, dtype=torch.int8)

    with pytest.raises(ValueError, match="log2_n must equal"):
        hadamard_quant_kernel(x, y, scale=1.0, log2_n=9)
