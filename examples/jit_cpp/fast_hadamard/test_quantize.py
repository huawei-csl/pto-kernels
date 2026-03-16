from pathlib import Path

import pytest
import torch

from jit_util_quantize import jit_compile

TEST_BATCHES = [1, 7, 65]
TEST_HIDDEN_DIMS = [128, 1024, 16384]
TEST_SCALES = [0.5, 1.0, 2.0]
TEST_SEEDS = [0, 1]
EDGE_DIMS = [
    1,
    2,
    3,
    31,
    32,
    33,
    127,
    128,
    129,
    255,
    257,
    1023,
    1025,
    16383,
    16384,
    16385,
]
ARBITRARY_SHAPES = [(1, 5), (2, 17), (3, 257), (7, 1000)]
DTYPE = torch.float16


def quantize_ref(x, scale):
    scale_half = torch.tensor(scale, device=x.device, dtype=DTYPE)
    y = torch.round((x * scale_half).float())
    y = torch.clamp(y, -128, 127)
    return y.to(torch.int8)


def assert_quantize_matches_ref(quantize_kernel, x, scale):
    y = torch.empty_like(x, dtype=torch.int8)
    y_ref = quantize_ref(x, scale)

    quantize_kernel(x, y, scale)
    torch.npu.synchronize()

    assert torch.equal(y, y_ref)


@pytest.fixture(scope="session")
def quantize_kernel(npu_device):
    base = Path(__file__).resolve().parent
    src = base / "quantize.cpp"
    return jit_compile(str(src), verbose=True, device=npu_device)


@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize("scale", TEST_SCALES)
@pytest.mark.parametrize("batch", TEST_BATCHES)
@pytest.mark.parametrize("n", TEST_HIDDEN_DIMS)
def test_quantize_correctness(quantize_kernel, npu_device, seed, scale, batch, n):
    torch.manual_seed(seed)
    x = torch.randn(batch, n, device=npu_device, dtype=DTYPE)
    assert_quantize_matches_ref(quantize_kernel, x, scale)


@pytest.mark.parametrize("n", EDGE_DIMS)
def test_quantize_edge_lengths(quantize_kernel, npu_device, n):
    x = torch.linspace(-4.0, 4.0, n, device=npu_device, dtype=DTYPE).reshape(1, n)
    assert_quantize_matches_ref(quantize_kernel, x, 1.0)


@pytest.mark.parametrize("batch,n", ARBITRARY_SHAPES)
def test_quantize_arbitrary_shapes(quantize_kernel, npu_device, batch, n):
    x = torch.linspace(-6.0, 6.0, batch * n, device=npu_device, dtype=DTYPE).reshape(
        batch, n
    )
    assert_quantize_matches_ref(quantize_kernel, x, 0.75)


@pytest.mark.parametrize("scale", [0.0, -0.5, -1.0, -2.0])
def test_quantize_zero_and_negative_scales(quantize_kernel, npu_device, scale):
    x = torch.tensor(
        [[-12.0, -3.5, -1.0, 0.0, 1.0, 3.5, 12.0]],
        device=npu_device,
        dtype=DTYPE,
    )
    assert_quantize_matches_ref(quantize_kernel, x, scale)


def test_quantize_rounding_halfway_values(quantize_kernel, npu_device):
    x = torch.tensor(
        [[-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]],
        device=npu_device,
        dtype=DTYPE,
    )
    assert_quantize_matches_ref(quantize_kernel, x, 1.0)


def test_quantize_saturates(quantize_kernel, npu_device):
    x = torch.tensor(
        [[-300.0, -129.2, -127.4, -0.6, 0.4, 126.8, 127.4, 300.0]],
        device=npu_device,
        dtype=DTYPE,
    )
    assert_quantize_matches_ref(quantize_kernel, x, 1.0)
