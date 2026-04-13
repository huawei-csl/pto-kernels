from pathlib import Path

import pytest
import torch

from jit_util_common import normalize_npu_device
from standard.jit_util_hadamard import jit_compile


def pytest_addoption(parser):
    parser.addoption(
        "--npu",
        action="store",
        default="npu:0",
        help="NPU device (examples: 0, npu:0, '0', 'npu:0').",
    )


@pytest.fixture(scope="session")
def npu_device(request):
    raw = request.config.getoption("--npu")
    return normalize_npu_device(raw)


@pytest.fixture(scope="session", autouse=True)
def setup_npu_device(npu_device):
    torch.npu.set_device(npu_device)


@pytest.fixture(scope="session")
def hadamard_kernel(npu_device):
    base = Path(__file__).resolve().parent
    src = base / "standard" / "fast_hadamard.cpp"
    return jit_compile(str(src), verbose=True, device=npu_device)