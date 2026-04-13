# pylint: disable=wrong-import-position
import sys
from pathlib import Path

import pytest
import torch

FAST_HADAMARD_DIR = Path(__file__).resolve().parent / "fast_hadamard"
if str(FAST_HADAMARD_DIR) not in sys.path:
    sys.path.insert(0, str(FAST_HADAMARD_DIR))

from jit_util_common import normalize_npu_device  # noqa: E402


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
