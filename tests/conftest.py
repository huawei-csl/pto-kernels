from pathlib import Path
import os
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

try:
    import torch
except ImportError:
    torch = None


NPU_DEVICE = os.environ.get("NPU_DEVICE", "npu:1")


def pytest_addoption(parser):
    parser.addoption(
        "--run-npu",
        action="store_true",
        default=False,
        help="Run tests marked with @pytest.mark.npu.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "npu: tests that require torch-npu, an available NPU, and compiled PTO custom ops",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-npu"):
        return
    skip_npu = pytest.mark.skip(reason="requires --run-npu")
    for item in items:
        if "npu" in item.keywords:
            item.add_marker(skip_npu)


def pytest_runtest_setup(item):
    if "npu" not in item.keywords:
        return
    if torch is None or not hasattr(torch, "npu"):
        pytest.skip("torch-npu is not available")
    torch.npu.config.allow_internal_format = False
    torch.npu.set_device(NPU_DEVICE)


@pytest.fixture(scope="session")
def npu_device():
    return NPU_DEVICE
