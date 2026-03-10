import pytest
import os
import torch

NPU_DEVICE = os.environ.get("NPU_DEVICE", "npu:1")
torch.npu.config.allow_internal_format = False
torch.npu.set_device(NPU_DEVICE)


@pytest.fixture(scope="session")
def npu_device():
    return NPU_DEVICE
