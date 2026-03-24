# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------

import pytest
import torch
from pto_kernels import get_aic_cores, get_aiv_cores


@pytest.fixture(autouse=True)
def npu_device_context():
    """Ensure an NPU device context is active before each test."""
    torch.zeros(1).npu()


def test_get_aic_cores_default_device():
    num_cores = get_aic_cores()
    assert isinstance(num_cores, int)
    assert num_cores in [20, 24, 25]


def test_get_aiv_cores_default_device():
    num_cores = get_aiv_cores()
    assert isinstance(num_cores, int)
    assert num_cores in [40, 48, 50]


@pytest.mark.parametrize("device_id", [0])
def test_get_aic_cores_explicit_device(device_id: int):
    num_cores = get_aic_cores(device_id=device_id)
    assert isinstance(num_cores, int)
    assert num_cores in [20, 24, 25]


@pytest.mark.parametrize("device_id", [0])
def test_get_aiv_cores_explicit_device(device_id: int):
    num_cores = get_aiv_cores(device_id=device_id)
    assert isinstance(num_cores, int)
    assert num_cores in [40, 48, 50]
