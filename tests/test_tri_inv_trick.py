# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------

import torch
from pto_kernels import pto_tri_inv_trick
import pytest
import numpy as np
import os
import random
from typing import Callable

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

NPU_DEVICE = os.environ.get("NPU_DEVICE", "npu:0")
torch.npu.config.allow_internal_format = False
torch.npu.set_device(NPU_DEVICE)


def random_matrix(n, block_dim_x, block_dim_y, scale=0.01):
    U = scale * torch.rand((block_dim_x, block_dim_y, n, n))
    return U


def block_ones_matrix(n, block_dim_x, block_dim_y):
    U_ = np.ones((16, 16))
    n_blocks = n // 16
    U = np.zeros((block_dim_x, block_dim_y, n, n))
    for k in range(block_dim_x):
        for l in range(block_dim_y):
            for i in range(n_blocks):
                start = i * 16
                end = i * 16 + 16
                U[k, l, start:end, start:end] = U_
    return torch.from_numpy(np.triu(U, 1))


def block_random_matrix(n, block_dim_x, block_dim_y, scale=0.2):
    U_ = scale * np.random.rand(16, 16)
    U_ = np.triu(U_, k=1)
    U = np.zeros((block_dim_x, block_dim_y, n, n))
    for k in range(block_dim_x):
        for l in range(block_dim_y):
            for i in range(0, n, 16):
                U[k, l, i : i + 16, i : i + 16] = U_.copy()
    return torch.from_numpy(U)


def _test_tri_inv_trick(U: torch.tensor, atol: float, rtol: float, ftol: float):

    n = U.shape[-1]
    U = U.to(torch.half)
    U_npu = U.to(NPU_DEVICE)
    torch.npu.synchronize()

    Identity = np.ones((n, n), dtype=np.double)
    Identity = np.triu(Identity)
    Identity = np.tril(Identity)
    golden_numpy = np.zeros((U.shape))
    for k in range(U.shape[0]):
        for l in range(U.shape[1]):
            golden_numpy[k, l] = np.linalg.inv(
                U[k, l].numpy().astype(np.double) + Identity
            )
    golden_cpu = torch.from_numpy(golden_numpy)

    torch.npu.synchronize()
    actual = pto_tri_inv_trick(U_npu)
    torch.npu.synchronize()
    actual_cpu = actual.cpu()
    torch.npu.synchronize()
    actual_cpu = actual_cpu.to(torch.float64)
    frob_error = torch.sqrt(
        torch.sum((golden_cpu - actual_cpu) * (golden_cpu - actual_cpu))
        / torch.sum(golden_cpu * golden_cpu)
    )
    actual_numpy = actual_cpu.numpy()
    golden_numpy = golden_cpu.numpy()
    # print(golden_cpu.shape)
    # print(actual_cpu.shape)

    assert np.allclose(
        actual_numpy, golden_numpy, atol=atol, rtol=rtol
    ), f"Error at allclose - tensor shape: {U.shape} - rtol: {rtol}."
    assert frob_error <= ftol, f"frob_error: {frob_error}"


@pytest.mark.parametrize("n", [16, 32, 64, 96, 128])
@pytest.mark.parametrize("block_dim_x", [1, 3, 7, 16])
@pytest.mark.parametrize("block_dim_y", [1, 2, 4, 16])
@pytest.mark.parametrize(
    "matrix_gen,atol,rtol,ftol",
    [
        (block_ones_matrix, 0, 0, 0),
        (block_random_matrix, 5e-5, 0.1, 1e-4),
    ],
)
def test_tri_inv_trick_ones(
    n: int,
    block_dim_x: int,
    block_dim_y: int,
    matrix_gen: Callable,
    atol: float,
    rtol: float,
    ftol: float,
):
    U = matrix_gen(n, block_dim_x, block_dim_y)
    _test_tri_inv_trick(U, atol, rtol, ftol)
