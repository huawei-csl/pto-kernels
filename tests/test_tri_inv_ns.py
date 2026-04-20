# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------

import math
import random
from typing import Callable

import numpy as np
import pytest
import torch

from pto_kernels import pto_tri_inv_ns

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


def random_triu_matrix(n, block_dim_x, block_dim_y, scale=0.1):
    U = scale * torch.triu(torch.rand((block_dim_x, block_dim_y, n, n)), diagonal=1)
    return U


def block_ones_matrix(n, block_dim_x, block_dim_y):
    U_ = np.ones((16, 16))
    n_blocks = n // 16
    U = np.zeros((block_dim_x, block_dim_y, n, n))
    for x in range(block_dim_x):
        for y in range(block_dim_y):
            for i in range(n_blocks):
                start = i * 16
                end = i * 16 + 16
                U[x, y, start:end, start:end] = U_
    return torch.from_numpy(np.triu(U, 1))


def ones_matrix(n, block_dim_x, block_dim_y):
    U = np.ones((block_dim_x, block_dim_y, n, n))
    return torch.from_numpy(np.triu(U, 1))


def zeros_matrix(n, block_dim_x, block_dim_y):
    return torch.zeros(block_dim_x, block_dim_y, n, n)


def block_random_matrix(n, block_dim_x, block_dim_y, scale=0.2):
    U_ = scale * np.random.rand(16, 16)
    U_ = np.triu(U_, k=1)
    U = np.zeros((block_dim_x, block_dim_y, n, n))
    for x in range(block_dim_x):
        for y in range(block_dim_y):
            for i in range(0, n, 16):
                U[x, y, i : i + 16, i : i + 16] = U_.copy()
    return torch.from_numpy(U)


def linalg_inv(U: torch.Tensor) -> torch.Tensor:
    n = U.shape[-1]
    identity = np.eye(n, dtype=np.double)
    golden_numpy = np.zeros(U.shape)
    for x in range(U.shape[0]):
        for y in range(U.shape[1]):
            golden_numpy[x, y] = np.linalg.inv(
                U[x, y].numpy().astype(np.double) + identity
            )
    return torch.from_numpy(golden_numpy)


def default_num_iters(n: int) -> int:
    return int(math.ceil(4.0 * math.log2(n)))


def _test_tri_inv_ns(
    U: torch.Tensor,
    atol: float,
    rtol: float,
    ftol: float,
):
    U = U.to(torch.half)
    golden_cpu = linalg_inv(U)

    U_npu = U.npu()

    torch.npu.synchronize()
    num_iters = int(4.0 * math.ceil(math.log2(U.shape[-1])))
    # num_iters = 1
    actual = pto_tri_inv_ns(U_npu, num_iters=num_iters)
    torch.npu.synchronize()

    actual_cpu = actual.cpu().to(torch.float64)

    frob_error = torch.sqrt(
        torch.sum((golden_cpu - actual_cpu) * (golden_cpu - actual_cpu))
        / torch.sum(golden_cpu * golden_cpu)
    )

    actual_numpy = actual_cpu.numpy()
    golden_numpy = golden_cpu.numpy()

    assert np.allclose(
        actual_numpy, golden_numpy, atol=atol, rtol=rtol
    ), f"Error at allclose - tensor shape: {U.shape} - rtol: {rtol}."
    assert frob_error <= ftol, f"frob_error: {frob_error}"


@pytest.mark.parametrize("n", [16, 32, 64, 96, 128])
@pytest.mark.parametrize("block_dim_x", [1, 3, 7, 16])
@pytest.mark.parametrize("block_dim_y", [1, 2, 4, 16])
# @pytest.mark.parametrize("n", [96])
# @pytest.mark.parametrize("block_dim_x", [16])
# @pytest.mark.parametrize("block_dim_y", [2])
@pytest.mark.parametrize(
    "matrix_gen,atol,rtol,ftol",
    [
        (zeros_matrix, 5e-5, 0.1, 1e-2),
        (ones_matrix, 5e-5, 0.1, 1e-2),
        (block_ones_matrix, 5e-5, 0.1, 1e-2),
        (block_random_matrix, 5e-5, 0.1, 1e-2),
        (random_triu_matrix, 5e-5, 0.1, 1e-2),
    ],
)
def test_tri_inv_ns(
    n: int,
    block_dim_x: int,
    block_dim_y: int,
    matrix_gen: Callable,
    atol: float,
    rtol: float,
    ftol: float,
):
    U = matrix_gen(n, block_dim_x, block_dim_y)
    _test_tri_inv_ns(U, atol, rtol, ftol)
