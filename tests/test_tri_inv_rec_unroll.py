# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import torch
from pto_kernels import pto_tri_inv_rec_unroll
import pytest
import numpy as np
import os
import random
from typing import Callable

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

NPU_DEVICE = os.environ.get("NPU_DEVICE", "npu:1")
torch.npu.config.allow_internal_format = False
torch.npu.set_device(NPU_DEVICE)


def random_triu_matrix(n, block_dim, scale=0.1):
    U = scale * torch.triu(torch.rand((block_dim, n, n)), diagonal=1)
    return U


def ones_triu_matrix(n, block_dim):
    U = torch.triu(torch.ones((block_dim, n, n)), diagonal=1)
    return U


def block_ones_triu_matrix(n, block_dim):
    U_ = np.ones((16, 16))
    n_blocks = n // 16
    U = np.zeros((block_dim, n, n))
    for k in range(block_dim):
        for i in range(n_blocks):
            start = i * 16
            end = i * 16 + 16
            U[k, start:end, start:end] = U_
    return torch.from_numpy(np.triu(U, 1))


def block_random_triu_matrix(n, block_dim, scale=0.2):
    U_ = scale * np.random.rand(16, 16)
    U_ = np.triu(U_, k=1)
    U = np.zeros((block_dim, n, n))
    for k in range(block_dim):
        for i in range(0, n, 16):
            U[k, i : i + 16, i : i + 16] = U_.copy()
    return torch.from_numpy(U)


def _test_tri_inv_trick(U: torch.tensor, atol: float, rtol: float, ftol: float):

    n = U.shape[-1]
    U = U.to(torch.half)
    U_npu = U.to(NPU_DEVICE)
    torch.npu.synchronize()

    assert U.stride() == (n * n, n, 1)
    Identity = np.ones((U.shape[0], n, n))
    Identity = np.triu(Identity)
    Identity = np.tril(Identity)
    U_plus_I = Identity.astype(np.float64) + U.numpy().astype(np.float64)
    golden_numpy = np.zeros((U_plus_I.shape))
    for k in range(U_plus_I.shape[0]):
        golden_numpy[k] = np.linalg.inv(U_plus_I[k])
    golden_cpu = torch.from_numpy(golden_numpy)

    torch.npu.synchronize()
    actual = pto_tri_inv_rec_unroll(U_npu)
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
    assert np.allclose(
        actual_numpy, golden_numpy, atol=atol, rtol=rtol
    ), f"Error at allclose - tensor shape: {U.shape} - rtol: {rtol}."
    assert frob_error <= ftol, f"frob_error: {frob_error}"


@pytest.mark.parametrize("n", [16, 32, 64, 128])
@pytest.mark.parametrize("block_dim", [1, 2, 3, 4, 5, 8, 11, 19, 57, 128, 256])
@pytest.mark.parametrize(
    "matrix_gen,atol,rtol,ftol",
    [
        (block_ones_triu_matrix, 0, 0, 0),
        (ones_triu_matrix, 0, 0, 0),
        (block_random_triu_matrix, 5e-5, 0.1, 1e-4),
        (random_triu_matrix, 5e-5, 0.1, 1e-4),
    ],
)
def test_tri_inv_trick_ones(
    n: int, block_dim: int, matrix_gen: Callable, atol: float, rtol: float, ftol: float
):
    U = matrix_gen(n, block_dim)
    _test_tri_inv_trick(U, atol, rtol, ftol)
