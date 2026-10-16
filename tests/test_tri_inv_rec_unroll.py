# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------

import functools

import torch
import pytest
import numpy as np
import random
from typing import Callable
from pto_kernels import pto_tri_inv_rec_unroll

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


def random_tri_matrix(n, block_dim_x, block_dim_y, scale=0.1, is_lower=False):
    if is_lower:
        return scale * torch.tril(
            torch.rand((block_dim_x, block_dim_y, n, n)), diagonal=-1
        )
    else:
        return scale * torch.triu(
            torch.rand((block_dim_x, block_dim_y, n, n)), diagonal=1
        )


def ones_tri_matrix(n, block_dim_x, block_dim_y, is_lower=False):
    if is_lower:
        return torch.tril(torch.ones((block_dim_x, block_dim_y, n, n)), diagonal=-1)
    else:
        return torch.triu(torch.ones((block_dim_x, block_dim_y, n, n)), diagonal=1)


def block_ones_triu_matrix(n, block_dim_x, block_dim_y):
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


def block_random_triu_matrix(n, block_dim_x, block_dim_y, scale=0.1):
    U = np.zeros((block_dim_x, block_dim_y, n, n))
    for x in range(block_dim_x):
        for y in range(block_dim_y):
            for i in range(0, n, 16):
                U_ = np.triu(scale * np.random.rand(16, 16), k=1)
                U[x, y, i : i + 16, i : i + 16] = U_
    return torch.from_numpy(U)


def deltanet_randn_triu_matrix(n, block_dim_x, block_dim_y):
    K = np.random.randn(block_dim_x, block_dim_y, n, n).astype(np.float64)
    for i in range(block_dim_x):
        for j in range(block_dim_y):
            K[i, j] = K[i, j] / np.linalg.norm(K[i, j], axis=-1, keepdims=True)
            K[i, j] = np.triu(K[i, j] @ K[i, j].T, k=1)
    return torch.from_numpy(K)


def deltanet_rand_triu_matrix(n, block_dim_x, block_dim_y):
    K = 0.5 - np.random.rand(block_dim_x, block_dim_y, n, n).astype(np.float64)
    for i in range(block_dim_x):
        for j in range(block_dim_y):
            K[i, j] = K[i, j] / np.linalg.norm(K[i, j], axis=-1, keepdims=True)
            K[i, j] = np.triu(K[i, j] @ K[i, j].T, k=1)
    return torch.from_numpy(K)


def max_condition_number(A: torch.tensor) -> float:
    """Maximum condition number of I + A over all 2D matrices formed on the
    last two dimensions (-2, -1) of A."""
    n = A.shape[-1]
    M = torch.eye(n, dtype=torch.float64) + A.reshape(-1, n, n).double()
    return float(torch.linalg.cond(M).max())


@functools.lru_cache(maxsize=None)
def scale_for_cond(n: int, cond_target: float) -> float:
    """Scale that puts cond(I + scale * A) at `cond_target`, for a strictly
    triangular A of size n with entries drawn uniformly from (-1, 1).

    Bisected once per (n, cond_target) on a fixed reference draw: the scale
    barely varies between draws from the same distribution, and calibrating per
    matrix would cost an SVD per bisection step per matrix.
    """
    g = torch.Generator().manual_seed(0)
    a = torch.triu(torch.rand((n, n), generator=g, dtype=torch.float64) * 2 - 1, 1)
    eye = torch.eye(n, dtype=torch.float64)
    lo, hi = 0.0, 50.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if float(torch.linalg.cond(eye + mid * a)) < cond_target:
            lo = mid
        else:
            hi = mid
    return lo


def conditioned_tri_matrix(
    n, block_dim_x, block_dim_y, cond_target=2.0, is_lower=False
):
    """Strictly triangular A, scaled so cond(I + A) is near `cond_target`.

    How hard the inversion is depends on the scale of A, and at a fixed scale it
    depends on n as well: `0.1 * rand` gives cond(I + A) = 1.5 at n = 16 but 6.9
    at n = 128, so a single tolerance means a different demand at every size.
    Scaling to a target makes the difficulty an explicit parameter. The achieved
    value varies a little between draws; the assertions report it.

    Sign matters too. All-positive entries accumulate constructively through the
    powers of A; at n = 128 and the same scale, mixed signs halve the condition
    number. Real callers (GDN's `(I + tril(beta * K K^T))`) produce mixed signs
    and land near cond 1.1 - 1.3.
    """
    base = torch.rand((block_dim_x, block_dim_y, n, n)) * 2 - 1
    base = torch.tril(base, diagonal=-1) if is_lower else torch.triu(base, diagonal=1)
    return scale_for_cond(n, cond_target) * base


def linalg_inv(U: torch.tensor) -> torch.tensor:
    n = U.shape[-1]
    Identity = np.eye(n, dtype=np.double)
    golden_numpy = np.zeros((U.shape))
    for x in range(U.shape[0]):
        for y in range(U.shape[1]):
            golden_numpy[x, y] = np.linalg.inv(
                U[x, y].double().numpy().astype(np.double) + Identity
            )
    return torch.from_numpy(golden_numpy)


def _test_tri_inv_rec_unroll(
    A: torch.tensor,
    atol: float,
    rtol: float,
    ftol: float,
    is_lower: bool,
    input_dtype: torch.dtype = torch.float16,
    max_doubling_block_size: int = 16,
):

    # Make sure A is lower triangular and contiguous in memory.
    if is_lower:
        A = A.transpose(-1, -2).contiguous().to(input_dtype)
    else:
        A = A.contiguous().to(input_dtype)

    golden_cpu = linalg_inv(A)

    A_npu = A.npu()

    torch.npu.synchronize()
    actual = pto_tri_inv_rec_unroll(
        A_npu,
        is_bsnd_format=False,
        is_lower=is_lower,
        max_doubling_block_size=max_doubling_block_size,
    )
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

    assert np.allclose(actual_numpy, golden_numpy, atol=atol, rtol=rtol), (
        f"Error at allclose - tensor shape: {A.shape} - rtol: {rtol} - "
        f"cond(I+A): {max_condition_number(A):.2f}."
        f" max abs error: {np.max(np.abs(actual_numpy-golden_numpy))}."
        f" frob error: {frob_error}."
    )
    assert (
        frob_error <= ftol
    ), f"frob_error: {frob_error} (ftol {ftol}, cond(I+A) {max_condition_number(A):.2f})"


# pylint: disable=too-many-function-args,too-many-positional-arguments
def _test_tri_inv_rec_unroll_bsnd(
    A: torch.tensor,
    B: int,
    S: int,
    N: int,
    D: int,
    atol: float,
    rtol: float,
    ftol: float,
    is_lower: bool,
    input_dtype: torch.dtype = torch.float16,
):

    # Make sure U is lower triangular and contiguous in memory.
    if is_lower:
        A = A.transpose(-1, -2).contiguous().to(input_dtype)
    else:
        A = A.contiguous().to(input_dtype)

    golden_cpu = linalg_inv(A)

    # Transform to bsnd layout
    A_bsnd = A.transpose(1, 2).contiguous().reshape(B, S, N, D)
    golden_cpu = golden_cpu.transpose(1, 2).contiguous().reshape(B, S, N, D)
    torch.npu.synchronize()

    A_bsnd_npu = A_bsnd.npu()

    torch.npu.synchronize()
    actual = pto_tri_inv_rec_unroll(
        A_bsnd_npu, is_bsnd_format=True, is_lower=is_lower, max_doubling_block_size=16
    )
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

    assert np.allclose(actual_numpy, golden_numpy, atol=atol, rtol=rtol), (
        f"Error at allclose - tensor shape: {A.shape} - rtol: {rtol} - "
        f"cond(I+A): {max_condition_number(A):.2f}."
    )
    assert (
        frob_error <= ftol
    ), f"frob_error: {frob_error} (ftol {ftol}, cond(I+A) {max_condition_number(A):.2f})"


@pytest.mark.parametrize("n", [16, 32, 64, 128])
@pytest.mark.parametrize("block_dim_x", [1, 2, 3, 4])
@pytest.mark.parametrize("block_dim_y", [2, 4, 8])
@pytest.mark.parametrize("is_lower", [False, True])
@pytest.mark.parametrize(
    "matrix_gen,atol,rtol,ftol,input_dtype",
    [
        # float16 tests
        (block_ones_triu_matrix, 0, 0, 0, torch.float16),
        (ones_tri_matrix, 0, 0, 0, torch.float16),
        (
            block_random_triu_matrix,
            5e-5,
            0.1,
            1e-4,
            torch.float16,
        ),
        (random_tri_matrix, 5e-5, 0.1, 1e-4, torch.float16),
        # bfloat16 tests (block-ones and all-ones tests fail in bfloat16 due to overflow)
        # (block_ones_triu_matrix, 0, 0, 0, torch.bfloat16),
        # (ones_tri_matrix, 0, 0, 0, torch.bfloat16),
        (
            block_random_triu_matrix,
            5e-4,
            0.1,
            1e-3,
            torch.bfloat16,
        ),
        (random_tri_matrix, 5e-4, 0.1, 1e-3, torch.bfloat16),
    ],
)
def test_tri_inv_rec_unroll(
    n: int,
    block_dim_x: int,
    block_dim_y: int,
    matrix_gen: Callable,
    atol: float,
    rtol: float,
    ftol: float,
    is_lower: bool,
    input_dtype: torch.dtype,
):
    U = matrix_gen(n, block_dim_x, block_dim_y)
    _test_tri_inv_rec_unroll(U, atol, rtol, ftol, is_lower, input_dtype)


@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("S", [128, 256, 1024])
@pytest.mark.parametrize("N", [4, 8])
@pytest.mark.parametrize("C", [16, 32, 64, 128])
@pytest.mark.parametrize("is_lower", [False, True])
@pytest.mark.parametrize(
    "matrix_gen,atol,rtol,ftol,input_dtype",
    [
        # float 16 tests
        (block_ones_triu_matrix, 0, 0, 0, torch.float16),
        (ones_tri_matrix, 0, 0, 0, torch.float16),
        (random_tri_matrix, 5e-5, 0.1, 1e-4, torch.float16),
        (ones_tri_matrix, 0, 0, 0, torch.float16),
        (
            block_random_triu_matrix,
            5e-4,
            0.1,
            1e-4,
            torch.float16,
        ),
        # bfloat16 tests (block-ones and all-ones tests fail in bfloat16 due to overflow)
        # (block_ones_triu_matrix, 0, 0, 0, torch.bfloat16),
        # (ones_tri_matrix, 0, 0, 0, torch.bfloat16),
        (random_tri_matrix, 5e-4, 0.1, 1e-3, torch.bfloat16),
        (
            block_random_triu_matrix,
            5e-4,
            0.1,
            1e-3,
            torch.bfloat16,
        ),
    ],
)
# pylint: disable=too-many-positional-arguments
def test_tri_inv_rec_unroll_bsnd(
    B: int,
    S: int,
    N: int,
    C: int,
    matrix_gen: Callable,
    atol: float,
    rtol: float,
    ftol: float,
    is_lower: bool,
    input_dtype: torch.dtype,
):
    # only test cases where the sequence length is a multiple of the chunk size are accepted
    if S % C != 0:
        pytest.skip("Sequence length must be a multiple of chunk size C.")
    U = matrix_gen(C, B * S // C, N)
    _test_tri_inv_rec_unroll_bsnd(
        U, B, S, N, C, atol, rtol, ftol, is_lower, input_dtype
    )


# Relative Frobenius error against the condition number of I + A, worst over
# n = 16 .. 128 on Ascend910B4. The error tracks the band rather than n, which
# is the point of scaling to a target: at a fixed scale the difficulty depends
# on n instead (`0.1 * rand` is cond 1.5 at n = 16 and 6.9 at n = 128), so one
# tolerance means a different demand at every size.
#
#   cond(I+A)    float16    bfloat16
#         1.2   2.0e-05     1.6e-04
#         2.0   7.6e-05     6.1e-04
#         4.0   1.5e-04     1.2e-03
#         8.0   2.3e-04     1.8e-03
#
# The tolerances below are those measurements with ~3x headroom for the draw.
# cond 1.2 is the band real callers land in: GDN's (I + tril(beta * K K^T))
# measures 1.10 - 1.24 on pipeline data. cond 8 is a stress case, not a
# workload -- it is here to be documented, not to gate a kernel on.
@pytest.mark.parametrize("n", [16, 32, 64, 128])
@pytest.mark.parametrize("block_dim_x", [1, 3])
@pytest.mark.parametrize("block_dim_y", [4])
@pytest.mark.parametrize("is_lower", [False, True])
@pytest.mark.parametrize(
    "cond_target,ftol,input_dtype",
    [
        (1.2, 5e-5, torch.float16),
        (2.0, 2e-4, torch.float16),
        (4.0, 4e-4, torch.float16),
        (8.0, 6e-4, torch.float16),
        (1.2, 4e-4, torch.bfloat16),
        (2.0, 1.5e-3, torch.bfloat16),
        (4.0, 3e-3, torch.bfloat16),
        (8.0, 5e-3, torch.bfloat16),
    ],
)
# pylint: disable=too-many-positional-arguments
def test_tri_inv_rec_unroll_conditioning(
    n: int,
    block_dim_x: int,
    block_dim_y: int,
    cond_target: float,
    ftol: float,
    is_lower: bool,
    input_dtype: torch.dtype,
):
    """Accuracy as a function of how hard the matrix is to invert."""
    # generators in this file produce upper triangular matrices;
    # _test_tri_inv_rec_unroll transposes them for the lower case
    A = conditioned_tri_matrix(n, block_dim_x, block_dim_y, cond_target=cond_target)
    _test_tri_inv_rec_unroll(
        A, atol=ftol, rtol=0.1, ftol=ftol, is_lower=is_lower, input_dtype=input_dtype
    )


# The kernel's input-range contract, stated as a test rather than left implicit
# in what the other cases happen to use. Phase 1 forms the powers A^(2^j) of
# each doubling block and holds them in the input dtype, so an input that is
# scaled too large comes back as NaN rather than as a large error. The powers
# grow fastest for dense, same-sign matrices, which is what ones_tri_matrix is;
# at the default doubling block entries of 1.0 peak at 3.4e3, inside fp16, and
# entries of 1.5 do not.
#
# This is the test that fails first, and says why, if TRI_INV_DOUBLING_BLOCK is
# raised past what the input range allows.
@pytest.mark.parametrize("n", [16, 32, 64, 128])
@pytest.mark.parametrize("scale", [0.1, 0.5, 1.0])
@pytest.mark.parametrize("is_lower", [False, True])
@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16])
def test_tri_inv_rec_unroll_dynamic_range(
    n: int, scale: float, is_lower: bool, input_dtype: torch.dtype
):
    """Dense same-sign input up to 1.0 must produce finite output."""
    A = scale * ones_tri_matrix(n, 2, 4)
    A = A.transpose(-1, -2).contiguous() if is_lower else A.contiguous()
    A = A.to(input_dtype)
    actual = pto_tri_inv_rec_unroll(
        A.npu(), is_bsnd_format=False, is_lower=is_lower, max_doubling_block_size=16
    )
    torch.npu.synchronize()
    finite = torch.isfinite(actual.cpu().to(torch.float64))
    assert finite.all(), (
        f"non-finite output for dense same-sign entries of {scale} at n={n}, "
        f"{input_dtype}: {(~finite).sum().item()} of {finite.numel()} elements. "
        "Phase 1 holds the powers A^(2^j) of each doubling block in the input "
        "dtype, so raising TRI_INV_DOUBLING_BLOCK narrows the input range the "
        "kernel supports; see run_tri_inv_rec_unroll's docstring."
    )


@pytest.mark.parametrize("n", [32, 64, 128])
@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("doubling_block_size", [32, 64, 128])
@pytest.mark.parametrize(
    "matrix_gen", [deltanet_rand_triu_matrix, deltanet_randn_triu_matrix]
)
def test_tri_inv_rec_unroll_varying_doubling_block_size(
    n: int, input_dtype: torch.dtype, doubling_block_size: int, matrix_gen
):
    A = matrix_gen(n, block_dim_x=7, block_dim_y=13)
    A = A.to(input_dtype)
    this_eps = torch.finfo(input_dtype).eps
    rtol = 0.1 * (doubling_block_size / 16)
    atol = (this_eps / 10) * (doubling_block_size / 16)
    ftol = (2 * this_eps / 10) * (doubling_block_size / 16)
    _test_tri_inv_rec_unroll(
        A,
        atol=atol,
        rtol=rtol,
        ftol=ftol,
        is_lower=False,
        input_dtype=input_dtype,
        max_doubling_block_size=doubling_block_size,
    )
