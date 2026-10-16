# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------

import functools
from typing import Callable

import torch
from pto_kernels import pto_tri_inv_col_sweep
import pytest
import numpy as np
import random

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


def _assert_close_to_inverse(
    actual: torch.Tensor, golden: torch.Tensor, atol: float, rtol: float
):
    """allclose, except that atol is relative to the largest entry of each
    inverse. The inverse of a random unit triangular matrix has entries up to
    ~4e3 at n = 128, so a fixed atol does not hold, while entries close to zero
    (from cancellation) make any elementwise rtol fail."""
    atol = atol * golden.abs().amax(dim=(-2, -1), keepdim=True)
    assert torch.all((actual - golden).abs() <= atol + rtol * golden.abs())


def np_triu_inv_cs(input_x, dtype: np.dtype = np.float16):
    """Return the matrix inverse of a 3d tensor where first dimension is batch dimension.
    Each batch dimension returns U_inv_n of input U_n, i.e., U_n U_inv_n = I_n.

    The algorithm is the column sweep's algorithm (vectorized) applied on each column of I, i.e.,
    Ax=e_j for j=0,1,...,(n-1).
    """
    output = np.zeros_like(input_x)
    batch_dim = input_x.shape[0]
    for reps in range(batch_dim):
        U_n = input_x[reps, :, :].copy()
        n = U_n.shape[1]
        U_inv_n = np.zeros_like(U_n)

        for j in range(n):

            b = np.zeros(n, dtype=dtype)
            b[j] = 1  # b = e_j

            x = np.zeros(n, dtype=dtype)
            for k in range(n - 1, -1, -1):
                x[k] = b[k]  # must be b[k] / U_n[k,k]

                if k > 0:
                    b[:k] -= U_n[:k, k] * x[k]
            U_inv_n[:, j] = x

        output[reps, :, :] = U_inv_n

    return output.astype(dtype)


def rand_np_tril(batch_size: int, n: int, dtype: np.dtype):
    "Returns a random unit lower triangular matrix of size n."
    A = np.random.rand(batch_size, n, n).astype(dtype)
    A = np.tril(A)
    for k in range(batch_size):
        np.fill_diagonal(A[k, :, :], 1.0)
    return A.astype(dtype)


def ones_np_tril(batch_size: int, n: int, dtype: np.dtype):
    "Returns an all-ones lower triangular matrix of size n."
    A = np.ones((batch_size, n, n)).astype(dtype)
    A = np.tril(A)
    return A.astype(dtype)


@pytest.mark.parametrize("batch_size", [2, 4, 40, 256])
@pytest.mark.parametrize("matrix_size", [16, 32, 64, 128])
@pytest.mark.parametrize("data_type", [np.float16, np.float32], ids=str)
@pytest.mark.parametrize(
    "mat_gen",
    (rand_np_tril, ones_np_tril),
)
def test_tri_inv_col_sweep(
    batch_size: int,
    matrix_size: int,
    data_type: np.dtype,
    mat_gen: callable,
):

    input_x_cpu = mat_gen(batch_size, matrix_size, data_type)
    expected_cpu = np_triu_inv_cs(input_x_cpu.transpose(0, 2, 1), data_type)

    # Convert input matrices from row-major order to column-major order
    input_x_cpu = input_x_cpu.transpose(0, 2, 1)
    input_x = torch.from_numpy(input_x_cpu).npu()
    expected = torch.from_numpy(expected_cpu).npu()

    torch.npu.synchronize()
    actual = pto_tri_inv_col_sweep(input_x)
    torch.npu.synchronize()
    # Transpose matrices back to row-major order
    actual = actual.transpose(2, 1)
    torch.npu.synchronize()

    assert actual.shape == expected.shape, "Output shape does not match expected shape."
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("batch_size", [1, 2, 4, 40, 256])
@pytest.mark.parametrize("matrix_size", [16, 32, 64, 128])
@pytest.mark.parametrize("data_type", [np.float32], ids=str)
@pytest.mark.parametrize(
    "mat_gen,atol,rtol",
    # Observed max |actual - golden| / max |golden| is up to ~2.2e-6 for float32
    # at n = 128 (40 seeds); see _assert_close_to_inverse.
    [(rand_np_tril, 4.1e-6, 1e-5), (ones_np_tril, 0, 0)],
)
def test_tri_inv_col_sweep_np_linalg_inv(
    batch_size: int,
    matrix_size: int,
    data_type: np.dtype,
    mat_gen: callable,
    atol: float,
    rtol: float,
):

    input_x_cpu = mat_gen(batch_size, matrix_size, data_type)
    golden_numpy_cpu = np.linalg.inv(input_x_cpu)

    # Convert input matrices from row-major order to column-major order
    input_x_cpu = input_x_cpu.transpose(0, 2, 1)
    input_x = torch.from_numpy(input_x_cpu).npu()
    golden_numpy_as_torch = torch.from_numpy(golden_numpy_cpu).npu()

    torch.npu.synchronize()
    actual = pto_tri_inv_col_sweep(input_x)
    torch.npu.synchronize()

    # rtol must be scaled w.r.t to the input size, see Higham's paper, Eq. (2.3)
    # https://nhigham.com/wp-content/uploads/2023/08/high89t.pdf
    scaled_rtol = min([0.05, 10 * (matrix_size + batch_size) * rtol])
    _assert_close_to_inverse(actual, golden_numpy_as_torch, atol, scaled_rtol)


@pytest.mark.parametrize("batch_size", [1, 4, 40])
@pytest.mark.parametrize("matrix_size", [16, 32, 64, 128])
@pytest.mark.parametrize("is_lower", [True, False])
@pytest.mark.parametrize(
    "data_type,mat_gen,atol,rtol",
    # Observed max |actual - golden| / max |golden| at n = 128 is up to ~2.2e-6
    # for float32 (40 seeds) and ~1.4e-2 for float16 (20 seeds, batch 40); see
    # _assert_close_to_inverse. The float16 inverse stays finite (max entry
    # ~2.5e3 at n = 128), it is only less accurate.
    [
        (np.float16, rand_np_tril, 2.5e-2, 1e-5),
        (np.float16, ones_np_tril, 0, 0),
        (np.float32, rand_np_tril, 4.1e-6, 1e-5),
        (np.float32, ones_np_tril, 0, 0),
    ],
    ids=lambda v: getattr(v, "__name__", None),
)
# pylint: disable=too-many-positional-arguments
def test_tri_inv_col_sweep_is_lower(
    batch_size: int,
    matrix_size: int,
    data_type: np.dtype,
    is_lower: bool,
    mat_gen: callable,
    atol: float,
    rtol: float,
):
    input_x_cpu = mat_gen(batch_size, matrix_size, data_type)
    if not is_lower:
        input_x_cpu = np.ascontiguousarray(input_x_cpu.transpose(0, 2, 1))
    golden_cpu = np.linalg.inv(input_x_cpu.astype(np.float64)).astype(data_type)

    input_x = torch.from_numpy(input_x_cpu).npu()
    golden = torch.from_numpy(golden_cpu).npu()

    actual = pto_tri_inv_col_sweep(input_x, is_lower=is_lower)
    torch.npu.synchronize()

    assert actual.shape == golden.shape
    scaled_rtol = min([0.05, 10 * (matrix_size + batch_size) * rtol])
    _assert_close_to_inverse(actual, golden, atol, scaled_rtol)


@pytest.mark.parametrize("matrix_size", [16, 128])
def test_tri_inv_col_sweep_default_is_lower(matrix_size: int):
    input_x = torch.from_numpy(rand_np_tril(4, matrix_size, np.float32)).npu()
    assert torch.equal(
        pto_tri_inv_col_sweep(input_x), pto_tri_inv_col_sweep(input_x, is_lower=True)
    )


# ---------------------------------------------------------------------------
# Tests replicated from test_tri_inv_rec_unroll.py (BSND tests excluded).
#
# The generators below return strictly upper triangular A of shape
# (block_dim_x, block_dim_y, n, n); the kernel inverts the unit triangular
# matrix I + A (transposed for the lower case).
# ---------------------------------------------------------------------------


def random_tri_matrix(n, block_dim_x, block_dim_y, scale=0.1):
    return scale * torch.triu(torch.rand((block_dim_x, block_dim_y, n, n)), diagonal=1)


def ones_tri_matrix(n, block_dim_x, block_dim_y):
    return torch.triu(torch.ones((block_dim_x, block_dim_y, n, n)), diagonal=1)


def block_ones_triu_matrix(n, block_dim_x, block_dim_y):
    U_ = np.ones((16, 16))
    U = np.zeros((block_dim_x, block_dim_y, n, n))
    for x in range(block_dim_x):
        for y in range(block_dim_y):
            for i in range(0, n, 16):
                U[x, y, i : i + 16, i : i + 16] = U_
    return torch.from_numpy(np.triu(U, 1))


def block_random_triu_matrix(n, block_dim_x, block_dim_y, scale=0.1):
    U = np.zeros((block_dim_x, block_dim_y, n, n))
    for x in range(block_dim_x):
        for y in range(block_dim_y):
            for i in range(0, n, 16):
                U[x, y, i : i + 16, i : i + 16] = np.triu(
                    scale * np.random.rand(16, 16), k=1
                )
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


def max_condition_number(A: torch.Tensor) -> float:
    """Maximum condition number of I + A over the last two dimensions of A."""
    n = A.shape[-1]
    M = torch.eye(n, dtype=torch.float64) + A.reshape(-1, n, n).double()
    return float(torch.linalg.cond(M).max())


@functools.lru_cache(maxsize=None)
def scale_for_cond(n: int, cond_target: float) -> float:
    """Scale that puts cond(I + scale * A) at `cond_target`, for a strictly
    triangular A of size n with entries drawn uniformly from (-1, 1)."""
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


def conditioned_tri_matrix(n, block_dim_x, block_dim_y, cond_target=2.0):
    """Strictly upper triangular A, scaled so cond(I + A) is near `cond_target`."""
    base = torch.triu(torch.rand((block_dim_x, block_dim_y, n, n)) * 2 - 1, diagonal=1)
    return scale_for_cond(n, cond_target) * base


def _to_unit_tri(A: torch.Tensor, is_lower: bool, input_dtype: torch.dtype):
    """Returns the unit triangular I + A (transposed if lower) in input_dtype."""
    if is_lower:
        A = A.transpose(-1, -2)
    A = A.contiguous().to(input_dtype)
    eye = torch.eye(A.shape[-1], dtype=input_dtype)
    return A, (A + eye).contiguous()


def _test_tri_inv_col_sweep(
    A: torch.Tensor,
    atol: float,
    rtol: float,
    ftol: float,
    is_lower: bool,
    input_dtype: torch.dtype = torch.float16,
):
    A, M = _to_unit_tri(A, is_lower, input_dtype)
    golden_cpu = torch.linalg.inv(M.double())

    actual = pto_tri_inv_col_sweep(M.npu(), is_lower=is_lower)
    torch.npu.synchronize()
    actual_cpu = actual.cpu().to(torch.float64)

    frob_error = torch.sqrt(
        torch.sum((golden_cpu - actual_cpu) ** 2) / torch.sum(golden_cpu**2)
    )
    actual_numpy = actual_cpu.numpy()
    golden_numpy = golden_cpu.numpy()

    assert np.allclose(actual_numpy, golden_numpy, atol=atol, rtol=rtol), (
        f"Error at allclose - tensor shape: {A.shape} - rtol: {rtol} - "
        f"cond(I+A): {max_condition_number(A):.2f}."
        f" max abs error: {np.max(np.abs(actual_numpy - golden_numpy))}."
        f" frob error: {frob_error}."
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
        (block_random_triu_matrix, 5e-5, 0.1, 1e-4, torch.float16),
        # Observed at n = 128: max abs error ~1.8e-4 and frob error ~4.0e-4.
        (random_tri_matrix, 3.4e-4, 0.1, 6.7e-4, torch.float16),
        # float32 tests
        (block_ones_triu_matrix, 0, 0, 0, torch.float32),
        (ones_tri_matrix, 0, 0, 0, torch.float32),
        (block_random_triu_matrix, 1e-6, 1e-3, 1e-6, torch.float32),
        (random_tri_matrix, 1e-6, 1e-3, 1e-6, torch.float32),
    ],
)
# pylint: disable=too-many-positional-arguments
def test_tri_inv_col_sweep_rec_unroll_cases(
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
    _test_tri_inv_col_sweep(U, atol, rtol, ftol, is_lower, input_dtype)


@pytest.mark.parametrize("n", [16, 32, 64, 128])
@pytest.mark.parametrize("block_dim_x", [1, 3])
@pytest.mark.parametrize("block_dim_y", [4])
@pytest.mark.parametrize("is_lower", [False, True])
@pytest.mark.parametrize(
    "cond_target,ftol,input_dtype",
    [
        # float16: observed max(atol, frob) at n = 128 is ~1.1e-4, 4.1e-4,
        # 8.6e-4 and 2.4e-3 for the four condition numbers; ~1.6x margin below.
        (1.2, 2.3e-4, torch.float16),
        (2.0, 6.7e-4, torch.float16),
        (4.0, 1.5e-3, torch.float16),
        (8.0, 3.9e-3, torch.float16),
        (1.2, 1e-6, torch.float32),
        (2.0, 1e-6, torch.float32),
        (4.0, 2e-6, torch.float32),
        (8.0, 4e-6, torch.float32),
    ],
)
# pylint: disable=too-many-positional-arguments
def test_tri_inv_col_sweep_conditioning(
    n: int,
    block_dim_x: int,
    block_dim_y: int,
    cond_target: float,
    ftol: float,
    is_lower: bool,
    input_dtype: torch.dtype,
):
    """Accuracy as a function of how hard the matrix is to invert."""
    A = conditioned_tri_matrix(n, block_dim_x, block_dim_y, cond_target=cond_target)
    _test_tri_inv_col_sweep(
        A, atol=ftol, rtol=0.1, ftol=ftol, is_lower=is_lower, input_dtype=input_dtype
    )


@pytest.mark.parametrize("n", [16, 32, 64, 128])
@pytest.mark.parametrize("scale", [0.1, 0.5, 1.0])
@pytest.mark.parametrize("is_lower", [False, True])
@pytest.mark.parametrize("input_dtype", [torch.float16, torch.float32])
def test_tri_inv_col_sweep_dynamic_range(
    n: int, scale: float, is_lower: bool, input_dtype: torch.dtype
):
    """Dense same-sign input up to 1.0 must produce finite output."""
    _, M = _to_unit_tri(scale * ones_tri_matrix(n, 2, 4), is_lower, input_dtype)
    actual = pto_tri_inv_col_sweep(M.npu(), is_lower=is_lower)
    torch.npu.synchronize()
    finite = torch.isfinite(actual.cpu().to(torch.float64))
    assert finite.all(), (
        f"non-finite output for dense same-sign entries of {scale} at n={n}, "
        f"{input_dtype}: {(~finite).sum().item()} of {finite.numel()} elements."
    )


@pytest.mark.parametrize("n", [32, 64, 128])
@pytest.mark.parametrize("input_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize(
    "matrix_gen", [deltanet_rand_triu_matrix, deltanet_randn_triu_matrix]
)
def test_tri_inv_col_sweep_deltanet(n: int, input_dtype: torch.dtype, matrix_gen):
    A = matrix_gen(n, block_dim_x=7, block_dim_y=13)
    this_eps = torch.finfo(input_dtype).eps
    # Unlike test_tri_inv_rec_unroll.py, which uses ftol = 2 * eps / 10 (scaled
    # by the doubling block size / 16), the column sweep applies n sequential
    # rank-1 updates, so rounding errors accumulate with the matrix size. The
    # observed relative Frobenius error is ~0.14 * eps * sqrt(n) for both
    # float16 and float32; ftol = eps * sqrt(n) / 8 leaves a ~1.7x margin.
    _test_tri_inv_col_sweep(
        A,
        atol=this_eps,
        rtol=0.1,
        ftol=this_eps * np.sqrt(n) / 8,
        is_lower=False,
        input_dtype=input_dtype,
    )
