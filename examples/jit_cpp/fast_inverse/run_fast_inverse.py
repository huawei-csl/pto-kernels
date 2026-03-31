# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------

"""
Correctness tests for the JIT-compiled triangular inverse (recursive unroll)
kernel.  Run from the fast_inverse/ directory:

    python run_fast_inverse.py
"""

import numpy as np
import torch
import torch_npu  # noqa: F401 – registers the NPU backend

from jit_util_fast_inverse import jit_compile

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Matrix generators  (identical to the unit-test suite)
# ---------------------------------------------------------------------------

def random_triu_matrix(n, block_dim_x, block_dim_y, scale=0.1):
    return scale * torch.triu(torch.rand((block_dim_x, block_dim_y, n, n)), diagonal=1)


def ones_triu_matrix(n, block_dim_x, block_dim_y):
    return torch.triu(torch.ones((block_dim_x, block_dim_y, n, n)), diagonal=1)


def block_ones_triu_matrix(n, block_dim_x, block_dim_y):
    U_ = np.ones((16, 16))
    n_blocks = n // 16
    U = np.zeros((block_dim_x, block_dim_y, n, n))
    for x in range(block_dim_x):
        for y in range(block_dim_y):
            for i in range(n_blocks):
                s, e = i * 16, i * 16 + 16
                U[x, y, s:e, s:e] = U_
    return torch.from_numpy(np.triu(U, 1))


def block_random_triu_matrix(n, block_dim_x, block_dim_y, scale=0.2):
    U_ = np.triu(scale * np.random.rand(16, 16), k=1)
    U = np.zeros((block_dim_x, block_dim_y, n, n))
    for x in range(block_dim_x):
        for y in range(block_dim_y):
            for i in range(0, n, 16):
                U[x, y, i : i + 16, i : i + 16] = U_.copy()
    return torch.from_numpy(U)


# ---------------------------------------------------------------------------
# Reference implementation  (CPU / numpy)
# ---------------------------------------------------------------------------

def linalg_inv_ref(U: torch.Tensor) -> torch.Tensor:
    """Invert (U + I) for each matrix in the batch using numpy."""
    n = U.shape[-1]
    identity = np.triu(np.tril(np.ones((n, n), dtype=np.double)))
    out = np.zeros(U.shape)
    for x in range(U.shape[0]):
        for y in range(U.shape[1]):
            out[x, y] = np.linalg.inv(U[x, y].numpy().astype(np.double) + identity)
    return torch.from_numpy(out)


# ---------------------------------------------------------------------------
# Kernel helpers
# ---------------------------------------------------------------------------

def _make_minus_identity(matrix_size: int, device: str) -> torch.Tensor:
    I_neg = torch.zeros(matrix_size, matrix_size, dtype=torch.half, device=device)
    I_neg.fill_diagonal_(-1)
    return I_neg


def _run_kernel(tri_inv_func, U_fp16: torch.Tensor):
    """
    Allocate output, build -I, run kernel, return fp64 CPU result.

    U_fp16 : (block_dim_x, block_dim_y, n, n) half tensor on NPU.
    """
    matrix_size = U_fp16.shape[-1]
    num_matrices = U_fp16.numel() // (matrix_size * matrix_size)
    device = U_fp16.device

    tensor_out = torch.zeros_like(U_fp16, dtype=torch.float32)
    I_neg = _make_minus_identity(matrix_size, str(device))

    torch.npu.synchronize()
    tri_inv_func(tensor_out, U_fp16, I_neg, matrix_size, num_matrices)
    torch.npu.synchronize()

    return tensor_out.cpu().to(torch.float64)


# ---------------------------------------------------------------------------
# Single test
# ---------------------------------------------------------------------------

def _test_case(tri_inv_func, U: torch.Tensor, atol: float, rtol: float, ftol: float,
               label: str):
    U_fp16 = U.to(torch.half)
    golden = linalg_inv_ref(U_fp16)

    actual = _run_kernel(tri_inv_func, U_fp16.npu())

    frob = torch.sqrt(
        torch.sum((golden - actual) ** 2) / torch.sum(golden ** 2)
    ).item()

    assert np.allclose(
        actual.numpy(), golden.numpy(), atol=atol, rtol=rtol
    ), f"[{label}] allclose failed — shape {U.shape}, rtol={rtol}"
    assert frob <= ftol, f"[{label}] Frobenius error {frob:.2e} > {ftol:.2e}"

    print(f"  PASS  {label}  frob={frob:.2e}")


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

def run_tests(tri_inv_func):
    cases = [
        ("block_ones",   block_ones_triu_matrix,   0,    0,    0),
        ("ones",         ones_triu_matrix,          0,    0,    0),
        ("block_random", block_random_triu_matrix,  5e-5, 0.1,  1e-4),
        ("random",       random_triu_matrix,        5e-5, 0.1,  1e-4),
    ]
    sizes   = [16, 32, 64, 128]
    x_dims  = [1, 2, 4]
    y_dims  = [2, 4]

    total = passed = 0
    for n in sizes:
        for bdx in x_dims:
            for bdy in y_dims:
                for name, gen, atol, rtol, ftol in cases:
                    total += 1
                    label = f"n={n} x={bdx} y={bdy} [{name}]"
                    try:
                        U = gen(n, bdx, bdy)
                        _test_case(tri_inv_func, U, atol, rtol, ftol, label)
                        passed += 1
                    except AssertionError as err:
                        print(f"  FAIL  {label}: {err}")

    print(f"\n{passed}/{total} tests passed.")
    return passed == total


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    torch.npu.set_device("npu:0")

    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fast_inverse.cpp")
    print(f"Compiling {src} ...")
    tri_inv_func = jit_compile(src)
    print("Compilation successful.\n")

    ok = run_tests(tri_inv_func)
    raise SystemExit(0 if ok else 1)
