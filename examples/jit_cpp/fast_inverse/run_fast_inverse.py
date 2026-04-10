# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------

"""
Sanity-check tests for the JIT-compiled triangular inverse (recursive unroll)
kernel. Run from the fast_inverse/ directory:

    python run_fast_inverse.py

For more detailed test cases see also tests/test_tri_inv_rec_unroll.py
and tests/test_tri_inv_rec_unroll_variable_sequence_lengths.py
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
# Matrix generator (identical to the unit-test suite)
# ---------------------------------------------------------------------------
def random_triu_matrix(n, block_dim_x, block_dim_y, scale=0.1):
    return scale * torch.triu(
        torch.rand((block_dim_x, block_dim_y, n, n)),
        diagonal=1,
    )


# ---------------------------------------------------------------------------
# Reference implementation (CPU / numpy)
# ---------------------------------------------------------------------------
def invert_single_chunk_ref(U: torch.Tensor) -> torch.Tensor:
    """Invert one upper-triangular chunk U where U is (..., m, m)."""
    m = U.shape[-1]
    return torch.from_numpy(
        np.linalg.inv(U.numpy().astype(np.double) + np.eye(m, dtype=np.double))
    )


# ---------------------------------------------------------------------------
# Kernel helpers
# ---------------------------------------------------------------------------
def _make_minus_identity(matrix_size: int, device: str) -> torch.Tensor:
    I_neg = torch.zeros(matrix_size, matrix_size, dtype=torch.half, device=device)
    I_neg.fill_diagonal_(-1)
    return I_neg


def _count_varlen_chunks(
    cu_seqlens: torch.Tensor | list[int],
    chunk_size: int,
) -> int:
    if isinstance(cu_seqlens, torch.Tensor):
        cu_seqlens_list = [int(x) for x in cu_seqlens.detach().cpu().tolist()]
    else:
        cu_seqlens_list = [int(x) for x in cu_seqlens]
    return sum(
        (cu_seqlens_list[i + 1] - cu_seqlens_list[i] + chunk_size - 1) // chunk_size
        for i in range(len(cu_seqlens_list) - 1)
    )


def _run_kernel_bsnd(
    tri_inv_func,
    U_bsnd_fp16: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
):
    """
    Run the kernel in BSND mode and return fp64 CPU result.

    U_bsnd_fp16 : (B, S, N, D) half tensor on NPU where each (D, D) block
                  along the S dimension is one matrix to invert.
    cu_seqlens : optional int32 tensor containing cumulative sequence lengths
                 for varlen BSND inputs.
    """
    matrix_size = U_bsnd_fp16.shape[-1]
    num_bsnd_heads = U_bsnd_fp16.shape[-2]
    if cu_seqlens is not None:
        num_matrices = _count_varlen_chunks(cu_seqlens, matrix_size) * num_bsnd_heads
    else:
        num_matrices = U_bsnd_fp16.numel() // (matrix_size * matrix_size)
    device = U_bsnd_fp16.device

    tensor_out = torch.zeros_like(U_bsnd_fp16, dtype=torch.float32)
    I_neg = _make_minus_identity(matrix_size, str(device))
    if cu_seqlens is not None:
        cu_seqlens = cu_seqlens.to(device=device, dtype=torch.int32).contiguous()

    torch.npu.synchronize()
    tri_inv_func(
        tensor_out,
        U_bsnd_fp16,
        I_neg,
        matrix_size,
        num_matrices,
        num_bsnd_heads,
        cu_seqlens=cu_seqlens,
    )
    torch.npu.synchronize()

    return tensor_out.cpu().to(torch.float64)


def _build_varlen_bsnd_case(
    cu_seqlens: list[int],
    num_heads: int,
    chunk_size: int,
):
    """
    Build an unpadded BSND tensor plus reference output for varlen testing.

    Each sequence contributes only its true rows in the packed BSND tensor.
    """
    seq_lens = [cu_seqlens[i + 1] - cu_seqlens[i] for i in range(len(cu_seqlens) - 1)]
    print(
        f"    varlen sequence lengths: {seq_lens} "
        f"(chunk_size={chunk_size}, num_heads={num_heads})"
    )

    total_tokens = cu_seqlens[-1]
    num_chunks = sum(
        (cu_seqlens[i + 1] - cu_seqlens[i] + chunk_size - 1) // chunk_size
        for i in range(len(cu_seqlens) - 1)
    )
    chunk_mats = random_triu_matrix(chunk_size, num_chunks, num_heads).to(torch.float64)

    U = torch.zeros((1, total_tokens, num_heads, chunk_size), dtype=torch.float64)
    golden = torch.zeros((1, total_tokens, num_heads, chunk_size), dtype=torch.float64)

    chunk_idx = 0

    for seq_idx in range(len(cu_seqlens) - 1):
        seq_start = cu_seqlens[seq_idx]
        seq_end = cu_seqlens[seq_idx + 1]
        for chunk_start in range(seq_start, seq_end, chunk_size):
            actual_size = min(chunk_size, seq_end - chunk_start)
            chunk = chunk_mats[chunk_idx]
            for head_idx in range(num_heads):
                U_valid = chunk[head_idx, :actual_size, :actual_size]
                U[
                    0,
                    chunk_start : chunk_start + actual_size,
                    head_idx,
                    :actual_size,
                ] = U_valid
                golden[
                    0,
                    chunk_start : chunk_start + actual_size,
                    head_idx,
                    :actual_size,
                ] = invert_single_chunk_ref(U_valid)

            chunk_idx += 1

    return (
        U,
        golden,
        torch.tensor(cu_seqlens, dtype=torch.int32),
    )


# ---------------------------------------------------------------------------
# Single test – BSND varlen layout
# ---------------------------------------------------------------------------


def _test_case_bsnd_varlen(
    tri_inv_func,
    cu_seqlens: list[int],
    N: int,
    D: int,
    label: str,
):

    U_varlen, golden, cu_seqlens_tensor = _build_varlen_bsnd_case(
        cu_seqlens,
        N,
        D,
    )
    actual_varlen = _run_kernel_bsnd(
        tri_inv_func,
        U_varlen.to(torch.half).npu(),
        cu_seqlens=cu_seqlens_tensor.npu(),
    )
    actual = actual_varlen

    frob = torch.sqrt(torch.sum((golden - actual) ** 2) / torch.sum(golden**2)).item()

    atol = 5e-5
    rtol = 5e-2
    ftol = 1e-4
    assert np.allclose(
        actual.numpy(),
        golden.numpy(),
        atol=atol,
        rtol=rtol,
    ), f"[{label}] allclose failed - shape {actual.shape}, rtol={rtol}"
    assert frob <= ftol, f"[{label}] Frobenius error {frob:.2e} > {ftol:.2e}"

    print(f"  PASS  {label}  frob={frob:.2e}")


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------
def run_tests(tri_inv_func):

    total = passed = 0

    print("\n=== BSND varlen layout ===")
    varlen_configs = [
        (4, 16, [0, 15]),
        (4, 32, [0, 256, 500, 1000]),
        (4, 64, [0, 15, 100, 300, 1200, 2000]),
        (4, 16, [0, 1, 100, 300, 1200, 2048]),
        (4, 32, [0, 200, 512, 1200, 2048]),
    ]
    for N, D, cu_seqlens in varlen_configs:
        total += 1
        label = f"N={N} D={D} cu={cu_seqlens}"
        try:
            _test_case_bsnd_varlen(
                tri_inv_func,
                cu_seqlens,
                N,
                D,
                label,
            )
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
