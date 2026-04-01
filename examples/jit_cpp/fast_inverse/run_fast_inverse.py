# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------

"""
Correctness tests for the JIT-compiled triangular inverse (recursive unroll)
kernel. Run from the fast_inverse/ directory:

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
# Matrix generators (identical to the unit-test suite)
# ---------------------------------------------------------------------------
def random_triu_matrix(n, block_dim_x, block_dim_y, scale=0.1):
    return scale * torch.triu(
        torch.rand((block_dim_x, block_dim_y, n, n)),
        diagonal=1,
    )


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
# Reference implementation (CPU / numpy)
# ---------------------------------------------------------------------------
def linalg_inv_ref(U: torch.Tensor) -> torch.Tensor:
    """Invert (U + I) for each matrix in the batch using numpy."""
    n = U.shape[-1]
    identity = np.eye(n, dtype=np.double)
    out = np.zeros(U.shape, dtype=np.double)
    for x in range(U.shape[0]):
        for y in range(U.shape[1]):
            out[x, y] = np.linalg.inv(U[x, y].numpy().astype(np.double) + identity)
    return torch.from_numpy(out)


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


def _chunk_metadata_from_cu_seqlens(
    cu_seqlens: torch.Tensor | list[int],
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(cu_seqlens, torch.Tensor):
        cu_seqlens_np = cu_seqlens.detach().cpu().numpy().astype(np.int64, copy=False)
    else:
        cu_seqlens_np = np.asarray(cu_seqlens, dtype=np.int64)

    seq_starts = cu_seqlens_np[:-1]
    seq_lens = cu_seqlens_np[1:] - seq_starts
    seq_num_chunks = (seq_lens + chunk_size - 1) // chunk_size
    total_chunks = int(seq_num_chunks.sum())

    chunk_indices = np.empty(total_chunks, dtype=np.int32)
    chunk_valid_sizes = np.empty(total_chunks, dtype=np.int32)
    cursor = 0
    for seq_start, seq_len, num_chunks in zip(seq_starts, seq_lens, seq_num_chunks):
        num_chunks_int = int(num_chunks)
        local_offsets = np.arange(num_chunks_int, dtype=np.int64) * chunk_size
        next_cursor = cursor + num_chunks_int
        chunk_indices[cursor:next_cursor] = (seq_start + local_offsets).astype(
            np.int32,
            copy=False,
        )
        chunk_valid_sizes[cursor:next_cursor] = np.minimum(
            chunk_size,
            seq_len - local_offsets,
        ).astype(np.int32, copy=False)
        cursor = next_cursor

    return torch.from_numpy(chunk_indices), torch.from_numpy(chunk_valid_sizes)


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
        seq_lens = cu_seqlens[1:].to(torch.int64) - cu_seqlens[:-1].to(torch.int64)
        num_chunks = ((seq_lens + matrix_size - 1) // matrix_size).sum().item()
        num_matrices = int(num_chunks) * num_bsnd_heads
    else:
        num_matrices = U_bsnd_fp16.numel() // (matrix_size * matrix_size)
    device = U_bsnd_fp16.device

    tensor_out = torch.zeros_like(U_bsnd_fp16, dtype=torch.float32)
    I_neg = _make_minus_identity(matrix_size, str(device))
    if cu_seqlens is not None:
        cu_seqlens = cu_seqlens.to(device=device, dtype=torch.int32).contiguous()
        chunk_indices, chunk_valid_sizes = _chunk_metadata_from_cu_seqlens(
            cu_seqlens,
            matrix_size,
        )
        chunk_indices = chunk_indices.to(device=device).contiguous()
        chunk_valid_sizes = chunk_valid_sizes.to(device=device).contiguous()
    else:
        chunk_indices = None
        chunk_valid_sizes = None

    torch.npu.synchronize()
    tri_inv_func(
        tensor_out,
        U_bsnd_fp16,
        I_neg,
        matrix_size,
        num_matrices,
        num_bsnd_heads,
        chunk_indices=chunk_indices,
        chunk_valid_sizes=chunk_valid_sizes,
    )
    torch.npu.synchronize()

    return tensor_out.cpu().to(torch.float64)


def _build_varlen_bsnd_case(
    gen,
    cu_seqlens: list[int],
    num_heads: int,
    chunk_size: int,
):
    """
    Build an unpadded BSND tensor plus reference output for varlen testing.

    Each sequence contributes only its true rows in the packed BSND tensor.
    """
    seq_lens = [
        cu_seqlens[i + 1] - cu_seqlens[i]
        for i in range(len(cu_seqlens) - 1)
    ]
    print(
        f"    varlen sequence lengths: {seq_lens} "
        f"(chunk_size={chunk_size}, num_heads={num_heads})"
    )

    total_tokens = cu_seqlens[-1]
    num_chunks = sum(
        (cu_seqlens[i + 1] - cu_seqlens[i] + chunk_size - 1) // chunk_size
        for i in range(len(cu_seqlens) - 1)
    )
    chunk_mats = gen(chunk_size, num_chunks, num_heads).to(torch.float64)

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
# Single test – standard layout
# ---------------------------------------------------------------------------
def _test_case(
    tri_inv_func,
    U: torch.Tensor,
    atol: float,
    rtol: float,
    ftol: float,
    label: str,
):
    U_fp16 = U.to(torch.half)
    golden = linalg_inv_ref(U_fp16)

    actual = _run_kernel(tri_inv_func, U_fp16.npu())

    frob = torch.sqrt(
        torch.sum((golden - actual) ** 2) / torch.sum(golden ** 2)
    ).item()

    assert np.allclose(
        actual.numpy(),
        golden.numpy(),
        atol=atol,
        rtol=rtol,
    ), f"[{label}] allclose failed — shape {U.shape}, rtol={rtol}"
    assert frob <= ftol, f"[{label}] Frobenius error {frob:.2e} > {ftol:.2e}"

    print(f"  PASS  {label}  frob={frob:.2e}")


# ---------------------------------------------------------------------------
# Single test – BSND layout
# ---------------------------------------------------------------------------
def _test_case_bsnd(
    tri_inv_func,
    U: torch.Tensor,
    B: int,
    S: int,
    N: int,
    D: int,
    atol: float,
    rtol: float,
    ftol: float,
    label: str,
):
    """
    U has shape (B*S//D, N, D, D) – the raw generator output.
    It is converted to (B, S, N, D) before being fed to the kernel.
    """
    U_fp16 = U.to(torch.half)
    golden = linalg_inv_ref(U_fp16)
    golden = golden.transpose(1, 2).contiguous().reshape(B, S, N, D)

    U_bsnd = U_fp16.transpose(1, 2).contiguous().reshape(B, S, N, D)
    actual = _run_kernel_bsnd(tri_inv_func, U_bsnd.npu())

    frob = torch.sqrt(
        torch.sum((golden - actual) ** 2) / torch.sum(golden ** 2)
    ).item()

    assert np.allclose(
        actual.numpy(),
        golden.numpy(),
        atol=atol,
        rtol=rtol,
    ), f"[{label}] allclose failed — shape {U_bsnd.shape}, rtol={rtol}"
    assert frob <= ftol, f"[{label}] Frobenius error {frob:.2e} > {ftol:.2e}"

    print(f"  PASS  {label}  frob={frob:.2e}")


def _test_case_bsnd_varlen(
    tri_inv_func,
    gen,
    cu_seqlens: list[int],
    N: int,
    D: int,
    atol: float,
    rtol: float,
    ftol: float,
    label: str,
):
    U_varlen, golden, cu_seqlens_tensor = _build_varlen_bsnd_case(
        gen,
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

    frob = torch.sqrt(
        torch.sum((golden - actual) ** 2) / torch.sum(golden ** 2)
    ).item()

    assert np.allclose(
        actual.numpy(),
        golden.numpy(),
        atol=atol,
        rtol=rtol,
    ), f"[{label}] allclose failed — shape {actual.shape}, rtol={rtol}"
    assert frob <= ftol, f"[{label}] Frobenius error {frob:.2e} > {ftol:.2e}"

    print(f"  PASS  {label}  frob={frob:.2e}")


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------
def run_tests(tri_inv_func):
    cases = [
        ("block_ones", block_ones_triu_matrix, 0, 0, 0),
        ("ones", ones_triu_matrix, 0, 0, 0),
        ("block_random", block_random_triu_matrix, 5e-5, 0.1, 1e-4),
        ("random", random_triu_matrix, 5e-5, 0.1, 1e-4),
    ]

    total = passed = 0

    print("=== Standard layout ===")
    sizes = [16, 32, 64, 128]
    x_dims = [1, 2, 4]
    y_dims = [2, 4]

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

    print("\n=== BSND layout ===")
    bsnd_configs = [
        (B, S, N, D)
        for B in [1, 4]
        for S in [128, 256]
        for N in [4, 8]
        for D in [16, 32, 64, 128]
        if S % D == 0
    ]

    for B, S, N, D in bsnd_configs:
        for name, gen, atol, rtol, ftol in cases:
            total += 1
            label = f"B={B} S={S} N={N} D={D} [{name}]"
            try:
                U = gen(D, B * S // D, N)
                _test_case_bsnd(tri_inv_func, U, B, S, N, D, atol, rtol, ftol, label)
                passed += 1
            except AssertionError as err:
                print(f"  FAIL  {label}: {err}")

    print("\n=== BSND varlen layout ===")
    varlen_configs = [
        (4, 16, [0, 15]),
        (4, 32, [0, 256, 500, 1000]),
        (4, 64, [0, 15, 100, 300, 1200, 2000]),
        (4, 16, [0, 1, 100, 300, 1200, 2048]),
        (4, 32, [0, 200, 512, 1200, 2048]),
    ]

    for N, D, cu_seqlens in varlen_configs:
        for name, gen, atol, rtol, ftol in cases:
            total += 1
            label = f"N={N} D={D} cu={cu_seqlens} [{name}]"
            try:
                _test_case_bsnd_varlen(
                    tri_inv_func,
                    gen,
                    cu_seqlens,
                    N,
                    D,
                    atol,
                    rtol,
                    ftol,
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
