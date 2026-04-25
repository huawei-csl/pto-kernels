# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------


import torch
import torch.nn.functional as torch_functional
import pytest
import numpy as np
import random
from pto_kernels import pto_tri_inv_rec_unroll

pytestmark = pytest.mark.npu

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


def generate_random_sequence_lengths(
    num_sequences: int, total_tokens: int
) -> list[int]:
    """
    Generates a list of num_sequences integers in the range (1, total_tokens).
    These integers denote the index where each "input sequence" ends.
    """
    if total_tokens < num_sequences:
        raise ValueError("total_tokens must be >= num_sequences.")

    # num_sequences-1 sorted random integers in the range [1,...,total_tokens-1]
    cummulative_lengths = sorted(
        list(np.random.choice(total_tokens - 2, num_sequences - 1, replace=False) + 1)
    )
    cummulative_lengths = [0] + cummulative_lengths
    cummulative_lengths.append(total_tokens)
    return [
        cummulative_lengths[i + 1] - cummulative_lengths[i]
        for i in range(len(cummulative_lengths) - 1)
    ]


def transpose_valid_chunks(
    A: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    transposed = torch.zeros_like(A)
    for seq_start, seq_end in zip(
        cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=False
    ):
        for chunk_start in range(seq_start, seq_end, chunk_size):
            actual_size = min(chunk_size, seq_end - chunk_start)
            chunk = A[:, chunk_start : chunk_start + actual_size, :, :actual_size]
            transposed[:, chunk_start : chunk_start + actual_size, :, :actual_size] = (
                chunk.transpose(1, 3)
            )
    return transposed


def chunk_scaled_dot_kkt_fwd_emulated(
    k: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    total_tokens = int(cu_seqlens[-1].item())
    num_heads = k.shape[2]
    A = torch.zeros(
        (1, total_tokens, num_heads, chunk_size), dtype=k.dtype, device=k.device
    )

    for seq_start, seq_end in zip(
        cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=False
    ):
        for chunk_start in range(seq_start, seq_end, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_end)
            actual_size = chunk_end - chunk_start
            k_chunk = (
                k[:, chunk_start:chunk_end].transpose(1, 2).to(torch.float32).npu()
            )
            beta_chunk = (
                beta[:, chunk_start:chunk_end]
                .transpose(1, 2)
                .unsqueeze(-1)
                .to(torch.float32)
                .npu()
            )
            scores = torch.matmul(k_chunk, k_chunk.transpose(-1, -2))
            scores = torch.tril(scores * beta_chunk, diagonal=-1)  # .to(k.dtype)
            scores = torch.tril(torch.ones(scores.shape), diagonal=-1).to(k.dtype).npu()
            A[:, chunk_start:chunk_end, :, :actual_size] = scores.transpose(1, 2)

    return A


def all_ones_varlen_triangular_tensor(
    cu_seqlens: torch.Tensor, chunk_size: int, num_heads: int, feature_dim: int
) -> torch.Tensor:
    total_tokens = int(cu_seqlens[-1].item())
    A = torch.zeros((1, total_tokens, num_heads, chunk_size), dtype=torch.float16)
    ones_tensor = torch.ones(
        (1, total_tokens, num_heads, feature_dim),
        dtype=torch.float16,
    )
    for seq_start, seq_end in zip(
        cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=False
    ):
        for chunk_start in range(seq_start, seq_end, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_end)
            actual_size = chunk_end - chunk_start
            chunk_shape = list(
                ones_tensor[:, chunk_start:chunk_end].transpose(1, 2).shape
            )
            chunk_shape[-1] = chunk_shape[-2]
            chunk = torch.tril(torch.ones(chunk_shape), diagonal=-1)  # .to(k.dtype)
            A[:, chunk_start:chunk_end, :, :actual_size] = chunk.transpose(1, 2)

    return A


def build_variable_len_input(
    seq_lens: list[int],
    num_heads: int,
    chunk_size: int,
    feature_dim: int,
    matrix_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    cu_seqlens = np.cumsum([0, *seq_lens], dtype=np.int64)
    cu_seqlens_tensor = torch.tensor(cu_seqlens.tolist(), dtype=torch.int32)
    total_tokens = int(cu_seqlens[-1])
    if matrix_type == "ones":
        packed_input = transpose_valid_chunks(
            all_ones_varlen_triangular_tensor(
                cu_seqlens_tensor, chunk_size, num_heads, feature_dim
            ),
            cu_seqlens_tensor,
            chunk_size,
        )
    elif matrix_type == "random":
        k = torch_functional.normalize(
            torch.randn(
                (1, total_tokens, num_heads, feature_dim),
                dtype=torch.float16,
            ),
            dim=-1,
        )
        beta = torch.randn((1, total_tokens, num_heads), dtype=torch.float16).sigmoid()
        packed_input = transpose_valid_chunks(
            chunk_scaled_dot_kkt_fwd_emulated(k, beta, cu_seqlens_tensor, chunk_size),
            cu_seqlens_tensor,
            chunk_size,
        )
    else:
        raise RuntimeError(f"unknown matrix type to test: {matrix_type}")
    return packed_input.contiguous().npu(), cu_seqlens_tensor.npu()


def _reference_inverse(
    A: torch.Tensor, cu_seqlens: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    A_cpu = A.cpu().to(torch.float64)
    ref = torch.zeros_like(A_cpu, dtype=torch.float64)
    for seq_start, seq_end in zip(
        cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=False
    ):
        for chunk_start in range(seq_start, seq_end, chunk_size):
            actual_size = min(chunk_size, seq_end - chunk_start)
            mat_to_invert = (
                A_cpu[
                    :, chunk_start : chunk_start + actual_size, :, :actual_size
                ].transpose(1, 2)
                + torch.eye(actual_size, dtype=torch.float64)[None, None, ...]
            ).numpy()
            ref[:, chunk_start : chunk_start + actual_size, :, :actual_size] = (
                torch.tensor(np.linalg.inv(mat_to_invert)).transpose(1, 2)
            )
    return ref


def _test_inverse_accuracy(
    A: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_size: int,
    atol: float,
    rtol: float,
    ftol: float,
):

    ref = _reference_inverse(A, cu_seqlens, chunk_size)
    tri = pto_tri_inv_rec_unroll(A, True, cu_seqlens)
    torch.npu.synchronize()
    tri = tri.cpu().to(torch.float64)
    torch.npu.synchronize()

    assert torch.allclose(tri, ref, atol=atol, rtol=rtol)
    frob_error = torch.sqrt(torch.sum((ref - tri) ** 2) / torch.sum(ref**2)).item()
    assert frob_error <= ftol


@pytest.mark.parametrize("B", [1, 2, 7, 17, 32, 93])
@pytest.mark.parametrize("N", [4])
@pytest.mark.parametrize(
    "chunk_size", [32, 64, 128]
)  # Equal to matrix size for inversion
@pytest.mark.parametrize("total_tokens", [1024, 3031, 10937])
@pytest.mark.parametrize(
    "matrix_type,atol,rtol,ftol", [("ones", 0, 0, 0), ("random", 1e-5, 5e-2, 1e-2)]
)
def test_tri_inv_rec_unroll_variable_length(
    B: int,
    N: int,
    chunk_size: int,
    total_tokens: int,
    matrix_type: str,
    atol: float,
    rtol: float,
    ftol: float,
):
    """
    Args:
        B: Number of sequences
        N: Number of BSND heads
        chunk_size: Equal to matrix size for inversion
        total_tokens: Total number of tokens (sum of sequence lengths)
        matrix_type: Type of matrix to test
        atol: Max abs tolerance for torch.allclose
        rtol: Max rel tolerance for torch.allclose
        ftol: Frobenius norm-wise relative error tolerance
    """
    default_feature_dim = 64
    seq_lens = generate_random_sequence_lengths(B, total_tokens)
    packed_input, cu_seqlens = build_variable_len_input(
        seq_lens=seq_lens,
        num_heads=N,
        chunk_size=chunk_size,
        feature_dim=default_feature_dim,
        matrix_type=matrix_type,
    )
    _test_inverse_accuracy(packed_input, cu_seqlens, chunk_size, atol, rtol, ftol)
