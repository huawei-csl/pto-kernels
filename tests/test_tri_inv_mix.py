# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------

"""Tests for pto_tri_inv_mix, which inverts the leading cube_fraction of the
tiles on the cube units (the tri_inv_rec_unroll algorithm) and the rest on the
vector units (the tri_inv_col_sweep algorithm).

The input convention is the one of pto_tri_inv_rec_unroll: the tensor holds the
strictly triangular A and the kernel returns the inverse of I + A.

Tolerances are the looser of the two kernels' own tolerances for a given
generator, which in practice means the column sweep's: it applies n sequential
rank-1 updates, so its rounding error grows with the matrix size, while the
recursion keeps its accumulator in fp32 across the log2(n) levels.
"""

import random
from typing import Callable

import numpy as np
import pytest
import torch

from pto_kernels import pto_tri_inv_col_sweep, pto_tri_inv_mix, pto_tri_inv_rec_unroll

# Mirrors DefaultTriInvMixCubeFraction in csrc/host/torch_tri_inv_mix.h.
DEFAULT_CUBE_FRACTION = {16: 0.50, 32: 0.60, 64: 0.65, 128: 0.55}

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# Input generators. Each returns a strictly upper triangular A of shape
# (block_dim_x, block_dim_y, n, n); the kernel inverts I + A.
# ---------------------------------------------------------------------------


def random_tri_matrix(n, block_dim_x, block_dim_y, scale=0.1):
    return scale * torch.triu(torch.rand((block_dim_x, block_dim_y, n, n)), diagonal=1)


def ones_tri_matrix(n, block_dim_x, block_dim_y):
    return torch.triu(torch.ones((block_dim_x, block_dim_y, n, n)), diagonal=1)


def block_ones_triu_matrix(n, block_dim_x, block_dim_y):
    U = np.zeros((block_dim_x, block_dim_y, n, n))
    for x in range(block_dim_x):
        for y in range(block_dim_y):
            for i in range(0, n, 16):
                U[x, y, i : i + 16, i : i + 16] = np.ones((16, 16))
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


def deltanet_rand_triu_matrix(n, block_dim_x, block_dim_y):
    K = 0.5 - np.random.rand(block_dim_x, block_dim_y, n, n).astype(np.float64)
    for i in range(block_dim_x):
        for j in range(block_dim_y):
            K[i, j] = K[i, j] / np.linalg.norm(K[i, j], axis=-1, keepdims=True)
            K[i, j] = np.triu(K[i, j] @ K[i, j].T, k=1)
    return torch.from_numpy(K)


def _to_strict_tri(A: torch.Tensor, is_lower: bool, input_dtype: torch.dtype):
    """Returns the strictly triangular input the kernel expects."""
    if is_lower:
        A = A.transpose(-1, -2)
    return A.contiguous().to(input_dtype)


def _golden(A: torch.Tensor) -> torch.Tensor:
    """Inverse of I + A over the last two dimensions, in float64."""
    n = A.shape[-1]
    return torch.linalg.inv(A.double() + torch.eye(n, dtype=torch.float64))


def _frob_error(actual: torch.Tensor, golden: torch.Tensor) -> float:
    return float(torch.sqrt(torch.sum((golden - actual) ** 2) / torch.sum(golden**2)))


def _assert_inverse(actual: torch.Tensor, golden: torch.Tensor, atol, rtol, ftol):
    frob_error = _frob_error(actual, golden)
    assert np.allclose(actual.numpy(), golden.numpy(), atol=atol, rtol=rtol), (
        f"Error at allclose - shape: {tuple(actual.shape)} - rtol: {rtol}."
        f" max abs error: {float((actual - golden).abs().max())}."
        f" frob error: {frob_error}."
    )
    assert frob_error <= ftol, f"frob_error: {frob_error} (ftol {ftol})"


# ---------------------------------------------------------------------------
# Dense input: the tiles are stored one after the other, row-major.
# ---------------------------------------------------------------------------


def _test_tri_inv_mix(
    A: torch.Tensor,
    atol: float,
    rtol: float,
    ftol: float,
    is_lower: bool,
    input_dtype: torch.dtype = torch.float16,
):
    A = _to_strict_tri(A, is_lower, input_dtype)
    golden = _golden(A)

    actual = pto_tri_inv_mix(A.npu(), is_lower=is_lower)
    torch.npu.synchronize()

    _assert_inverse(actual.cpu().to(torch.float64), golden, atol, rtol, ftol)


@pytest.mark.parametrize("n", [16, 32, 64, 128])
@pytest.mark.parametrize("block_dim_x", [1, 2, 3, 4])
@pytest.mark.parametrize("block_dim_y", [2, 4, 8])
@pytest.mark.parametrize("is_lower", [False, True])
@pytest.mark.parametrize(
    "matrix_gen,atol,rtol,ftol",
    [
        (block_ones_triu_matrix, 0, 0, 0),
        (ones_tri_matrix, 0, 0, 0),
        (block_random_triu_matrix, 5e-5, 0.1, 1e-4),
        (random_tri_matrix, 3.4e-4, 0.1, 6.7e-4),
    ],
)
# pylint: disable=too-many-positional-arguments
def test_tri_inv_mix(
    n: int,
    block_dim_x: int,
    block_dim_y: int,
    matrix_gen: Callable,
    atol: float,
    rtol: float,
    ftol: float,
    is_lower: bool,
):
    _test_tri_inv_mix(
        matrix_gen(n, block_dim_x, block_dim_y), atol, rtol, ftol, is_lower
    )


@pytest.mark.parametrize("n", [16, 32, 64, 128])
# Tile counts below the cube core count run on the cube only and leave the
# vector part empty; the larger ones are not divided evenly by the split.
@pytest.mark.parametrize("num_tiles", [1, 2, 3, 47, 48, 97])
@pytest.mark.parametrize("is_lower", [False, True])
def test_tri_inv_mix_tile_counts(n: int, num_tiles: int, is_lower: bool):
    """The split must hold for tile counts that do not divide evenly over the
    cores, and when one of the two halves is empty."""
    A = random_tri_matrix(n, num_tiles, 1)
    _test_tri_inv_mix(A, atol=3.4e-4, rtol=0.1, ftol=6.7e-4, is_lower=is_lower)


@pytest.mark.parametrize("n", [32, 64, 128])
@pytest.mark.parametrize("is_lower", [False, True])
def test_tri_inv_mix_deltanet(n: int, is_lower: bool):
    """Matrices shaped like a gated linear-attention chunk."""
    A = deltanet_rand_triu_matrix(n, block_dim_x=7, block_dim_y=13)
    eps = torch.finfo(torch.float16).eps
    _test_tri_inv_mix(
        A, atol=eps, rtol=0.1, ftol=eps * np.sqrt(n) / 8, is_lower=is_lower
    )


@pytest.mark.parametrize("n", [16, 32, 64, 128])
@pytest.mark.parametrize("num_tiles", [2, 3, 48, 97])
@pytest.mark.parametrize("is_lower", [False, True])
@pytest.mark.parametrize("cube_fraction", [None, 0.0, 0.5, 0.8, 1.0])
def test_tri_inv_mix_matches_both_halves(
    n: int, num_tiles: int, is_lower: bool, cube_fraction: float | None
):
    """The contract of the kernel: the leading round(cube_fraction * num_tiles)
    tiles come out of the cube algorithm and the rest out of the vector one,
    bit for bit. None checks the per-size default fraction. With fewer tiles
    than cube cores the fraction is ignored and every tile runs on the cube.

    The column sweep never reads the diagonal of its input -- it assumes a unit
    one -- so the strictly triangular tensor the recursion expects means the
    same matrix to both kernels.
    """
    A = _to_strict_tri(random_tri_matrix(n, num_tiles, 1), is_lower, torch.float16)
    A_npu = A.npu()

    kwargs = {} if cube_fraction is None else {"cube_fraction": cube_fraction}
    mixed = pto_tri_inv_mix(A_npu, is_lower=is_lower, **kwargs).cpu()
    cube_only = pto_tri_inv_rec_unroll(A_npu, is_lower=is_lower).cpu()
    vector_only = pto_tri_inv_col_sweep(A_npu, is_lower=is_lower).cpu()
    torch.npu.synchronize()

    num_cube_tiles = num_tiles
    if num_tiles >= torch.npu.get_device_properties(0).cube_core_num:
        # Round half away from zero, as std::lround does on the host.
        fraction = DEFAULT_CUBE_FRACTION[n] if cube_fraction is None else cube_fraction
        num_cube_tiles = int(np.floor(fraction * num_tiles + 0.5))
    assert torch.equal(mixed[:num_cube_tiles], cube_only[:num_cube_tiles])
    assert torch.equal(mixed[num_cube_tiles:], vector_only[num_cube_tiles:])


@pytest.mark.parametrize("cube_fraction", [-0.1, 1.5])
def test_tri_inv_mix_rejects_bad_cube_fraction(cube_fraction: float):
    A = random_tri_matrix(32, 4, 1).to(torch.float16).npu()
    with pytest.raises(RuntimeError, match="cube_fraction"):
        pto_tri_inv_mix(A, cube_fraction=cube_fraction)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_tri_inv_mix_rejects_unsupported_dtype(dtype: torch.dtype):
    """The vector half has no bf16 and no fp32 rank-1 update at this tile size,
    so the kernel takes fp16 only, unlike pto_tri_inv_rec_unroll."""
    A = random_tri_matrix(32, 4, 1).to(dtype).npu()
    with pytest.raises(RuntimeError, match="fp16"):
        pto_tri_inv_mix(A)


# ---------------------------------------------------------------------------
# BSND layout: the matrices are the C x C blocks over the (chunked) sequence
# dimension of a [B, S, N, C] tensor, so consecutive rows of a tile are N * C
# elements apart in memory.
# ---------------------------------------------------------------------------


def _to_bsnd(tiles: torch.Tensor, B: int, S: int, N: int, C: int) -> torch.Tensor:
    """(B * S // C, N, C, C) tiles -> [B, S, N, C] BSND tensor."""
    return tiles.transpose(1, 2).contiguous().reshape(B, S, N, C)


@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("S", [128, 1024])
@pytest.mark.parametrize("N", [4, 8])
@pytest.mark.parametrize("C", [16, 32, 64, 128])
@pytest.mark.parametrize("is_lower", [False, True])
@pytest.mark.parametrize(
    "matrix_gen,atol,rtol,ftol",
    [
        (block_ones_triu_matrix, 0, 0, 0),
        (ones_tri_matrix, 0, 0, 0),
        (block_random_triu_matrix, 5e-5, 0.1, 1e-4),
        (random_tri_matrix, 3.4e-4, 0.1, 6.7e-4),
    ],
)
# pylint: disable=too-many-positional-arguments
def test_tri_inv_mix_bsnd(
    B: int,
    S: int,
    N: int,
    C: int,
    matrix_gen: Callable,
    atol: float,
    rtol: float,
    ftol: float,
    is_lower: bool,
):
    A = _to_strict_tri(matrix_gen(C, B * S // C, N), is_lower, torch.float16)
    golden = _to_bsnd(_golden(A), B, S, N, C)

    actual = pto_tri_inv_mix(
        _to_bsnd(A, B, S, N, C).npu(), is_bsnd_format=True, is_lower=is_lower
    )
    torch.npu.synchronize()

    _assert_inverse(actual.cpu().to(torch.float64), golden, atol, rtol, ftol)


# ---------------------------------------------------------------------------
# Variable-length BSND: the last chunk of a sequence is shorter than the chunk
# size, so only that leading square block of it is inverted.
# ---------------------------------------------------------------------------


def generate_random_sequence_lengths(
    num_sequences: int, total_tokens: int
) -> list[int]:
    """num_sequences random sequence lengths summing to total_tokens."""
    if total_tokens < num_sequences:
        raise ValueError("total_tokens must be >= num_sequences.")
    cuts = sorted(
        (
            np.random.choice(total_tokens - 2, num_sequences - 1, replace=False) + 1
        ).tolist()
    )
    bounds = [0, *cuts, total_tokens]
    return [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]


def _varlen_bsnd_input(
    seq_lens: list[int],
    num_heads: int,
    chunk_size: int,
    is_lower: bool,
    scale: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Packed [1, total_tokens, N, C] strictly triangular input, and the
    per-chunk inverse of I + A over the valid block of each chunk.

    The padding of a short chunk, and of the C columns beyond it, stays zero in
    both the input and the golden output.
    """
    cu_seqlens = np.cumsum([0, *seq_lens], dtype=np.int64)
    total_tokens = int(cu_seqlens[-1])
    shape = (1, total_tokens, num_heads, chunk_size)
    packed = torch.zeros(shape, dtype=torch.float16)
    golden = torch.zeros(shape, dtype=torch.float64)

    for seq_start, seq_end in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=False):
        for chunk_start in range(int(seq_start), int(seq_end), chunk_size):
            valid = min(chunk_size, int(seq_end) - chunk_start)
            strict = scale * torch.rand((num_heads, valid, valid))
            strict = torch.tril(strict, -1) if is_lower else torch.triu(strict, 1)
            strict = strict.to(torch.float16)
            inverse = torch.linalg.inv(strict.double() + torch.eye(valid).double())
            packed[0, chunk_start : chunk_start + valid, :, :valid] = strict.permute(
                1, 0, 2
            )
            golden[0, chunk_start : chunk_start + valid, :, :valid] = inverse.permute(
                1, 0, 2
            )

    return (
        packed.contiguous(),
        golden,
        torch.tensor(cu_seqlens.tolist(), dtype=torch.int32),
    )


@pytest.mark.parametrize("B", [1, 7, 32])
@pytest.mark.parametrize("N", [4])
@pytest.mark.parametrize("chunk_size", [16, 32, 64, 128])
@pytest.mark.parametrize("total_tokens", [1024, 3031])
@pytest.mark.parametrize("is_lower", [False, True])
# pylint: disable=too-many-positional-arguments
def test_tri_inv_mix_varlen_bsnd(
    B: int,
    N: int,
    chunk_size: int,
    total_tokens: int,
    is_lower: bool,
):
    """Both halves have to agree on which global tile a chunk is, so that the
    tail chunks land on the right engine and the padding comes back untouched."""
    seq_lens = generate_random_sequence_lengths(B, total_tokens)
    packed, golden, cu_seqlens = _varlen_bsnd_input(seq_lens, N, chunk_size, is_lower)

    actual = pto_tri_inv_mix(
        packed.npu(),
        cu_seqlens.npu(),
        is_bsnd_format=True,
        is_lower=is_lower,
    )
    torch.npu.synchronize()

    _assert_inverse(actual.cpu().to(torch.float64), golden, 1e-3, 5e-2, 1e-3)
