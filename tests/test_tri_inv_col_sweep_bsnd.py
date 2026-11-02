# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------
"""BSND layout tests for the column sweep triangular inverse.

The oracle here is the kernel's own dense path, which test_tri_inv_col_sweep.py
already checks against numpy. BSND only changes where a tile lives in memory,
not what is computed, so feeding the same tiles through both layouts has to
give the same bits -- which keeps these tests free of the transposition
convention the dense test has to spell out.
"""

import numpy as np
import pytest
import torch

from pto_kernels import pto_tri_inv_col_sweep

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def random_tiles(num_chunks: int, num_heads: int, n: int, dtype: torch.dtype):
    """Strictly lower triangular tiles; the kernel takes a unit diagonal."""
    t = 0.1 * torch.rand((num_chunks, num_heads, n, n))
    return torch.tril(t, diagonal=-1).to(dtype)


def to_bsnd(tiles: torch.Tensor) -> torch.Tensor:
    """(num_chunks, N, D, D) tiles -> [1, num_chunks * D, N, D]."""
    num_chunks, num_heads, n, _ = tiles.shape
    return tiles.transpose(1, 2).contiguous().reshape(1, num_chunks * n, num_heads, n)


def from_bsnd(t: torch.Tensor, num_chunks: int, num_heads: int, n: int):
    """Inverse of to_bsnd."""
    return t.reshape(num_chunks, n, num_heads, n).transpose(1, 2).contiguous()


@pytest.mark.parametrize("n", [16, 32, 64, 128])
@pytest.mark.parametrize("num_chunks", [1, 3])
@pytest.mark.parametrize("num_heads", [1, 2, 5])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_bsnd_matches_dense(n, num_chunks, num_heads, dtype):
    """A tile inverted in BSND layout must equal the same tile inverted dense."""
    tiles = random_tiles(num_chunks, num_heads, n, dtype)

    dense = pto_tri_inv_col_sweep(tiles.npu()).cpu()
    bsnd = pto_tri_inv_col_sweep(to_bsnd(tiles).npu(), is_bsnd_format=True).cpu()

    assert torch.equal(from_bsnd(bsnd, num_chunks, num_heads, n), dense)


@pytest.mark.parametrize("n", [16, 32, 64])
@pytest.mark.parametrize("num_heads", [1, 3])
@pytest.mark.parametrize(
    "seq_lens",
    [
        [5],  # single partial chunk, shorter than n
        [40],  # one full chunk plus a tail, for n = 16 and 32
        [16, 24],  # two sequences, the second with a tail
    ],
)
def test_varlen_bsnd_matches_padded_dense(n, num_heads, seq_lens):
    """Every valid chunk, including a partial tail, matches the dense result
    of that chunk zero padded out to n (which the implicit unit diagonal turns
    into diag(A, I), so the padding inverts to itself and does not disturb the
    valid block)."""
    dtype = torch.float16
    cu = np.cumsum([0, *seq_lens], dtype=np.int64)
    total_tokens = int(cu[-1])
    cu_t = torch.tensor(cu.tolist(), dtype=torch.int32)

    packed = torch.zeros((1, total_tokens, num_heads, n), dtype=dtype)
    chunks = []  # (row_start, valid_size)
    for s, e in zip(cu[:-1].tolist(), cu[1:].tolist()):
        for start in range(s, e, n):
            v = min(n, e - start)
            blk = torch.tril(0.1 * torch.rand((num_heads, v, v)), diagonal=-1).to(dtype)
            packed[0, start : start + v, :, :v] = blk.permute(1, 0, 2)
            chunks.append((start, v))

    actual = pto_tri_inv_col_sweep(
        packed.npu(), is_bsnd_format=True, cu_seqlens=cu_t.npu()
    ).cpu()

    for start, v in chunks:
        # Rebuild this chunk as a dense n x n tile, zero padded past v.
        tile = torch.zeros((1, num_heads, n, n), dtype=dtype)
        tile[0, :, :v, :v] = packed[0, start : start + v, :, :v].permute(1, 0, 2)
        expected = pto_tri_inv_col_sweep(tile.npu()).cpu()[0]

        got = actual[0, start : start + v, :, :v].permute(1, 0, 2)
        assert torch.equal(
            got, expected[:, :v, :v]
        ), f"chunk at row {start} (valid {v}) differs"
