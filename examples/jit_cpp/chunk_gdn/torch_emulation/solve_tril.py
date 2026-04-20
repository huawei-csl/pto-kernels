"""
Educational emulation of ``solve_tril`` (``fla_vendor/solve_tril.py``).

For a strictly lower-triangular block :math:`L` (zeros on/above diagonal), the kernel
computes :math:`(I + L)^{-1}` in the same packed layout ``[B, T, H, BT]``.

For each chunk, let :math:`L \\in \\mathbb{R}^{BT \\times BT}` be strictly lower.
Then :math:`(I+L)^{-1}` is the inverse of a unit lower-triangular matrix, equivalent
to the inverse WY factor used in the recurrence.

Chunk iteration matches Triton ``chunk_indices`` (partial tiles zero-padded before inverse).
"""

from __future__ import annotations

import torch

from ._common import iter_packed_bt_chunks, prepare_chunk_indices


def solve_tril(
    A: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices_large_block: torch.Tensor | None = None,
    chunk_indices_bt: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    """
    Same arguments as ``fla_vendor.solve_tril.solve_tril``.

    Reference inverse: ``Ai = inv(I + L)`` in float32 per chunk, where ``L`` is read from
    the strict-lower part of the packed block rows of ``A``.
    """
    b, t, h, bt = A.shape
    assert bt in (16, 32, 64)
    out_dt = output_dtype if output_dtype is not None else A.dtype
    ai = torch.empty(b, t, h, bt, device=A.device, dtype=out_dt)

    if cu_seqlens is not None and chunk_indices_bt is None:
        chunk_indices_bt = prepare_chunk_indices(cu_seqlens, bt)

    eye = torch.eye(bt, dtype=torch.float32, device=A.device)

    for bos, _i_tc, span in iter_packed_bt_chunks(
        cu_seqlens=cu_seqlens, total_t=t, bt=bt, chunk_indices=chunk_indices_bt
    ):
        if span <= 0:
            continue
        s = bos + _i_tc * bt
        for i_h in range(h):
            l_pad = torch.zeros(bt, bt, device=A.device, dtype=torch.float32)
            l_pad[:span, :] = A[0, s : s + span, i_h, :].float()
            l_t = torch.tril(l_pad, diagonal=-1)
            inv_block = torch.linalg.inv(eye + l_t)
            ai[0, s : s + span, i_h, :] = inv_block[:span, :].to(out_dt)

    return ai
