"""
Educational emulation of ``solve_tril`` (``fla_vendor/solve_tril.py``).

For a strictly lower-triangular block :math:`L` (zeros on/above diagonal), the kernel
computes :math:`(I + L)^{-1}` in the same packed layout ``[B, T, H, BT]``.

For each chunk, let :math:`L \\in \\mathbb{R}^{BT \\times BT}` be strictly lower.
Then :math:`(I+L)^{-1}` is the inverse of a unit lower-triangular matrix, equivalent
to the inverse WY factor used in the recurrence.
"""

from __future__ import annotations

import numpy as np
import torch


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

    if cu_seqlens is None:
        seg_ranges = [(0, t - (t % bt))]
    else:
        cu = cu_seqlens.detach().cpu().tolist()
        seg_ranges = []
        for i in range(len(cu) - 1):
            bos, eos = cu[i], cu[i + 1]
            seg_ranges.append((bos, eos - ((eos - bos) % bt)))

    eye = torch.eye(bt, dtype=torch.float32, device=A.device)

    for bos, eos in seg_ranges:
        for ic in range((eos - bos) // bt):
            s = bos + ic * bt
            e = s + bt
            for i_h in range(h):
                # SRAM tile: one BT x BT block (rows loaded from A's packed layout)
                rows = []
                for r in range(bt):
                    # GLOBAL row s+r stores L[r, :]
                    row_global = A[0, s + r, i_h, :].detach().float().cpu().numpy().astype(np.float32)
                    rows.append(row_global.copy())
                l_mat = np.stack(rows, axis=0)
                # Strictly lower: zero diagonal and upper (matches KKT construction)
                l_t = np.tril(l_mat, k=-1).astype(np.float32)
                l_torch = torch.from_numpy(np.ascontiguousarray(l_t)).to(device=A.device)
                # (I + L)^{-1}
                inv_block = torch.linalg.inv(eye + l_torch)
                for r in range(bt):
                    ai[0, s + r, i_h, :] = inv_block[r, :].to(out_dt)

    return ai
