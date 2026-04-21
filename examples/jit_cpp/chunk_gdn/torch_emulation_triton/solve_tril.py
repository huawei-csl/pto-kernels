"""
Educational emulation of ``solve_tril`` (``fla_vendor/solve_tril.py``).

Mathematics
-----------
Input ``A`` holds strictly **lower** triangular blocks from ``chunk_scaled_dot_kkt`` (zeros on and
above the diagonal within each ``BT × BT`` chunk view). Let ``L`` be that strict-lower part. The
kernel computes

.. math::

    (I + L)^{-1}

in the same packed layout ``[B, T, H, BT]``: each global time row stores one row of the **inverse**
block for its chunk. This is the WY factor inverse used before ``recompute_w_u_fwd``.

**Note:** Reference Triton may use a multi-stage 16×16 pipeline; this emulation uses a single
``torch.linalg.inv(I + tril(A,-1))`` on **padded** ``BT × BT`` tiles — same algebra per chunk.

Memory: global vs tile
----------------------
**Global:**

- ``A``: ``[B, T, H, BT]`` — packed lower rows (input).
- Output ``ai``: same shape — packed rows of ``(I+L)^{-1}``.

**Tile:**

- ``l_pad``: ``[BT, BT]`` — one chunk’s rows of ``A`` copied and strict-lower extracted; zeros
  below ``span`` mimic masked load.
- ``inv_block``: ``[BT, BT]`` — full inverse in fp32; rows ``[:span]`` written back to **global** ``ai``.
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

    ``chunk_indices_large_block`` is accepted for API parity but **ignored** here (Triton uses it
    for an internal 16×16 pass); only ``chunk_indices_bt``-style chunking at ``BT`` matters for
    this pure-PyTorch path.
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
            # l_pad [BT, BT]: GLOBAL A rows for this chunk; tail rows (span..BT) stay zero (mask).
            l_pad = torch.zeros(bt, bt, device=A.device, dtype=torch.float32)
            l_pad[:span, :] = A[0, s : s + span, i_h, :].float()
            # Strict-lower L from the block (diag and upper zero); same as KKT output convention.
            l_t = torch.tril(l_pad, diagonal=-1)
            # eye [BT, BT]; inv_block [BT, BT] = (I + L)^{-1} in fp32 (full tile, then store prefix rows).
            inv_block = torch.linalg.inv(eye + l_t)
            # GLOBAL ai [B,T,H,BT]: one inverse row per global time row (same packed layout as A).
            ai[0, s : s + span, i_h, :] = inv_block[:span, :].to(out_dt)

    return ai
