"""
PyTorch emulation of the five ``dynamic_bsnd`` PTO kernels (educational).

Modules mirror kernel filenames:

- ``chunk_cumsum`` — Vec prefix sum inside each chunk
- ``scaled_dot_kkt`` — Cube ``K@K^T`` + Vec gating + strict-lower mask
- ``wy_fast`` — two gated GEMMs for ``W`` and ``U``
- ``chunk_h`` — recurrent ``D×D`` state update
- ``chunk_o`` — three GEMMs + PTO Vec gating (``exp(min Δg, 0)`` on QK)

See each module's docstring for UB / L1 / L0 annotations. Call sites pre-allocate SRAM stand-ins and
route copies through ``_memory`` helpers so the layout matches the PTO kernels.
"""

from __future__ import annotations

from .chunk_cumsum import chunk_cumsum_fwd
from .chunk_h import chunk_h_fwd
from .chunk_o import chunk_o_fwd, chunk_o_fwd_fla
from .scaled_dot_kkt import scaled_dot_kkt_fwd
from .wy_fast import wy_fast_fwd

__all__ = [
    "chunk_cumsum_fwd",
    "scaled_dot_kkt_fwd",
    "wy_fast_fwd",
    "chunk_h_fwd",
    "chunk_o_fwd",
    "chunk_o_fwd_fla",
]
