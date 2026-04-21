"""
Shared helpers for educational PyTorch emulation of GDN **PTO** (NPU) kernels.

This mirrors the role of ``torch_emulation_triton/_common.py``, but terminology matches
the Ascend / PTO stack used in ``dynamic_bsnd/*.cpp``.

Memory hierarchy (conceptual, per AI core)
------------------------------------------
**GM (global memory)** — Off-chip HBM. All kernel arguments live here. In Torch we use
ordinary tensors (``torch.Tensor``).

**UB (unified buffer)** — On-chip SRAM (~256 KB), **Vec engine** operands. In emulation
we name workspace tensors ``*_ub`` when a kernel keeps a full chunk row-strip or ``C×C``
tile in UB before ``TSTORE`` to GM.

**L1** — Cube matrix unit cache. GEMM operands ``K``, ``Q``, ``V``, ``S`` are ``TLOAD``'d
into L1 in NZ fractal layout; ``TRESHAPE`` can reinterpret as ``K^T`` (ZN) without moving
data.

**L0A / L0B / L0C** — Register tiles feeding the Cube ``TMATMUL``. **L0C** holds the fp32
accumulator (even when inputs are fp16).

Concrete ``TLOAD`` / ``TSTORE`` / ``TMOV`` / ``TADD`` / ``TEXTRACT`` / K-tiled ``TMATMUL`` stand-ins
live in ``_memory.py`` (``gemm_v0_accum_fp16`` mirrors ``chunk_h_kernel.cpp`` ``gemm_v0`` with
explicit L1→L0A/L0B stripes).

Sequential Torch code does not model **set_flag / wait_flag** or **ffts_cross_core_sync**;
we express the same mathematics as if Cube and Vec ran one after another.

Chunk iteration
---------------
``prepare_chunk_indices`` / ``iter_packed_bt_chunks`` follow the same packed-sequence
convention as the Triton emulation: one logical program per ``(sequence, chunk_index)``
when ``cu_seqlens`` is set.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch


def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """
    Build the varlen chunk launch table (same layout as ``torch_emulation_triton``).

    Returns ``[num_chunks, 2]`` with ``(seq_id, chunk_index_within_seq)`` rows.
    """
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    nc = (lens + chunk_size - 1) // chunk_size
    parts = [torch.arange(int(n), device=cu_seqlens.device, dtype=torch.long) for n in nc.tolist()]
    indices = torch.cat(parts, dim=0) if parts else cu_seqlens.new_empty(0, dtype=torch.long)
    seq_ids = (indices == 0).cumsum(0) - 1
    return torch.stack([seq_ids, indices], dim=1).to(cu_seqlens)


def iter_packed_bt_chunks(
    *,
    cu_seqlens: torch.Tensor | None,
    total_t: int,
    bt: int,
    chunk_indices: torch.Tensor | None,
) -> Iterator[tuple[int, int, int]]:
    """Yield ``(bos, i_tc, span)`` in the same order as the Triton emulation."""
    if cu_seqlens is None:
        nt = (total_t + bt - 1) // bt
        for i_tc in range(nt):
            span = min(bt, total_t - i_tc * bt)
            yield 0, i_tc, span
    else:
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, bt)
        for row in chunk_indices:
            i_n = int(row[0].item())
            i_tc = int(row[1].item())
            bos = int(cu_seqlens[i_n].item())
            eos = int(cu_seqlens[i_n + 1].item())
            t_seg = eos - bos
            span = min(bt, t_seg - i_tc * bt)
            yield bos, i_tc, span


def safe_exp_torch(x: torch.Tensor) -> torch.Tensor:
    """``exp(x)`` where ``x <= 0``, else ``0`` — matches ``verify_dynamic_bsnd._safe_exp``."""
    return torch.where(x <= 0, torch.exp(x), torch.zeros_like(x))


def total_chunks(
    batch_size: int,
    seq_len: int,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None,
) -> int:
    """Same chunk count as ``dynamic_bsnd.dynamic_kernel_libs.total_chunks``."""
    if cu_seqlens is None:
        return batch_size * ((seq_len + chunk_size - 1) // chunk_size)
    cu = cu_seqlens.detach().cpu().tolist()
    return sum((cu[i + 1] - cu[i] + chunk_size - 1) // chunk_size for i in range(len(cu) - 1))


def seq_ranges(total_t: int, cu_seqlens: torch.Tensor | None) -> list[tuple[int, int]]:
    """Inclusive-exclusive ``(bos, eos)`` segments in packed time."""
    if cu_seqlens is None:
        return [(0, total_t)]
    cu = cu_seqlens.tolist() if hasattr(cu_seqlens, "tolist") else list(cu_seqlens)
    return [(cu[i], cu[i + 1]) for i in range(len(cu) - 1)]


def print_tile_like(name: str, t: torch.Tensor) -> None:
    """Optional debug helper (same spirit as ``step1_baseline_numpy_sim._print_tile_memory``)."""
    kib = t.numel() * t.element_size() / 1024.0
    print(f"[tile-mem] {name}: shape={tuple(t.shape)}, dtype={t.dtype}, ~{kib:.1f} KiB")
