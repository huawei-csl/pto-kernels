"""
Shared helpers for educational PyTorch emulation of GDN Triton kernels.

Memory model (conceptual)
---------------------------
Triton kernels distinguish **on-chip** state (registers / shared memory tiles loaded with
``tl.load``, computed with ``tl.dot``, then written with ``tl.store``) from **global** tensors
in device memory (DRAM). In this emulation:

- Variables named like ``*_pad``, ``blk``, ``a_tile``, or holding a full ``BT × BT`` / ``BT × K``
  micro-block are **tile / SRAM stand-ins**: float32 workspace that mirrors what a block of
  threads holds **before** scattering results back to the output tensor.
- ``prepare_chunk_indices`` / ``iter_packed_bt_chunks`` encode the same **launch grid** as
  Triton: one logical program per ``(sequence, chunk_index)`` pair, including **partial** tail
  chunks (``span < BT``) with zero-padding like ``boundary_check``.

``safe_exp`` matches ``fla_vendor.utils.safe_exp`` (Triton): ``exp(x)`` where ``x <= 0``, else
``0``. Used for pairwise gate factors ``exp(g_i - g_j)`` so non-causal pairs do not contribute.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch


def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """
    Build the **varlen chunk launch table** (same as ``fla_vendor.utils.prepare_chunk_indices``).

    **Global input:** ``cu_seqlens`` shape ``[N+1]`` with cumulative starts of packed sequences.

    **Output:** shape ``[num_chunks, 2]``, dtype long, on the same device as ``cu_seqlens``.
    Row ``r`` is ``(i_n, i_t)`` where:

    - ``i_n`` = which sequence in the batch (0 .. N-1),
    - ``i_t`` = chunk index **within that sequence** (0 .. ceil(seq_len/chunk_size)-1).

    Rows are concatenated in order over all sequences—this is the iteration order Triton uses
    when ``IS_VARLEN`` is true. Partial last chunks are **included** (one row per chunk tile).
    """
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    nc = (lens + chunk_size - 1) // chunk_size
    # indices: flat list of **within-sequence** chunk indices 0,1,..,n0-1, 0,1,..,n1-1, ...
    parts = [torch.arange(int(n), device=cu_seqlens.device, dtype=torch.long) for n in nc.tolist()]
    indices = torch.cat(parts, dim=0) if parts else cu_seqlens.new_empty(0, dtype=torch.long)
    # seq_ids: which sequence each row belongs to (increment at each restart of chunk index at 0).
    seq_ids = (indices == 0).cumsum(0) - 1
    # Column 0 = sequence id i_n; column 1 = chunk index i_t within that sequence.
    return torch.stack([seq_ids, indices], dim=1).to(cu_seqlens)


def iter_packed_bt_chunks(
    *,
    cu_seqlens: torch.Tensor | None,
    total_t: int,
    bt: int,
    chunk_indices: torch.Tensor | None,
) -> Iterator[tuple[int, int, int]]:
    """
    Iterate chunk tiles in **Triton program order** for kernels that use fixed ``BT × …`` tiles.

    Yields ``(bos, i_tc, span)``:

    - ``bos`` — **global** offset in the packed time dimension where the current sequence starts.
    - ``i_tc`` — chunk index **within** that sequence (the ``i_t`` in ``chunk_indices``).
    - ``span`` — valid timesteps in this tile: ``min(BT, seq_end - (bos + i_tc*BT))``, so
      ``span < BT`` for a **partial** final chunk.

    **Global slice** written/read by that program: ``times [bos + i_tc*BT, bos + i_tc*BT + span)``.

    When ``cu_seqlens is None``, there is one sequence of length ``total_t`` starting at 0, and
    ``bos`` is always 0 (matches non-varlen Triton with batch stride in the kernel).
    """
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
            # Remaining timesteps in this sequence after skipping i_tc full BT blocks: clip to BT.
            span = min(bt, t_seg - i_tc * bt)
            yield bos, i_tc, span


def safe_exp_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Elementwise: ``exp(x)`` if ``x <= 0``, else ``0`` (Triton ``safe_exp``).

    **Shape:** same as ``x`` (broadcasting preserved). Used so ``exp(g_i - g_j)`` is zero for
    non-causal or masked pairs where the exponent would be positive.
    """
    return torch.where(x <= 0, torch.exp(x), torch.zeros_like(x))


def k_head_index(i_h: int, num_heads: int, num_k_heads: int) -> int:
    """
    GQA head map: output head ``i_h`` (0 .. H-1) → key/value head index ``i_h // (H // Hg)``.

    **Global tensors** ``k``, ``w`` use this to pick the correct head slice along ``Hg``.
    """
    return i_h // (num_heads // num_k_heads)


def tensor_r2_score(reference: torch.Tensor, prediction: torch.Tensor) -> float:
    """
    Coefficient of determination :math:`R^2` with ``reference`` as the ground truth (e.g. Triton).

    Uses the standard definition :math:`1 - \\mathrm{SS}_{\\mathrm{res}} / \\mathrm{SS}_{\\mathrm{tot}}`.
    If ``SS_tot`` is negligible (near-constant reference), returns ``1.0`` when residuals are tiny.
    """
    ref = reference.detach().float().reshape(-1)
    pred = prediction.detach().float().reshape(-1)
    ss_res = torch.sum((ref - pred) ** 2)
    mean_ref = ref.mean()
    ss_tot = torch.sum((ref - mean_ref) ** 2)
    if float(ss_tot.item()) < 1e-20:
        return 1.0 if float(ss_res.item()) < 1e-12 else 0.0
    return float((1.0 - ss_res / ss_tot).item())


def relative_rmse(reference: torch.Tensor, prediction: torch.Tensor) -> float:
    """
    :math:`\\mathrm{RMSE}(\\mathrm{ref}, \\mathrm{pred}) / \\sqrt{\\mathbb{E}[\\mathrm{ref}^2]}`.

    Scale-invariant vs the reference magnitude (Triton output).
    """
    ref = reference.detach().float().reshape(-1)
    pred = prediction.detach().float().reshape(-1)
    rmse = torch.sqrt(torch.mean((ref - pred) ** 2))
    denom = torch.sqrt(torch.mean(ref**2)).clamp(min=1e-30)
    return float((rmse / denom).item())
