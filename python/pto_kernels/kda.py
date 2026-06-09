"""Python wrappers for KDA PTO kernels exposed via pybind11.

Mirrors the interface of megagdn_pto.kda_kernel_libs so that
tests/test_kda_single_kernels.py can import run_chunk_h_kda from here.
"""

from __future__ import annotations

import torch

from pto_kernels import pto_chunk_h_kda


def _ensure_int32(cu_seqlens: torch.Tensor | None) -> torch.Tensor:
    if cu_seqlens is None:
        return torch.zeros(1, dtype=torch.int32)
    return cu_seqlens.to(torch.int32)


def run_chunk_h_kda(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g_cs: torch.Tensor,
    s_snapshots_out: torch.Tensor,
    v_corr_out: torch.Tensor,
    *,
    stream=None,
    chunk_size: int = 128,
    cu_seqlens: torch.Tensor | None = None,
    batch_size_override: int | None = None,
    block_dim: int | None = None,
) -> None:
    """Sequential recurrent state pass for KDA.

    Fills ``s_snapshots_out`` and ``v_corr_out`` in-place on the NPU.

    Args:
        k:               ``[B, T, HV, K]`` float16, keys (BSND).
        w:               ``[B, T, HV, K]`` float16, from wy_kda (BSND).
        u:               ``[B, T, HV, V]`` float16, from wy_kda (BSND).
        g_cs:            ``[B, T, HV, K]`` float16, within-chunk cumulative gate sum.
        s_snapshots_out: ``[total_chunks, HV, K, V]`` float16 (output).
        v_corr_out:      ``[B, T, HV, V]`` float16 (output).
        stream:          NPU stream handle (ignored; stream managed internally).
        chunk_size:      Tokens per chunk C; must match the compiled kernel.
        cu_seqlens:      ``int32`` cumulative sequence lengths for packed varlen input.
        batch_size_override: Number of sequences (use with ``cu_seqlens``).
        block_dim:       Unused; AI-Core count is detected inside the kernel.
    """
    assert k.dtype == torch.float16, f"k must be fp16, got {k.dtype}"
    assert w.dtype == torch.float16, f"w must be fp16, got {w.dtype}"
    assert u.dtype == torch.float16, f"u must be fp16, got {u.dtype}"
    assert g_cs.dtype == torch.float16, f"g_cs must be fp16, got {g_cs.dtype}"
    assert s_snapshots_out.dtype == torch.float16
    assert v_corr_out.dtype == torch.float16

    T = k.shape[1]
    batch = k.shape[0] if batch_size_override is None else batch_size_override

    # Permute K and G to head-major [B, HV, T, K] for the kernel's memory layout.
    k_t = k.permute(0, 2, 1, 3).contiguous()
    g_cs_t = g_cs.permute(0, 2, 1, 3).contiguous()

    cu32 = _ensure_int32(cu_seqlens)
    if cu32.numel() > 1:
        cu32 = cu32.to(k.device)

    pto_chunk_h_kda(
        k_t,
        w.contiguous(),
        u,
        g_cs_t,
        s_snapshots_out,
        v_corr_out,
        cu32,
        batch,
        T,
        T,
    )
