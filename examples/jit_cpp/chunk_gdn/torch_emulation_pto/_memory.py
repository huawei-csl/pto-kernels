"""
Explicit **data-movement** stand-ins for PTO DMA / MTE1 ops used in ``dynamic_bsnd/*_kernel.cpp``:

- ``TLOAD`` / ``TSTORE`` — GM ↔ UB / L1 (MTE2 / MTE3).
- ``TMOV`` — element-wise copy in UB/L1 (Vec).
- ``TADD`` — element-wise add in UB (Vec); listed for ``chunk_cumsum`` parity.
- ``TEXTRACT`` — L1 sub-tile → L0A / L0B (MTE1), used before ``TMATMUL``.
- ``TRESHAPE`` — NZ↔ZN reinterpretation of an L1 tile (no HBM traffic); we use ``.transpose``.

Tutorial cross-ref: ``pto-dsl/.../step1_baseline_numpy_sim.py`` (``a_l0[:,:] = a_l1[:, ...]``).

Memory roles:

- **GM** — global memory (a ``torch.Tensor`` view).
- **UB** — Vec SRAM (we allocate a tensor and copy slices).
- **L1** — Cube tile cache (``*_l1`` tensors).
- **L0A / L0B / L0C** — operands / accumulator; matmul accumulates in fp32 L0C.

Each function is a **synchronous** copy or pad. Real hardware uses async MTE2/MTE3/MTE1 pipes
with ``set_flag`` / ``wait_flag``; we omit sync but keep the **read/write sites** explicit.

Higher-level helpers include ``tload_bsnd_chunk_rows_to_l1`` (BSND row ``TLOAD`` into ``[C×D]`` L1),
``tload_gm_fp32_dd_to_l1_half`` (state ``S`` tile), ``tmov_l1_half_rows`` / ``tmov_l1_half_dc_cols``,
``tmov_l1_cc_gate_mask_from_l0c`` (Vec QK gate), ``alloc_l0_stripes_gemm_v0`` / ``alloc_l0c_fp32`` for
**reused** L0 tiles during ``gemm_v0_accum_fp16``.

Tile size (comments in call sites)
----------------------------------
SRAM tile footprint: ``numel × sizeof(elem)`` bytes; **KiB** = bytes / 1024.
fp16 = 2 B, fp32 = 4 B. Example **GDN** defaults ``C=128``, ``D=128``: ``[C×D]`` fp16 → 32 KiB.
"""

from __future__ import annotations

import torch


def tile_kib(numel: int, elem_bytes: int) -> float:
    """Return tile size in KiB (for docstrings / comments)."""
    return numel * elem_bytes / 1024.0


def alloc_l0_stripes_gemm_v0(
    max_m: int,
    max_n: int,
    k_tile: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pre-allocated **L0A** / **L0B** stripes reused across every ``K`` step of ``gemm_v0`` (hardware-style).

    Shapes: ``[max_m, k_tile]``, ``[k_tile, max_n]`` — each step uses slices ``[:m,:kt]`` and ``[:kt,:n]``.

    **KiB (fp16):** L0A **max_m·k_tile/512**, L0B **k_tile·max_n/512** (e.g. **32 KiB** each @ 128×128).
    """
    l0a = torch.empty((max_m, k_tile), device=device, dtype=dtype)
    l0b = torch.empty((k_tile, max_n), device=device, dtype=dtype)
    return l0a, l0b


def alloc_l0c_fp32(max_m: int, max_n: int, *, device: torch.device) -> torch.Tensor:
    """
    Pre-allocated **L0C** fp32 accumulator ``[max_m, max_n]``.

    **KiB:** **max_m·max_n/256** (e.g. **64 KiB** @ 128×128).
    """
    return torch.empty((max_m, max_n), device=device, dtype=torch.float32)


def tmov(dst: torch.Tensor, src: torch.Tensor) -> None:
    """
    ``TMOV(dst, src)`` — bitwise/element-wise copy (UB or L1 tiles).

    C++: ``dst = src`` with matching tile shapes (see ``chunk_cumsum_kernel`` row copies,
    ``wy_fast`` / Vec staging). Broadcasts are **not** PTO-correct; keep shapes aligned.
    """
    dst.copy_(src.to(dtype=dst.dtype))


def tadd(dst: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> None:
    """``TADD(dst, a, b)`` — ``dst = a + b`` (Vec UB), used in chunk-local prefix scan."""
    dst.copy_((a + b).to(dtype=dst.dtype))


def treshape_l1_nz_to_zn(l1: torch.Tensor) -> torch.Tensor:
    """
    ``TRESHAPE(l1_zn, l1_nz)`` — logical transpose for Cube (NZ→ZN fractal).

    On device this is a **metadata** change; numerically we use ``l1.transpose(-2, -1)``.
    ``scaled_dot_kkt_kernel`` uses this so ``K^T`` feeds L0B without a second GM load.
    """
    return l1.transpose(-2, -1)


def textract_l1_to_l0a_contracting(
    l0a_dst: torch.Tensor,
    a_l1: torch.Tensor,
    *,
    k_begin: int,
    k_end: int,
) -> None:
    """
    ``TEXTRACT(l0a, A, 0, kBlock)`` when ``A`` is the **left** GEMM operand (non-transpose).

    Copies ``A[:, k_begin:k_end]`` into the L0A tile (contracting columns of ``A``).
    Matches ``gemm_v0`` non-transpose-A path: ``TEXTRACT(l0a, A, 0, kL0Idx * kL0Size)``.
    """
    l0a_dst.copy_(a_l1[:, k_begin:k_end].to(dtype=l0a_dst.dtype))


def textract_l1_to_l0b_contracting(
    l0b_dst: torch.Tensor,
    b_l1: torch.Tensor,
    *,
    k_begin: int,
    k_end: int,
) -> None:
    """
    ``TEXTRACT(l0b, B, kBlock, 0)`` when ``B`` is the **right** operand (non-transpose).

    Copies ``B[k_begin:k_end, :]`` into L0B (contracting **rows** of ``B``).
    """
    l0b_dst.copy_(b_l1[k_begin:k_end, :].to(dtype=l0b_dst.dtype))


def htc_align(num_heads: int) -> int:
    """Head tile columns rounded up to 8 floats (32 B), matching ``chunk_cumsum_kernel``."""
    return ((num_heads + 7) // 8) * 8


def tload_gm_to_ub_g_chunk(
    g_ub: torch.Tensor,
    g_gm: torch.Tensor,
    *,
    valid: int,
    num_heads: int,
    htc: int,
) -> None:
    """
    ``TLOAD(g_load, g_gm)`` in ``chunk_cumsum_kernel.cpp``:

    ``g_ub[:valid, :num_heads] = g_gm[chunk rows]``; caller owns ``g_ub`` shape ``[C, HTC]``.
    """
    g_ub[:valid, :num_heads] = g_gm[:valid, :num_heads].to(g_ub.dtype)


def tfillpad_ub_g_inplace(g_ub: torch.Tensor, *, valid: int, chunk_size: int, num_heads: int, htc: int) -> None:
    """
    ``TFILLPAD_INPLACE(g_pad, g_load)`` — zero rows ``valid:`` and cols ``num_heads:HTC``.
    """
    if valid < chunk_size:
        g_ub[valid:chunk_size, :].zero_()
    if num_heads < htc:
        g_ub[:, num_heads:htc].zero_()


def tstore_ub_to_gm_gsum(
    g_sum_gm: torch.Tensor,
    s_ub: torch.Tensor,
    *,
    chunk_start: int,
    valid: int,
    num_heads: int,
) -> None:
    """
    ``TSTORE(gs_gm, s_store)`` — UB → GM for the prefix-sum output tile.
    """
    g_sum_gm[chunk_start : chunk_start + valid, :num_heads] = s_ub[:valid, :num_heads].to(g_sum_gm.dtype)


def alloc_l1_cd(
    chunk_size: int,
    hidden_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Uninitialized L1 stand-in ``[C, D]`` (NZ layout emulated as row-major for math).

    **Size:** ``C×D×2`` B (fp16) → ``C×D/512`` KiB (e.g. **32 KiB** when ``C=D=128``).
    """
    return torch.empty((chunk_size, hidden_size), device=device, dtype=dtype)


def tload_bsnd_chunk_rows_to_l1(
    l1: torch.Tensor,
    gm_bsnd: torch.Tensor,
    *,
    token_start: int,
    valid_rows: int,
    head_idx: int,
    hidden_size: int,
) -> None:
    """
    ``TLOAD(_l1, _gm)`` — BSND ``[T, H, D]`` chunk rows into L1 ``[C, D]`` (NZ stand-in).

    Used for ``Q``, ``K``, ``V``, ``W`` in ``chunk_o_kernel`` / ``chunk_h_kernel`` / ``scaled_dot_kkt_kernel``.
    """
    for i in range(valid_rows):
        t = token_start + i
        l1[i, :] = gm_bsnd[t, head_idx, :].to(l1.dtype)


# Back-compat alias (older name referenced ``K`` only).
tload_k_bsnd_chunk_to_k_l1 = tload_bsnd_chunk_rows_to_l1


def tload_gm_fp32_dd_to_l1_half(
    s_l1: torch.Tensor,
    s_gm_fp32: torch.Tensor,
) -> None:
    """
    ``TLOAD`` fp32 ``S`` ``[D×D]`` from GM into L1 fp16 (``chunk_h`` / ``chunk_o`` state tile).

    Numerically ``s_l1.copy_(s_gm_fp32.half())``.
    """
    s_l1.copy_(s_gm_fp32.to(dtype=s_l1.dtype))


def tmov_l1_half_rows(
    l1_dst: torch.Tensor,
    src_rows: torch.Tensor,
    *,
    valid_rows: int,
) -> None:
    """
    ``TMOV`` / row broadcast — copy ``src_rows`` ``[valid, D]`` into top of ``l1_dst`` ``[C, D]``.
    """
    l1_dst[:valid_rows, :].copy_(src_rows.to(dtype=l1_dst.dtype))


def tmov_l1_half_dc_cols(
    k_l1: torch.Tensor,
    kt_rowmajor: torch.Tensor,
    *,
    valid_cols: int,
) -> None:
    """
    ``TMOV`` — ``K̃`` as ``[D×C]`` L1: ``k_l1[:, :valid] = kt_rowmajor.T`` (``kt`` is ``[valid, D]``).
    """
    k_l1[:, :valid_cols].copy_(kt_rowmajor.T.to(dtype=k_l1.dtype))


def tfillpad_k_l1_tail_rows(l1: torch.Tensor, *, valid_rows: int, chunk_size: int) -> None:
    """``TFILLPAD(_l1, _l1)`` when ``valid_rows < ChunkSize`` — zero pad bottom rows."""
    if valid_rows < chunk_size:
        l1[valid_rows:chunk_size, :].zero_()


def tstore_l0c_to_workspace_kk_half(
    workspace_kk: torch.Tensor,
    a_l0_fp32: torch.Tensor,
    *,
    slot: int,
    chunk_square: int,
) -> None:
    """
    ``TSTORE(_gm, _l0)`` after KKT — fp32 L0C cast to fp16 in GM workspace for Vec consumption.
    ``workspace_kk`` is the flat per-slot buffer of length ``chunk_square`` (``C*C``).
    """
    h = a_l0_fp32.half()
    workspace_kk.view(-1)[: chunk_square].copy_(h.view(-1))


def tload_workspace_kk_half_to_ub_rows(
    a_ub_half: torch.Tensor,
    workspace_kk: torch.Tensor,
    *,
    row_begin: int,
    n_rows: int,
    chunk_size: int,
) -> None:
    """
    Vec ``TLOAD(_ld, _gm)`` — load ``[n_rows, C]`` stripe of KK^T from workspace into UB.
    ``a_ub_half`` shape ``[HalfChunk, C]`` or subset rows.
    """
    w = workspace_kk.view(chunk_size, chunk_size)
    a_ub_half[:n_rows, :].copy_(w[row_begin : row_begin + n_rows, :])


def tstore_ub_half_to_gm_a_rows(
    a_gm: torch.Tensor,
    a_ub_half: torch.Tensor,
    *,
    token_begin: int,
    head_idx: int,
    n_rows: int,
    n_cols: int,
    chunk_size: int,
) -> None:
    """
    ``TSTORE(_gm, _st)`` — write gated ``A`` sub-block to BSND ``A`` tensor ``[T,H,C]``.
    """
    for i in range(n_rows):
        t = token_begin + i
        a_gm[t, head_idx, :n_cols] = a_ub_half[i, :n_cols].float()
        if n_cols < chunk_size:
            a_gm[t, head_idx, n_cols:chunk_size] = 0


def gemm_v0_accum_fp16(
    a_l1: torch.Tensor,
    b_l1: torch.Tensor,
    *,
    transpose_a: bool = False,
    transpose_b: bool = False,
    k_tile: int = 128,
    l0c_out: torch.Tensor | None = None,
    l0a_buf: torch.Tensor | None = None,
    l0b_buf: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    ``chunk_h_kernel.cpp`` / ``chunk_o_kernel.cpp`` ``gemm_v0``:

    Effective operands ``A_eff = A`` or ``A.T``, ``B_eff = B`` or ``B.T`` (``transpose_*``
    match PTO ``TRESHAPE`` on L1 before ``TEXTRACT``).

    Each K-tile step:

    - ``TEXTRACT`` → ``l0a`` = ``A_eff[:, k0:k1]`` (``textract_l1_to_l0a_contracting``),
    - ``TEXTRACT`` → ``l0b`` = ``B_eff[k0:k1, :]`` (``textract_l1_to_l0b_contracting``),
    - ``TMATMUL`` / ``TMATMUL_ACC`` into fp32 L0C.

    ``K @ K^T`` uses ``transpose_b=True`` with ``b_l1 = k_l1`` so ``B_eff = k_l1.T``.

    Optional **pre-allocated** ``l0c_out``, ``l0a_buf``, ``l0b_buf`` mirror fixed on-chip tiles
    reused each GEMM (see ``alloc_l0_stripes_gemm_v0`` / ``alloc_l0c_fp32``).
    """
    a_eff = a_l1.transpose(-2, -1) if transpose_a else a_l1
    b_eff = b_l1.transpose(-2, -1) if transpose_b else b_l1
    m, kdim = a_eff.shape
    kdim2, n = b_eff.shape
    assert kdim == kdim2
    device = a_l1.device
    dtype = a_l1.dtype
    if l0c_out is None:
        # L0C fp32 [m×n] — **m·n/256** KiB; fallback path when caller did not pre-allocate
        out = torch.zeros(m, n, dtype=torch.float32, device=device)
    else:
        out = l0c_out[:m, :n]
        out.zero_()
    if l0a_buf is not None:
        assert l0a_buf.shape[0] >= m and l0a_buf.shape[1] >= k_tile
    if l0b_buf is not None:
        assert l0b_buf.shape[0] >= k_tile and l0b_buf.shape[1] >= n
    k0 = 0
    while k0 < kdim:
        k1 = min(k0 + k_tile, kdim)
        kt = k1 - k0
        if l0a_buf is None:
            # L0A fp16 stripe [m×kt] — ephemeral fallback (**m·kt/512** KiB at fp16)
            l0a = torch.empty((m, kt), device=device, dtype=dtype)
        else:
            l0a = l0a_buf[:m, :kt]
        if l0b_buf is None:
            # L0B fp16 stripe [kt×n] — ephemeral fallback (**kt·n/512** KiB at fp16)
            l0b = torch.empty((kt, n), device=device, dtype=dtype)
        else:
            l0b = l0b_buf[:kt, :n]
        textract_l1_to_l0a_contracting(l0a, a_eff, k_begin=k0, k_end=k1)
        textract_l1_to_l0b_contracting(l0b, b_eff, k_begin=k0, k_end=k1)
        out += l0a.float() @ l0b.float()
        k0 = k1
    if l0c_out is None:
        return out
    return l0c_out[:m, :n]


def tmov_l1_cc_gate_mask_from_l0c(
    qk_gated_l1: torch.Tensor,
    qk_l0_fp32: torch.Tensor,
    gate: torch.Tensor,
    mask: torch.Tensor,
    *,
    vlen: int,
) -> None:
    """
    Vec path after ``QK`` in L0C: apply gate + causal mask, ``TMOV`` / cast into ``qk_gated_l1`` ``[C×C]`` L1.
    """
    qk_gated_l1[:vlen, :vlen].copy_(
        (qk_l0_fp32[:vlen, :vlen] * gate * mask.to(dtype=qk_l0_fp32.dtype)).to(dtype=qk_gated_l1.dtype)
    )


def tmatmul_kkt_l1_to_l0c(
    k_l1: torch.Tensor,
    *,
    k_tile: int = 128,
    l0c_out: torch.Tensor | None = None,
    l0a_buf: torch.Tensor | None = None,
    l0b_buf: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Cube path ``K @ K^T`` (``scaled_dot_kkt_kernel``):

    ``TEXTRACT`` stripes from ``k_l1`` and ``TRESHAPE`` / ``K^T`` into L0A/L0B, then
    ``TMATMUL`` — same inner path as ``Q @ K^T`` with ``transpose_b=True``.
    """
    return gemm_v0_accum_fp16(
        k_l1,
        k_l1,
        transpose_b=True,
        k_tile=k_tile,
        l0c_out=l0c_out,
        l0a_buf=l0a_buf,
        l0b_buf=l0b_buf,
    )
