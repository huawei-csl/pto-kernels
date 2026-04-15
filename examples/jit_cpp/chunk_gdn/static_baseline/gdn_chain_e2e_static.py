"""
End-to-end GDN using static PTO kernels (tilelang_codegen extracts) + solve_tril.

Matches the pipeline in tilelang-ascend ``opt_gdn_full.py``:
  cumsum -> KKT -> solve_tril -> wy_fast -> chunk_h -> chunk_o

``solve_tril`` for C==128 uses ``(I+A)^{-1}`` with strict-lower A from KKT.
We implement that via ``pto_tri_inv_rec_unroll`` (upper triangular U = A^T), same as
``inv(I+A^T)`` transposed = ``inv(I+A)``. If ``pto_kernels`` is not importable, falls
back to batched ``torch.linalg.inv`` (mathematically identical).

Reference: ``ref_seq_gdn`` from ``opt_gdn_full.py`` (sequential formulation).

Fixed shapes must match the extracted ``*_kernel.cpp`` specializations:
  B=16, H=16, L=16384, DK=128, DV=128, C=128.
"""
from __future__ import annotations

import ctypes
import os
import sys

import torch
import torch.nn.functional as F

import pto_static_common  # noqa: F401 — env validation
from static_kernel_libs import (
    lib_chunk_cumsum,
    lib_chunk_h,
    lib_chunk_o,
    lib_scaled_dot_kkt,
    lib_wy_fast,
)

torch_npu = torch.npu  # noqa: F401

# Must match static kernel cpp
B, H, L, DK, DV, C = 16, 16, 16384, 128, 128, 128
CHUNK_NUM = (L + C - 1) // C
BV_NUM = (DV + DV - 1) // DV

_PTO_KERNELS_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
_PTO_PYTHON = os.path.join(_PTO_KERNELS_REPO, "python")
if os.path.isdir(_PTO_PYTHON) and _PTO_PYTHON not in sys.path:
    sys.path.insert(0, _PTO_PYTHON)


def _try_import_pto_tri_inv():
    try:
        from pto_kernels import pto_tri_inv_rec_unroll  # type: ignore

        return pto_tri_inv_rec_unroll
    except Exception:
        return None


pto_tri_inv_rec_unroll = _try_import_pto_tri_inv()


def ref_seq_gdn(q, k, v, g, beta):
    """Sequential GDN reference (from ``opt_gdn_full.py``)."""
    g = torch.exp(g)
    q = q.float()
    k = k.float()
    v = v.float()
    beta = beta.float()
    batch, h, l_, dk = q.shape
    dv = v.shape[-1]
    s = torch.zeros((batch, h, dv, dk), device=q.device, dtype=torch.float)
    o = torch.empty((batch, h, l_, dv), device=q.device, dtype=torch.float)
    i_ = torch.eye(dk, device=q.device, dtype=torch.float).view(1, 1, dk, dk)
    for t in range(0, l_):
        q_t = q[:, :, t, :]
        k_t = k[:, :, t, :]
        v_t = v[:, :, t, :]
        beta_t = beta[:, :, t].view(batch, h, 1, 1)
        g_t = g[:, :, t].view(batch, h, 1, 1)
        kkt = k_t.unsqueeze(-1) * k_t.unsqueeze(-2)
        vkt = v_t.unsqueeze(-1) * k_t.unsqueeze(-2)
        a_t = g_t * (i_ - beta_t * kkt)
        term_1 = torch.matmul(s, a_t)
        term_2 = beta_t * vkt
        s = term_1 + term_2
        o[:, :, t, :] = torch.einsum("bhpq,bhq->bhp", s, q_t)
    return o.to(torch.float16)


def solve_tril_inv_lower(a: torch.Tensor, idt: torch.Tensor) -> torch.Tensor:
    """
    O = (I + A)^{-1} with A strict lower per C×C block along L.
    ``a``: [B,H,L,C] fp16 — rows of each block; ``idt``: unused (identity implicit).

    PTO path: ``pto_tri_inv_rec_unroll(U)`` with ``U = A^T`` (upper), then transpose.
    Fallback: float64 CPU ``inv(I+A)`` for numerical stability (matches test_tri_inv).
    """
    del idt  # TileLang passes I; PTO builds I_neg internally
    b_, h_, l_, c_ = a.shape
    assert l_ % c_ == 0
    chunk = l_ // c_
    # [B*H*chunk, C, C] — rows of each KKT block; enforce strict lower (fp16 noise on diag).
    blocks = a.view(b_, h_, chunk, c_, c_).reshape(b_ * h_ * chunk, c_, c_)
    blocks = torch.tril(blocks, diagonal=-1)
    if pto_tri_inv_rec_unroll is not None:
        u = blocks.transpose(-2, -1).contiguous().to(torch.float16)
        inv_upper = pto_tri_inv_rec_unroll(u.npu(), is_bsnd_format=False)
        torch.npu.synchronize()
        o = inv_upper.transpose(-2, -1).to(dtype=torch.float16, device=a.device)
    else:
        # CPU float32 inverse: I + A with A strict lower is unit lower-triangular; well-conditioned.
        blk = blocks.float().cpu()
        m_ = torch.eye(c_, dtype=torch.float32) + blk
        o = torch.linalg.inv(m_).to(torch.float16).to(device=a.device)
    return o.reshape(b_, h_, l_, c_)


def run_chain(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_log: torch.Tensor,
    beta: torch.Tensor,
):
    """Run full static kernel chain; returns ``o`` [B,H,L,DV] fp16."""
    stream = torch.npu.current_stream()._as_parameter_

    def vp(p):
        return ctypes.c_void_p(p)

    # 1) cumsum on logsigmoid g
    g_sum = torch.empty((B, H, L), device=q.device, dtype=torch.float32)
    lib_chunk_cumsum().call(vp(g_log.data_ptr()), vp(g_sum.data_ptr()), stream)
    torch.npu.synchronize()

    # 2) KKT
    msk1 = torch.tril(torch.ones((C, C), device=q.device), diagonal=-1).to(torch.float32)
    workspace_kkt = torch.zeros((B, H, L, C), device=q.device, dtype=torch.float16)
    a = torch.empty((B, H, L, C), device=q.device, dtype=torch.float16)
    lib_scaled_dot_kkt().call(
        vp(k.data_ptr()),
        vp(beta.data_ptr()),
        vp(g_sum.data_ptr()),
        vp(msk1.data_ptr()),
        vp(workspace_kkt.data_ptr()),
        vp(a.data_ptr()),
        stream,
    )
    torch.npu.synchronize()

    # 3) solve_tril
    idt = torch.eye(C, device=q.device, dtype=torch.float32)
    a_sol = solve_tril_inv_lower(a, idt)

    # 4) wy_fast
    workspace_a1 = torch.zeros((B, H, L, C), device=q.device, dtype=torch.float16)
    workspace_a2 = torch.zeros((B, H, L, C), device=q.device, dtype=torch.float16)
    w = torch.empty((B, H, L, DK), device=q.device, dtype=torch.float16)
    u = torch.empty((B, H, L, DV), device=q.device, dtype=torch.float16)
    lib_wy_fast().call(
        vp(k.data_ptr()),
        vp(v.data_ptr()),
        vp(beta.data_ptr()),
        vp(g_sum.data_ptr()),
        vp(a_sol.data_ptr()),
        vp(workspace_a1.data_ptr()),
        vp(workspace_a2.data_ptr()),
        vp(w.data_ptr()),
        vp(u.data_ptr()),
        stream,
    )
    torch.npu.synchronize()

    # 5) chunk_h
    workspace_1 = torch.zeros((B * H * BV_NUM, C, DV), device=q.device, dtype=torch.float16)
    workspace_2 = torch.zeros((B * H * BV_NUM, C, DK), device=q.device, dtype=torch.float16)
    workspace_3 = torch.zeros((B * H * BV_NUM, DK, DV), device=q.device, dtype=torch.float16)
    workspace_4 = torch.zeros((B * H * BV_NUM, DK, DV), device=q.device, dtype=torch.float16)
    s = torch.zeros((B, H, CHUNK_NUM, DK, DV), device=q.device, dtype=torch.float16)
    nv = torch.empty((B, H, L, DV), device=q.device, dtype=torch.float16)
    fs = torch.empty((B, H, DK, DV), device=q.device, dtype=torch.float16)
    lib_chunk_h().call(
        vp(k.data_ptr()),
        vp(w.data_ptr()),
        vp(u.data_ptr()),
        vp(g_sum.data_ptr()),
        vp(workspace_1.data_ptr()),
        vp(workspace_2.data_ptr()),
        vp(workspace_3.data_ptr()),
        vp(workspace_4.data_ptr()),
        vp(s.data_ptr()),
        vp(nv.data_ptr()),
        vp(fs.data_ptr()),
        stream,
    )
    torch.npu.synchronize()

    # 6) chunk_o
    nblk = B * H * CHUNK_NUM
    workspace_o1 = torch.zeros((nblk, C, C), device=q.device, dtype=torch.float16)
    workspace_o2 = torch.zeros((nblk, C, DV), device=q.device, dtype=torch.float16)
    workspace_o3 = torch.zeros((nblk, C, C), device=q.device, dtype=torch.float16)
    msk2 = torch.tril(torch.ones((C, C), device=q.device), diagonal=0).to(torch.float32)
    o = torch.empty((B, H, L, DV), device=q.device, dtype=torch.float16)
    lib_chunk_o().call(
        vp(q.data_ptr()),
        vp(k.data_ptr()),
        vp(nv.data_ptr()),
        vp(s.data_ptr()),
        vp(g_sum.data_ptr()),
        vp(msk2.data_ptr()),
        vp(workspace_o1.data_ptr()),
        vp(workspace_o2.data_ptr()),
        vp(workspace_o3.data_ptr()),
        vp(o.data_ptr()),
        stream,
    )
    torch.npu.synchronize()
    return o


def main():
    torch.manual_seed(0)
    torch.npu.set_device("npu:0")

    q = torch.randn((B, H, L, DK), device="npu", dtype=torch.float16)
    k = torch.randn((B, H, L, DK), device="npu", dtype=torch.float16)
    v = torch.randn((B, H, L, DV), device="npu", dtype=torch.float16)
    q, k = F.normalize(q, dim=-1, p=2), F.normalize(k, dim=-1, p=2)
    g_raw = torch.randn((B, H, L), device="npu", dtype=torch.float32)
    g_log = F.logsigmoid(g_raw)
    beta = torch.rand((B, H, L), device="npu", dtype=torch.float16)

    o = run_chain(q, k, v, g_log, beta)
    ref_o = ref_seq_gdn(q, k, v, g_log, beta)

    torch.testing.assert_close(o.cpu(), ref_o.cpu(), rtol=1e-3, atol=1e-3)
    mode = "pto_tri_inv_rec_unroll" if pto_tri_inv_rec_unroll is not None else "torch.linalg.inv"
    print(f"GDN e2e static chain OK (solve_tril: {mode}).")


if __name__ == "__main__":
    main()
