"""
CPU-only PyTorch references matching ``verify_dynamic_bsnd.ref_*`` (same math).

This module imports only ``torch`` / ``numpy`` and ``._common`` — **not** ``dynamic_kernel_libs``
or ``pto_dynamic_common``. Importing ``verify_dynamic_bsnd`` pulls in Ascend kernel compilation
and can block for a long time; ``verify_torch_emulation_pto`` uses these refs instead.
"""

from __future__ import annotations

import torch

from ._common import seq_ranges as _seq_ranges, total_chunks


def _safe_exp(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0, torch.exp(x), torch.zeros_like(x))


def ref_cumsum(g: torch.Tensor, cs: int, cu_seqlens=None):
    B, T, Hd = g.shape
    g32, out = g.float(), torch.zeros_like(g, dtype=torch.float32)
    for bos, eos in _seq_ranges(T, cu_seqlens):
        for j in range(0, eos - bos, cs):
            s, e = bos + j, min(bos + j + cs, eos)
            out[:, s:e, :] = g32[:, s:e, :].cumsum(dim=1)
    return out


def ref_kkt(k: torch.Tensor, beta: torch.Tensor, g_cumsum: torch.Tensor, cs: int, cu_seqlens=None):
    B, T, Hd, Dd = k.shape
    out = torch.zeros(B, T, Hd, cs, device=k.device, dtype=torch.float32)
    kf, bf, gf = k.float(), beta.float(), g_cumsum.float()
    for bos, eos in _seq_ranges(T, cu_seqlens):
        for j in range(0, eos - bos, cs):
            s, e = bos + j, min(bos + j + cs, eos)
            v = e - s
            for h in range(Hd):
                kc, gc = kf[0, s:e, h, :], gf[0, s:e, h]
                blk = (kc @ kc.T) * _safe_exp(gc[:, None] - gc[None, :]) * bf[0, s:e, h, None]
                mask = torch.arange(v, device=blk.device)[:, None] > torch.arange(v, device=blk.device)[None, :]
                out[0, s:e, h, :v] = blk * mask.float()
    return out


def ref_wy(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g_cumsum: torch.Tensor,
    cs: int,
    cu_seqlens=None,
):
    B, T, Hd, Kd = k.shape
    w = torch.zeros(B, T, Hd, Kd, device=k.device, dtype=torch.float32)
    u = torch.zeros(B, T, Hd, v.shape[-1], device=k.device, dtype=torch.float32)
    kf, vf, bf, Af, gf = k.float(), v.float(), beta.float(), A.float(), g_cumsum.float()
    for bos, eos in _seq_ranges(T, cu_seqlens):
        for j in range(0, eos - bos, cs):
            s, e = bos + j, min(bos + j + cs, eos)
            valid = e - s
            for h in range(Hd):
                Ab = Af[0, s:e, h, :valid]
                gc = gf[0, s:e, h]
                vb = vf[0, s:e, h, :] * bf[0, s:e, h, None]
                kb = kf[0, s:e, h, :] * bf[0, s:e, h, None] * torch.exp(gc)[:, None]
                u[0, s:e, h, :] = Ab @ vb
                w[0, s:e, h, :] = Ab @ kb
    return w.to(k.dtype), u.to(v.dtype)


def ref_chunk_h(k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g_cumsum: torch.Tensor, cs: int, cu_seqlens=None):
    B, T, Hd, Dd = k.shape
    kf, wf, uf, gf = k.float(), w.float(), u.float(), g_cumsum.float()
    ranges = _seq_ranges(T, cu_seqlens)
    N = len(ranges)
    cu_t = torch.tensor(cu_seqlens) if isinstance(cu_seqlens, list) else cu_seqlens
    tc = total_chunks(N, T, cs, cu_t)
    h_out = torch.zeros(tc, Hd, Dd, Dd, device=k.device, dtype=torch.float32)
    v_new = torch.zeros_like(uf)
    final = torch.zeros(N, Hd, Dd, Dd, device=k.device, dtype=torch.float32)
    ci_base = 0
    for si, (bos, eos) in enumerate(ranges):
        nc = (eos - bos + cs - 1) // cs
        for h in range(Hd):
            S = torch.zeros(Dd, Dd, device=k.device, dtype=torch.float32)
            for ci in range(nc):
                s, e = bos + ci * cs, min(bos + (ci + 1) * cs, eos)
                gc = gf[0, s:e, h]
                gl = gc[e - s - 1]
                h_out[ci_base + ci, h] = S.clone()
                vc = uf[0, s:e, h, :] - wf[0, s:e, h, :] @ S
                v_new[0, s:e, h, :] = vc
                kv = kf[0, s:e, h, :].T @ (vc * torch.exp(gl - gc)[:, None])
                S = torch.exp(gl) * S + kv
            final[si, h] = S
        ci_base += nc
    return h_out, v_new, final


def _qk_gate_pto(gc: torch.Tensor) -> torch.Tensor:
    d = gc[:, None] - gc[None, :]
    return torch.exp(torch.minimum(d, torch.zeros_like(d)))


def _ref_chunk_o_gated(q, k, v_new, h_states, g_cumsum, cs, cu_seqlens, gate_fn):
    B, T, Hd, Dd = q.shape
    qf, kf, vf, gf = q.float(), k.float(), v_new.float(), g_cumsum.float()
    o = torch.zeros_like(qf)
    ranges = _seq_ranges(T, cu_seqlens)
    ci_base = 0
    for bos, eos in ranges:
        nc = (eos - bos + cs - 1) // cs
        for h in range(Hd):
            for ci in range(nc):
                s, e = bos + ci * cs, min(bos + (ci + 1) * cs, eos)
                vlen = e - s
                qc, kc, vc, gc = (
                    qf[0, s:e, h, :],
                    kf[0, s:e, h, :],
                    vf[0, s:e, h, :],
                    gf[0, s:e, h],
                )
                inter = (qc @ h_states[ci_base + ci, h]) * torch.exp(gc)[:, None]
                qk = qc @ kc.T
                mask = torch.arange(vlen, device=qk.device)[:, None] >= torch.arange(
                    vlen, device=qk.device
                )[None, :]
                gate = gate_fn(gc)
                o[0, s:e, h, :] = inter + (qk * gate * mask.float()) @ vc
        ci_base += nc
    return o


def ref_chunk_o(q, k, v_new, h_states, g_cumsum, cs, cu_seqlens=None):
    return _ref_chunk_o_gated(
        q, k, v_new, h_states, g_cumsum, cs, cu_seqlens, gate_fn=_qk_gate_pto
    )
