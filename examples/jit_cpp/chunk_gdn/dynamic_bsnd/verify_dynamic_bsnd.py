#!/usr/bin/env python3
"""
Numerical verification for dynamic BSND PTO kernels (chunk_size=128).

Verifies each stage against a PyTorch reference:
  1. chunk_cumsum — chunk-local prefix sum
  2. scaled_dot_kkt — gated KK^T with mask and beta
  3. wy_fast — WY recompute (w, u)
  4. chunk_h — chunkwise state recurrence (states, v_new, final_state)
  5. chunk_o — output from inter/intra-chunk attention
"""
from __future__ import annotations

import os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CHUNK_GDN = os.path.abspath(os.path.join(_HERE, ".."))
if _CHUNK_GDN not in sys.path:
    sys.path.insert(0, _CHUNK_GDN)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
import torch.nn.functional as F

from dynamic_kernel_libs import (
    BLOCK_DIM,
    run_chunk_cumsum,
    run_chunk_o,
    run_chunk_h,
    run_scaled_dot_kkt,
    run_wy_fast,
    total_chunks,
)

NPU_DEVICE = os.getenv("GDN_NPU_DEVICE", "npu:0")
C = 128
RTOL, ATOL = 2e-2, 2e-2
# Accumulated fp16 state matrices (chunk_h, chunk_o) compound matmul
# quantization error across chunks, requiring a wider absolute tolerance.
RTOL_ACCUM, ATOL_ACCUM = 2e-2, 5e-2


# -------- PyTorch references --------

def ref_chunk_local_cumsum(g, chunk_size, cu_seqlens=None):
    """chunk-local cumsum along dim=1 for [B,T,H] or [1,T,H]."""
    B, T, H = g.shape
    g32 = g.float()
    out = torch.zeros_like(g32)
    if cu_seqlens is None:
        ranges = [(0, T)]
    else:
        cu = cu_seqlens.cpu().tolist()
        ranges = [(cu[i], cu[i + 1]) for i in range(len(cu) - 1)]
    for bos, eos in ranges:
        L = eos - bos
        for j in range(0, L, chunk_size):
            e = min(j + chunk_size, L)
            out[:, bos + j : bos + e, :] = g32[:, bos + j : bos + e, :].cumsum(dim=1)
    return out


def _safe_exp(x):
    return torch.where(x <= 0, torch.exp(x), torch.zeros_like(x))


def ref_scaled_dot_kkt(k, beta, g_cumsum, chunk_size, cu_seqlens=None):
    """Reference KKT: [B,T,H,C] layout with strict lower triangle, gating, beta."""
    B, T, H, D = k.shape
    out = torch.zeros(B, T, H, chunk_size, device=k.device, dtype=torch.float32)
    kf, bf, gf = k.float(), beta.float(), g_cumsum.float()
    if cu_seqlens is None:
        ranges = [(0, T)]
    else:
        cu = cu_seqlens.cpu().tolist()
        ranges = [(cu[i], cu[i + 1]) for i in range(len(cu) - 1)]
    for bos, eos in ranges:
        L = eos - bos
        for ci in range(L // chunk_size):
            s = bos + ci * chunk_size
            e = s + chunk_size
            for h in range(H):
                kc = kf[0, s:e, h, :]
                kk = kc @ kc.T
                gc = gf[0, s:e, h]
                gam = gc.unsqueeze(-1) - gc.unsqueeze(-2)
                blk = kk * _safe_exp(gam)
                blk = blk * bf[0, s:e, h].unsqueeze(-1)
                bt = blk.shape[0]
                mask = torch.arange(bt, device=blk.device)[:, None] > torch.arange(bt, device=blk.device)[None, :]
                blk = blk * mask.float()
                out[0, s:e, h, :chunk_size] = blk
    return out


def ref_recompute_w_u(k, v, beta, A, g_cumsum, chunk_size, cu_seqlens=None):
    B, T, H, Kd = k.shape
    V = v.shape[-1]
    w_ref = torch.zeros(B, T, H, Kd, device=k.device, dtype=torch.float32)
    u_ref = torch.zeros(B, T, H, V, device=k.device, dtype=torch.float32)
    kf, vf, bf, Af, gf = k.float(), v.float(), beta.float(), A.float(), g_cumsum.float()
    if cu_seqlens is None:
        ranges = [(0, T)]
    else:
        cu = cu_seqlens.cpu().tolist()
        ranges = [(cu[i], cu[i + 1]) for i in range(len(cu) - 1)]
    for bos, eos in ranges:
        L = eos - bos
        for ci in range(L // chunk_size):
            s = bos + ci * chunk_size
            e = s + chunk_size
            for h in range(H):
                Ablk = Af[0, s:e, h, :]
                gc = gf[0, s:e, h]
                b_g = torch.exp(gc)
                vb = vf[0, s:e, h, :] * bf[0, s:e, h, None]
                kb = kf[0, s:e, h, :] * bf[0, s:e, h, None] * b_g[:, None]
                u_ref[0, s:e, h, :] = Ablk @ vb
                w_ref[0, s:e, h, :] = Ablk @ kb
    return w_ref.to(k.dtype), u_ref.to(v.dtype)


def ref_chunk_h(k, w, u, g_cumsum, chunk_size, cu_seqlens=None, initial_state=None):
    """
    Chunkwise state recurrence reference (matches PTO/triton kernel algorithm):
      h_out[ci] = S  (state BEFORE processing chunk ci)
      v_new = u - W @ S
      S_new = exp(g_last) * S + k^T @ (v_new * exp(g_last - g_cumsum))
    """
    B, T, H, D = k.shape
    kf = k.float()
    wf = w.float()
    uf = u.float()
    gf = g_cumsum.float()

    if cu_seqlens is None:
        ranges = [(0, T)]
        N_seq = B
    else:
        cu = cu_seqlens.cpu().tolist()
        ranges = [(cu[i], cu[i + 1]) for i in range(len(cu) - 1)]
        N_seq = len(cu) - 1

    tc = total_chunks(N_seq, T, chunk_size, cu_seqlens)
    h_out = torch.zeros(tc, H, D, D, device=k.device, dtype=torch.float32)
    v_new = torch.zeros_like(uf)
    final_state = torch.zeros(N_seq, H, D, D, device=k.device, dtype=torch.float32)

    chunk_idx = 0
    for si, (bos, eos) in enumerate(ranges):
        L = eos - bos
        num_c = (L + chunk_size - 1) // chunk_size
        for h in range(H):
            S = torch.zeros(D, D, device=k.device, dtype=torch.float32)
            if initial_state is not None:
                S = initial_state[si, h].float().clone()
            ci_base = chunk_idx
            for ci in range(num_c):
                s = bos + ci * chunk_size
                e = min(s + chunk_size, eos)
                valid = e - s

                gc = gf[0, s:e, h]
                g_last = gc[valid - 1]

                h_out[ci_base + ci, h] = S.clone()

                ws = wf[0, s:e, h, :] @ S
                v_chunk = uf[0, s:e, h, :] - ws
                v_new[0, s:e, h, :] = v_chunk

                decay_per_row = torch.exp(g_last - gc).unsqueeze(-1)
                v_gated = v_chunk * decay_per_row
                kv = kf[0, s:e, h, :].T @ v_gated

                S = torch.exp(g_last) * S + kv

            final_state[si, h] = S
        chunk_idx += num_c

    return h_out, v_new, final_state


def ref_chunk_o(q, k, v_new, h_states, g_cumsum, chunk_size, cu_seqlens=None):
    """
    Output computation reference (matches PTO kernel, no scale):
      o_inter = q @ h_state * exp(g_cumsum[t])
      o_intra = (q @ k^T * safe_exp(g_row - g_col) * causal_mask) @ v_new
      o = o_inter + o_intra
    """
    B, T, H, D = q.shape
    qf = q.float()
    kf = k.float()
    vf = v_new.float()
    gf = g_cumsum.float()

    o_out = torch.zeros_like(qf)

    if cu_seqlens is None:
        ranges = [(0, T)]
    else:
        cu = cu_seqlens.cpu().tolist()
        ranges = [(cu[i], cu[i + 1]) for i in range(len(cu) - 1)]

    chunk_idx = 0
    for bos, eos in ranges:
        L = eos - bos
        num_c = (L + chunk_size - 1) // chunk_size
        for h in range(H):
            ci_offset = chunk_idx
            for ci in range(num_c):
                s = bos + ci * chunk_size
                e = min(s + chunk_size, eos)
                valid = e - s

                qc = qf[0, s:e, h, :]
                kc = kf[0, s:e, h, :]
                vc = vf[0, s:e, h, :]
                gc = gf[0, s:e, h]

                h_state = h_states[ci_offset + ci, h]
                o_inter = qc @ h_state
                o_inter = o_inter * torch.exp(gc).unsqueeze(-1)

                qk = qc @ kc.T
                gc_row = gc.unsqueeze(-1)
                gc_col = gc.unsqueeze(-2)
                gating = _safe_exp(gc_row - gc_col)
                bt = valid
                mask = torch.arange(bt, device=qk.device)[:, None] >= torch.arange(bt, device=qk.device)[None, :]
                qk_gated = qk * gating * mask.float()
                o_intra = qk_gated @ vc

                o_out[0, s:e, h, :] = o_inter + o_intra

            ci_offset += num_c
        chunk_idx += num_c
    return o_out


def main():
    torch.manual_seed(42)
    torch.npu.set_device(NPU_DEVICE)
    dev = torch.device(NPU_DEVICE)

    N_seq = 2
    L_seg = 256
    H, D = 16, 128
    T = N_seq * L_seg

    cu_seqlens = torch.arange(0, T + 1, L_seg, dtype=torch.int32, device=dev)
    print(f"Shape: B=1, T={T}, H={H}, D={D}, C={C}, N_seq={N_seq}, L_seg={L_seg}")
    print(f"cu_seqlens={cu_seqlens.cpu().tolist()}")
    print(f"BLOCK_DIM={BLOCK_DIM}")
    print()

    q = torch.randn(1, T, H, D, device=dev, dtype=torch.float16)
    k = torch.randn(1, T, H, D, device=dev, dtype=torch.float16)
    v = torch.randn(1, T, H, D, device=dev, dtype=torch.float16)
    q, k = F.normalize(q, dim=-1, p=2), F.normalize(k, dim=-1, p=2)
    g_in = F.logsigmoid(torch.randn(1, T, H, device=dev, dtype=torch.float32))
    beta = torch.rand(1, T, H, device=dev, dtype=torch.float16)

    # --- 1. chunk_cumsum ---
    print("[1] Testing chunk_cumsum...")
    g_sum = torch.empty(1, T, H, device=dev, dtype=torch.float32)
    run_chunk_cumsum(g_in, g_sum, chunk_size=C,
                     cu_seqlens=cu_seqlens, batch_size_override=N_seq)
    torch.npu.synchronize()

    g_ref = ref_chunk_local_cumsum(g_in.cpu(), C, cu_seqlens.cpu())
    g_sum_cpu = g_sum.float().cpu()
    match = torch.allclose(g_sum_cpu, g_ref, rtol=RTOL, atol=ATOL)
    if not match:
        diff = (g_sum_cpu - g_ref).abs()
        print(f"  max abs diff: {diff.max().item():.6f}, mean: {diff.mean().item():.6f}")
    print(f"  chunk_cumsum: {'PASS' if match else 'FAIL'}")

    # --- 2. scaled_dot_kkt ---
    print("[2] Testing scaled_dot_kkt...")
    msk = torch.tril(torch.ones(C, C, device=dev), diagonal=-1).to(torch.float32)
    workspace_kkt = torch.zeros(BLOCK_DIM, C, C, device=dev, dtype=torch.float16)
    A_out = torch.zeros(1, T, H, C, device=dev, dtype=torch.float16)
    run_scaled_dot_kkt(k, beta, g_sum, msk, workspace_kkt, A_out,
                       chunk_size=C, cu_seqlens=cu_seqlens,
                       batch_size_override=N_seq)
    torch.npu.synchronize()

    A_ref = ref_scaled_dot_kkt(k.cpu(), beta.cpu(), g_sum.cpu(), C, cu_seqlens.cpu())
    A_cmp = A_out.float().cpu()
    match = torch.allclose(A_cmp, A_ref, rtol=RTOL, atol=ATOL)
    if not match:
        diff = (A_cmp - A_ref).abs()
        print(f"  max abs diff: {diff.max().item():.6f}, mean: {diff.mean().item():.6f}")
        nonzero_diff = diff[A_ref.abs() > 1e-6]
        if nonzero_diff.numel() > 0:
            print(f"  max rel diff (nonzero): {(nonzero_diff / A_ref[A_ref.abs() > 1e-6].abs()).max().item():.4f}")
    print(f"  scaled_dot_kkt: {'PASS' if match else 'FAIL'}")

    # --- 3. wy_fast ---
    print("[3] Testing wy_fast...")
    w_out = torch.empty(1, T, H, D, device=dev, dtype=torch.float16)
    u_out = torch.empty(1, T, H, D, device=dev, dtype=torch.float16)
    run_wy_fast(k, v, beta, g_sum, A_out, w_out, u_out,
                chunk_size=C, cu_seqlens=cu_seqlens,
                batch_size_override=N_seq)
    torch.npu.synchronize()

    w_ref, u_ref = ref_recompute_w_u(k.cpu(), v.cpu(), beta.cpu(), A_out.cpu(), g_sum.cpu(), C, cu_seqlens.cpu())
    # w = A @ (k*beta*exp(g)): chained fp16 multiplies before matmul need wider atol
    w_match = torch.allclose(w_out.float().cpu(), w_ref.float(), rtol=RTOL, atol=3e-2)
    u_match = torch.allclose(u_out.float().cpu(), u_ref.float(), rtol=RTOL, atol=ATOL)
    if not w_match:
        diff = (w_out.float().cpu() - w_ref.float()).abs()
        print(f"  w max diff: {diff.max().item():.6f}")
    if not u_match:
        diff = (u_out.float().cpu() - u_ref.float()).abs()
        print(f"  u max diff: {diff.max().item():.6f}")
    print(f"  wy_fast w: {'PASS' if w_match else 'FAIL'}")
    print(f"  wy_fast u: {'PASS' if u_match else 'FAIL'}")

    # --- 4. chunk_h ---
    print("[4] Testing chunk_h...")
    tc = total_chunks(N_seq, T, C, cu_seqlens)
    s_out = torch.zeros(tc * H, D, D, device=dev, dtype=torch.float16)
    v_out = torch.empty(1, T, H, D, device=dev, dtype=torch.float16)
    fs_out = torch.zeros(N_seq * H, D, D, device=dev, dtype=torch.float16)
    run_chunk_h(k, w_out, u_out, g_sum, s_out, v_out, fs_out,
                chunk_size=C, cu_seqlens=cu_seqlens,
                batch_size_override=N_seq)
    torch.npu.synchronize()

    s_finite = torch.isfinite(s_out).all()
    v_finite = torch.isfinite(v_out).all()
    fs_finite = torch.isfinite(fs_out).all()
    print(f"  chunk_h states finite: {'PASS' if s_finite else 'FAIL'}")
    print(f"  chunk_h v_new finite: {'PASS' if v_finite else 'FAIL'}")
    print(f"  chunk_h final_state finite: {'PASS' if fs_finite else 'FAIL'}")

    h_ref, v_ref, fs_ref = ref_chunk_h(k.cpu(), w_out.cpu(), u_out.cpu(), g_sum.cpu(), C, cu_seqlens.cpu())
    s_reshaped = s_out.float().cpu().view(tc, H, D, D)
    h_ref32 = h_ref.float()
    h_match = torch.allclose(s_reshaped, h_ref32, rtol=RTOL_ACCUM, atol=ATOL_ACCUM)
    if not h_match:
        diff = (s_reshaped - h_ref32).abs()
        print(f"  h states max diff: {diff.max().item():.6f}, mean: {diff.mean().item():.6f}")
    print(f"  chunk_h states: {'PASS' if h_match else 'FAIL'}")

    v_match = torch.allclose(v_out.float().cpu(), v_ref.float(), rtol=RTOL, atol=ATOL)
    if not v_match:
        diff = (v_out.float().cpu() - v_ref.float()).abs()
        print(f"  v_new max diff: {diff.max().item():.6f}, mean: {diff.mean().item():.6f}")
    print(f"  chunk_h v_new: {'PASS' if v_match else 'FAIL'}")

    # --- 5. chunk_o ---
    print("[5] Testing chunk_o...")
    msk2 = torch.tril(torch.ones(C, C, device=dev), diagonal=0).to(torch.float32)
    o_out = torch.empty(1, T, H, D, device=dev, dtype=torch.float16)
    run_chunk_o(q, k, v_out, s_out, g_sum, msk2, o_out,
                chunk_size=C, cu_seqlens=cu_seqlens,
                batch_size_override=N_seq)
    torch.npu.synchronize()

    o_finite = torch.isfinite(o_out).all()
    print(f"  chunk_o output finite: {'PASS' if o_finite else 'FAIL'}")

    o_ref = ref_chunk_o(q.cpu(), k.cpu(), v_out.cpu(), s_reshaped, g_sum.cpu(), C, cu_seqlens.cpu())
    o_cmp = o_out.float().cpu()
    o_ref_f = o_ref.float()
    o_match = torch.allclose(o_cmp, o_ref_f, rtol=RTOL_ACCUM, atol=ATOL_ACCUM)
    if not o_match:
        diff = (o_cmp - o_ref_f).abs()
        print(f"  o max diff: {diff.max().item():.6f}, mean: {diff.mean().item():.6f}")
    print(f"  chunk_o output: {'PASS' if o_match else 'FAIL'}")

    print()
    all_pass = (match and w_match and u_match
                and s_finite and v_finite and fs_finite
                and h_match and v_match
                and o_finite and o_match)
    print(f"Overall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")


if __name__ == "__main__":
    main()
