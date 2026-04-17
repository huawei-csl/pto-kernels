#!/usr/bin/env python3
"""
Numerical verification for dynamic BSND PTO kernels.

The script sweeps a deterministic list of fixed-length and variable-length
shape combinations chosen to cover chunk-boundary corner cases, and verifies
every PTO stage against a PyTorch reference:
  1. chunk_cumsum — chunk-local prefix sum
  2. scaled_dot_kkt — gated KK^T with mask and beta
  3. wy_fast — WY recompute (w, u)
  4. chunk_h — chunkwise state recurrence (states, v_new, final_state)
  5. chunk_o — output from inter/intra-chunk attention
"""
from __future__ import annotations

from dataclasses import dataclass
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
# chunk_o combines inter/intra-chunk matmuls with fp16 gating coefficients,
# accumulating up to ~0.08 max absolute error in outlier elements.
RTOL_ACCUM, ATOL_ACCUM = 2e-2, 8e-2


@dataclass(frozen=True)
class VerifyCase:
    name: str
    seq_lens: tuple[int, ...]
    use_cu_seqlens: bool
    num_heads: int = 16
    hidden_size: int = 128
    seed: int = 42

    @property
    def total_tokens(self) -> int:
        return sum(self.seq_lens)

    @property
    def num_sequences(self) -> int:
        return len(self.seq_lens)


def make_cu_seqlens(seq_lens: tuple[int, ...], device: torch.device) -> torch.Tensor:
    cu = [0]
    for length in seq_lens:
        cu.append(cu[-1] + int(length))
    return torch.tensor(cu, dtype=torch.int32, device=device)


def format_seq_lens(seq_lens: tuple[int, ...], *, max_items: int = 8) -> str:
    if len(seq_lens) <= max_items:
        return str(list(seq_lens))
    prefix = ", ".join(str(x) for x in seq_lens[: max_items // 2])
    suffix = ", ".join(str(x) for x in seq_lens[-(max_items // 2) :])
    return f"[{prefix}, ..., {suffix}]"


def build_cases() -> list[VerifyCase]:
    return [
        VerifyCase(
            name="varlen_single_seq_exact_one_chunk",
            seq_lens=(128,),
            use_cu_seqlens=True,
            seed=101,
        ),
        VerifyCase(
            name="varlen_single_seq_exact_two_chunks",
            seq_lens=(256,),
            use_cu_seqlens=True,
            seed=102,
        ),
        VerifyCase(
            name="varlen_single_seq_tail_four_chunks",
            seq_lens=(385,),
            use_cu_seqlens=True,
            seed=103,
        ),
        VerifyCase(
            name="varlen_all_chunk_aligned_multi_seq",
            seq_lens=(128, 256, 384),
            use_cu_seqlens=True,
            seed=104,
        ),
        VerifyCase(
            name="varlen_short_and_tail_mix",
            seq_lens=(1, 17, 128, 129, 255),
            use_cu_seqlens=True,
            seed=105,
        ),
        VerifyCase(
            name="varlen_boundary_ladder_8seq_1024tok",
            seq_lens=(1, 63, 64, 65, 127, 128, 129, 447),
            use_cu_seqlens=True,
            seed=106,
        ),
        VerifyCase(
            name="varlen_boundary_ladder_13seq_1536tok",
            seq_lens=(1, 17, 31, 32, 33, 95, 127, 128, 129, 191, 192, 193, 367),
            use_cu_seqlens=True,
            seed=107,
        ),
        VerifyCase(
            name="varlen_boundary_ladder_17seq_2048tok",
            seq_lens=(1, 2, 31, 32, 33, 63, 64, 65, 95, 127, 128, 129, 191, 319, 255, 256, 257),
            use_cu_seqlens=True,
            seed=108,
        ),
    ]


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
    """Reference KKT with strict lower triangle, gating, beta, and tail chunks."""
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
        for chunk_start in range(0, L, chunk_size):
            s = bos + chunk_start
            e = min(s + chunk_size, eos)
            valid = e - s
            for h in range(H):
                kc = kf[0, s:e, h, :]
                kk = kc @ kc.T
                gc = gf[0, s:e, h]
                gam = gc.unsqueeze(-1) - gc.unsqueeze(-2)
                blk = kk * _safe_exp(gam)
                blk = blk * bf[0, s:e, h].unsqueeze(-1)
                mask = torch.arange(valid, device=blk.device)[:, None] > torch.arange(valid, device=blk.device)[None, :]
                blk = blk * mask.float()
                out[0, s:e, h, :valid] = blk
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
        for chunk_start in range(0, L, chunk_size):
            s = bos + chunk_start
            e = min(s + chunk_size, eos)
            valid = e - s
            for h in range(H):
                Ablk = Af[0, s:e, h, :valid]
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


def run_case(case: VerifyCase, dev: torch.device) -> bool:
    torch.manual_seed(case.seed)

    H = case.num_heads
    D = case.hidden_size
    T = case.total_tokens
    seq_lens = case.seq_lens
    cu_seqlens = make_cu_seqlens(seq_lens, dev) if case.use_cu_seqlens else None
    batch_override = case.num_sequences if case.use_cu_seqlens else None
    cu_cpu = cu_seqlens.cpu() if cu_seqlens is not None else None

    print(f"== Case: {case.name} ==")
    print(
        f"  B=1, T={T}, H={H}, D={D}, C={C}, "
        f"N_seq={case.num_sequences}, use_cu_seqlens={case.use_cu_seqlens}"
    )
    print(f"  seq_lens={format_seq_lens(seq_lens)}")
    if cu_cpu is not None:
        print(f"  cu_seqlens={cu_cpu.tolist()}")
    print(f"  BLOCK_DIM={BLOCK_DIM}")

    q = torch.randn(1, T, H, D, device=dev, dtype=torch.float16)
    k = torch.randn(1, T, H, D, device=dev, dtype=torch.float16)
    v = torch.randn(1, T, H, D, device=dev, dtype=torch.float16)
    q, k = F.normalize(q, dim=-1, p=2), F.normalize(k, dim=-1, p=2)
    g_in = F.logsigmoid(torch.randn(1, T, H, device=dev, dtype=torch.float32))
    beta = torch.rand(1, T, H, device=dev, dtype=torch.float16)

    q_cpu = q.cpu()
    k_cpu = k.cpu()
    v_cpu = v.cpu()
    g_in_cpu = g_in.cpu()
    beta_cpu = beta.cpu()

    # --- 1. chunk_cumsum ---
    print("[1] Testing chunk_cumsum...")
    g_sum = torch.empty(1, T, H, device=dev, dtype=torch.float32)
    run_chunk_cumsum(
        g_in,
        g_sum,
        chunk_size=C,
        cu_seqlens=cu_seqlens,
        batch_size_override=batch_override,
    )
    torch.npu.synchronize()

    g_ref = ref_chunk_local_cumsum(g_in_cpu, C, cu_cpu)
    g_sum_cpu = g_sum.float().cpu()
    cumsum_match = torch.allclose(g_sum_cpu, g_ref, rtol=RTOL, atol=ATOL)
    if not cumsum_match:
        diff = (g_sum_cpu - g_ref).abs()
        print(f"  max abs diff: {diff.max().item():.6f}, mean: {diff.mean().item():.6f}")
    print(f"  chunk_cumsum: {'PASS' if cumsum_match else 'FAIL'}")

    # --- 2. scaled_dot_kkt ---
    print("[2] Testing scaled_dot_kkt...")
    msk = torch.tril(torch.ones(C, C, device=dev), diagonal=-1).to(torch.float32)
    workspace_kkt = torch.zeros(BLOCK_DIM, C, C, device=dev, dtype=torch.float16)
    A_out = torch.zeros(1, T, H, C, device=dev, dtype=torch.float16)
    run_scaled_dot_kkt(
        k,
        beta,
        g_sum,
        msk,
        workspace_kkt,
        A_out,
        chunk_size=C,
        cu_seqlens=cu_seqlens,
        batch_size_override=batch_override,
    )
    torch.npu.synchronize()

    A_ref = ref_scaled_dot_kkt(k_cpu, beta_cpu, g_sum_cpu, C, cu_cpu)
    A_cpu = A_out.float().cpu()
    kkt_match = torch.allclose(A_cpu, A_ref, rtol=RTOL, atol=ATOL)
    if not kkt_match:
        diff = (A_cpu - A_ref).abs()
        print(f"  max abs diff: {diff.max().item():.6f}, mean: {diff.mean().item():.6f}")
        ref_nonzero = A_ref.abs() > 1e-6
        nonzero_diff = diff[ref_nonzero]
        if nonzero_diff.numel() > 0:
            rel = nonzero_diff / A_ref[ref_nonzero].abs()
            print(f"  max rel diff (nonzero): {rel.max().item():.4f}")
    print(f"  scaled_dot_kkt: {'PASS' if kkt_match else 'FAIL'}")

    # --- 3. wy_fast ---
    print("[3] Testing wy_fast...")
    w_out = torch.empty(1, T, H, D, device=dev, dtype=torch.float16)
    u_out = torch.empty(1, T, H, D, device=dev, dtype=torch.float16)
    run_wy_fast(
        k,
        v,
        beta,
        g_sum,
        A_out,
        w_out,
        u_out,
        chunk_size=C,
        cu_seqlens=cu_seqlens,
        batch_size_override=batch_override,
    )
    torch.npu.synchronize()

    w_ref, u_ref = ref_recompute_w_u(k_cpu, v_cpu, beta_cpu, A_cpu, g_sum_cpu, C, cu_cpu)
    w_cpu = w_out.float().cpu()
    u_cpu = u_out.float().cpu()
    # w = A @ (k*beta*exp(g)): chained fp16 multiplies before matmul need wider atol
    w_match = torch.allclose(w_cpu, w_ref.float(), rtol=RTOL, atol=5e-2)
    u_match = torch.allclose(u_cpu, u_ref.float(), rtol=RTOL, atol=ATOL)
    if not w_match:
        diff = (w_cpu - w_ref.float()).abs()
        print(f"  w max diff: {diff.max().item():.6f}, mean: {diff.mean().item():.6f}")
    if not u_match:
        diff = (u_cpu - u_ref.float()).abs()
        print(f"  u max diff: {diff.max().item():.6f}, mean: {diff.mean().item():.6f}")
    print(f"  wy_fast w: {'PASS' if w_match else 'FAIL'}")
    print(f"  wy_fast u: {'PASS' if u_match else 'FAIL'}")

    # --- 4. chunk_h ---
    print("[4] Testing chunk_h...")
    total_case_chunks = total_chunks(case.num_sequences, T, C, cu_seqlens)
    s_out = torch.zeros(total_case_chunks * H, D, D, device=dev, dtype=torch.float16)
    v_out = torch.empty(1, T, H, D, device=dev, dtype=torch.float16)
    fs_out = torch.zeros(case.num_sequences * H, D, D, device=dev, dtype=torch.float16)
    run_chunk_h(
        k,
        w_out,
        u_out,
        g_sum,
        s_out,
        v_out,
        fs_out,
        chunk_size=C,
        cu_seqlens=cu_seqlens,
        batch_size_override=batch_override,
    )
    torch.npu.synchronize()

    s_finite = bool(torch.isfinite(s_out).all().item())
    v_finite = bool(torch.isfinite(v_out).all().item())
    fs_finite = bool(torch.isfinite(fs_out).all().item())
    print(f"  chunk_h states finite: {'PASS' if s_finite else 'FAIL'}")
    print(f"  chunk_h v_new finite: {'PASS' if v_finite else 'FAIL'}")
    print(f"  chunk_h final_state finite: {'PASS' if fs_finite else 'FAIL'}")

    h_ref, v_ref, fs_ref = ref_chunk_h(k_cpu, w_cpu, u_cpu, g_sum_cpu, C, cu_cpu)
    s_reshaped = s_out.float().cpu().view(total_case_chunks, H, D, D)
    h_match = torch.allclose(s_reshaped, h_ref.float(), rtol=RTOL_ACCUM, atol=ATOL_ACCUM)
    if not h_match:
        diff = (s_reshaped - h_ref.float()).abs()
        print(f"  h states max diff: {diff.max().item():.6f}, mean: {diff.mean().item():.6f}")
    print(f"  chunk_h states: {'PASS' if h_match else 'FAIL'}")

    v_cpu = v_out.float().cpu()
    v_match = torch.allclose(v_cpu, v_ref.float(), rtol=RTOL, atol=ATOL)
    if not v_match:
        diff = (v_cpu - v_ref.float()).abs()
        print(f"  v_new max diff: {diff.max().item():.6f}, mean: {diff.mean().item():.6f}")
    print(f"  chunk_h v_new: {'PASS' if v_match else 'FAIL'}")

    fs_cpu = fs_out.float().cpu().view(case.num_sequences, H, D, D)
    fs_match = torch.allclose(fs_cpu, fs_ref.float(), rtol=RTOL_ACCUM, atol=ATOL_ACCUM)
    if not fs_match:
        diff = (fs_cpu - fs_ref.float()).abs()
        print(f"  final_state max diff: {diff.max().item():.6f}, mean: {diff.mean().item():.6f}")
    print(f"  chunk_h final_state: {'PASS' if fs_match else 'FAIL'}")

    # --- 5. chunk_o ---
    print("[5] Testing chunk_o...")
    msk2 = torch.tril(torch.ones(C, C, device=dev), diagonal=0).to(torch.float32)
    o_out = torch.empty(1, T, H, D, device=dev, dtype=torch.float16)
    run_chunk_o(
        q,
        k,
        v_out,
        s_out,
        g_sum,
        msk2,
        o_out,
        chunk_size=C,
        cu_seqlens=cu_seqlens,
        batch_size_override=batch_override,
    )
    torch.npu.synchronize()

    o_finite = bool(torch.isfinite(o_out).all().item())
    print(f"  chunk_o output finite: {'PASS' if o_finite else 'FAIL'}")

    o_ref = ref_chunk_o(q_cpu, k_cpu, v_cpu, s_reshaped, g_sum_cpu, C, cu_cpu)
    o_cpu = o_out.float().cpu()
    o_match = torch.allclose(o_cpu, o_ref.float(), rtol=RTOL_ACCUM, atol=ATOL_ACCUM)
    if not o_match:
        diff = (o_cpu - o_ref.float()).abs()
        print(f"  o max diff: {diff.max().item():.6f}, mean: {diff.mean().item():.6f}")
    print(f"  chunk_o output: {'PASS' if o_match else 'FAIL'}")

    case_pass = (
        cumsum_match
        and kkt_match
        and w_match
        and u_match
        and s_finite
        and v_finite
        and fs_finite
        and h_match
        and v_match
        and fs_match
        and o_finite
        and o_match
    )
    print(f"Case result: {'PASS' if case_pass else 'FAIL'}")
    print()
    return case_pass


def main():
    torch.npu.set_device(NPU_DEVICE)
    dev = torch.device(NPU_DEVICE)

    cases = build_cases()
    passed = 0

    print(f"Running {len(cases)} verification cases on {NPU_DEVICE}")
    print()
    for case in cases:
        if run_case(case, dev):
            passed += 1

    all_pass = passed == len(cases)
    print(f"Summary: {passed}/{len(cases)} cases passed.")
    print(f"Overall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
