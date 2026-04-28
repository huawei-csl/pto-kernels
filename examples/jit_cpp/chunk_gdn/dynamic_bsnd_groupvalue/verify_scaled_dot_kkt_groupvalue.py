#!/usr/bin/env python3
"""
Numerical verification for ``scaled_dot_kkt`` with GQA (Hg key heads, H value heads).

Reference matches FLA/Triton: ``head_g = head // (H // Hg)`` for which ``K`` row is used.

Usage::
  cd .../chunk_gdn/dynamic_bsnd_groupvalue
  python3 verify_scaled_dot_kkt_groupvalue.py --device npu:7
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from dataclasses import dataclass

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import torch
import torch.nn.functional as F

from dynamic_kernel_libs import (
    BLOCK_DIM,
    _transpose_beta,
    _transpose_g,
    run_scaled_dot_kkt,
)

C = 128
D = 128
HG = 16

RTOL_CHECK = 1e-2
ATOL_CHECK = 1e-5
MAX_RMSE_OVER_MEAN_ABS = 0.05
MIN_R2_FALLBACK = 0.99
HARD_FAIL_THRESHOLD = 1.0


def _seq_ranges(T, cu_seqlens=None):
    if cu_seqlens is None:
        return [(0, T)]
    cu = cu_seqlens.tolist() if hasattr(cu_seqlens, "tolist") else cu_seqlens
    return [(cu[i], cu[i + 1]) for i in range(len(cu) - 1)]


def ref_cumsum(g, cs, cu_seqlens=None):
    """Chunk-local cumulative gates — same as ``verify_dynamic_bsnd.ref_cumsum``."""
    B, T, Hd = g.shape
    g32, out = g.float(), torch.zeros_like(g, dtype=torch.float32)
    for bos, eos in _seq_ranges(T, cu_seqlens):
        for j in range(0, eos - bos, cs):
            s, e = bos + j, min(bos + j + cs, eos)
            out[:, s:e, :] = g32[:, s:e, :].cumsum(dim=1)
    return out


def _safe_exp(x):
    return torch.where(x <= 0, torch.exp(x), torch.zeros_like(x))


def ref_kkt_group(k, beta, g_cumsum, cs, cu_seqlens=None):
    """``k``: [B,T,Hg,D]; ``beta``, ``g_cumsum``: [B,T,H] — value heads."""
    B, T, Hg, Dd = k.shape
    H = beta.shape[2]
    assert H % Hg == 0
    grp = H // Hg
    out = torch.zeros(B, T, H, cs, device=k.device, dtype=torch.float32)
    kf, bf, gf = k.float(), beta.float(), g_cumsum.float()
    for bos, eos in _seq_ranges(T, cu_seqlens):
        for j in range(0, eos - bos, cs):
            s, e = bos + j, min(bos + j + cs, eos)
            v = e - s
            for h in range(H):
                hg = h // grp
                kc = kf[0, s:e, hg, :]
                gc = gf[0, s:e, h]
                blk = (
                    (kc @ kc.T)
                    * _safe_exp(gc[:, None] - gc[None, :])
                    * bf[0, s:e, h, None]
                )
                mask = torch.arange(v, device=blk.device)[:, None] > torch.arange(
                    v, device=blk.device
                )[None, :]
                out[0, s:e, h, :v] = blk * mask.float()
    return out


def r2_score_vs_ref(y_ref: torch.Tensor, y_pred: torch.Tensor) -> float:
    ref = np.asarray(y_ref.detach().cpu().numpy().ravel(), dtype=np.float64)
    pred = np.asarray(y_pred.detach().cpu().numpy().ravel(), dtype=np.float64)
    ss_res = float(np.sum((ref - pred) ** 2))
    ss_tot = float(np.sum((ref - np.mean(ref)) ** 2))
    if ss_tot <= 1e-30 * max(ref.size, 1):
        return float("nan")
    return 1.0 - ss_res / ss_tot


# ─── Same case list spirit as verify_dynamic_bsnd_groupvalue ───


@dataclass
class TestCase:
    label: str
    cu_seqlens_list: list[int] | None
    T: int


def _align_cu_seqlens(raw: list[int], cs: int) -> list[int]:
    aligned = [0]
    for i in range(1, len(raw) - 1):
        val = ((raw[i] + cs - 1) // cs) * cs
        if val <= aligned[-1]:
            val = aligned[-1] + cs
        aligned.append(val)
    total = max(raw[-1], aligned[-1] + cs)
    total = ((total + cs - 1) // cs) * cs
    aligned.append(total)
    return aligned


def _rand_cu_seqlens(n_seq: int, total: int, rng: random.Random) -> list[int]:
    if n_seq == 1:
        return [0, total]
    bnd = sorted(rng.sample(range(1, total), n_seq - 1))
    return [0] + bnd + [total]


def build_test_cases() -> list[TestCase]:
    c = []
    c.append(TestCase("fixed T=128 (1 chunk)", None, 128))
    c.append(TestCase("fixed T=256 (2 chunks)", None, 256))
    c.append(TestCase("fixed T=385 (tail 1)", None, 385))
    c.append(TestCase("fixed T=512 (4 chunks)", None, 512))
    c.append(TestCase("varlen [128,256]", [0, 128, 384], 384))
    c.append(TestCase("varlen [129,255] (tails)", [0, 129, 384], 384))
    rng = random.Random(42)
    for n_seq, total in [(3, 768)]:
        raw = _rand_cu_seqlens(n_seq, total, rng)
        aligned = _align_cu_seqlens(raw, C)
        c.append(TestCase(
            f"varlen {n_seq} seqs random T={aligned[-1]}",
            aligned, aligned[-1],
        ))
    return c


def run_case(tc: TestCase, dev: torch.device, H: int):
    checks_ok = []
    T = tc.T
    if tc.cu_seqlens_list is not None:
        cu = torch.tensor(tc.cu_seqlens_list, dtype=torch.int32, device=dev)
        N_seq = len(tc.cu_seqlens_list) - 1
    else:
        cu = None
        N_seq = 1

    torch.manual_seed(42)
    torch.npu.manual_seed(42)
    k = F.normalize(torch.randn(1, T, HG, D, device=dev, dtype=torch.float16), dim=-1, p=2)
    beta = torch.rand(1, T, H, device=dev, dtype=torch.float16)
    cu_cpu = cu.cpu() if cu is not None else None
    stream = torch.npu.current_stream()._as_parameter_
    g_in = F.logsigmoid(torch.randn(1, T, H, device=dev, dtype=torch.float32))
    g_sum = ref_cumsum(g_in.cpu(), C, cu_cpu).to(device=dev)
    g_t = _transpose_g(g_sum)
    beta_t = _transpose_beta(beta)

    msk = torch.tril(torch.ones(C, C, device=dev), diagonal=-1).float()
    A_out = torch.zeros(1, T, H, C, device=dev, dtype=torch.float16)

    torch.npu.synchronize()
    run_scaled_dot_kkt(
        k, beta, g_sum, msk, None, A_out,
        stream=stream,
        g_t=g_t, beta_t=beta_t,
        chunk_size=C,
        cu_seqlens=cu,
        batch_size_override=N_seq,
        key_heads=HG,
    )
    torch.npu.synchronize()

    ref = ref_kkt_group(k.cpu(), beta.cpu(), g_sum.cpu(), C, cu_cpu)

    diff = (A_out.float().cpu() - ref).abs()
    mx = diff.max().item()
    expected = ref
    actual = A_out.float().cpu()
    bound = ATOL_CHECK + RTOL_CHECK * expected.abs()
    pass_allclose = bool((diff <= bound).all().item())
    ref_1d = expected.float().flatten()
    mean_abs_ref = float(ref_1d.abs().mean().item())
    rmse = float(torch.sqrt((diff.float().flatten() ** 2).mean()).item())
    ratio = rmse / max(mean_abs_ref, 1e-15)
    r2 = r2_score_vs_ref(expected, actual)
    std_ref = float(ref_1d.std().item())
    if mean_abs_ref < 1e-9:
        pass_stats = rmse < 5e-4
    elif std_ref < 1e-12:
        pass_stats = ratio <= MAX_RMSE_OVER_MEAN_ABS
    else:
        pass_stats = (
            ratio <= MAX_RMSE_OVER_MEAN_ABS
            and np.isfinite(r2)
            and r2 >= MIN_R2_FALLBACK
        )
    ok = (pass_allclose or pass_stats) and mx <= HARD_FAIL_THRESHOLD
    checks_ok.append(ok)
    return all(checks_ok)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=os.getenv("GDN_NPU_DEVICE", "npu:0"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--H-list", default="16,32,48,64",
                        help="Comma-separated value head counts (Hg fixed at 16)")
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    dev = torch.device(args.device)
    heads_list = [int(x.strip()) for x in args.H_list.split(",")]

    cases = (
        [TestCase("quick fixed T=128", None, 128)]
        if args.quick
        else build_test_cases()
    )

    print(f"Device {args.device}  H in {heads_list}  Hg={HG}  D={D}  C={C}  BLOCK_DIM={BLOCK_DIM}")
    ok_all = True
    for H in heads_list:
        assert H % HG == 0, f"H={H} must be divisible by Hg={HG}"
        print(f"\n--- Value heads H={H} ---")
        for i, tc in enumerate(cases):
            t0 = time.time()
            ok = run_case(tc, dev, H)
            dt = time.time() - t0
            status = "PASS" if ok else "FAIL"
            if not ok:
                ok_all = False
            print(f"  [{i+1}/{len(cases)}] {status}  {tc.label}  ({dt:.2f}s)")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
