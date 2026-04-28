#!/usr/bin/env python3
"""
Numerical verification for ``wy_fast`` with GQA grouping (Hg key heads, H value heads).

Uses synthetic ``A`` tiles (same layout as scaled-dot output per **value** head) and the same
case list as ``verify_dynamic_bsnd_groupvalue.py``. Reference matches FLA indexing:
``hg = h // (H // Hg)`` for ``K``.

Usage:
  cd .../chunk_gdn/dynamic_bsnd_groupvalue
  python3 verify_wy_fast_groupvalue.py --device npu:7
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

from dynamic_kernel_libs import BLOCK_DIM, run_wy_fast

C = 128
D = 128
HG = 16


def _transpose_g(g_sum):
    return g_sum.squeeze(0).t().contiguous()


def _transpose_beta(beta):
    return beta.squeeze(0).t().contiguous()


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


def ref_wy_group(k, v, beta, A, g_cumsum, cs, cu_seqlens=None):
    """``k``: [B,T,Hg,D]; ``v``: [B,T,H,D]; ``A``: [B,T,H,C]; gates/beta per value head."""
    B, T, Hg, Kd = k.shape
    H = v.shape[2]
    assert H % Hg == 0
    grp = H // Hg
    w = torch.zeros(B, T, H, Kd, device=k.device, dtype=torch.float32)
    u = torch.zeros(B, T, H, v.shape[-1], device=k.device, dtype=torch.float32)
    kf, vf, bf, Af, gf = k.float(), v.float(), beta.float(), A.float(), g_cumsum.float()
    for bos, eos in _seq_ranges(T, cu_seqlens):
        for j in range(0, eos - bos, cs):
            s, e = bos + j, min(bos + j + cs, eos)
            valid = e - s
            for h in range(H):
                hg = h // grp
                Ab = Af[0, s:e, h, :valid]
                gc = gf[0, s:e, h]
                vb = vf[0, s:e, h, :] * bf[0, s:e, h, None]
                kb = (
                    kf[0, s:e, hg, :]
                    * bf[0, s:e, h, None]
                    * torch.exp(gc)[:, None]
                )
                u[0, s:e, h, :] = Ab @ vb
                w[0, s:e, h, :] = Ab @ kb
    return w.to(k.dtype), u.to(v.dtype)


def r2_score_vs_ref(y_ref: torch.Tensor, y_pred: torch.Tensor) -> float:
    ref = np.asarray(y_ref.detach().cpu().numpy().ravel(), dtype=np.float64)
    pred = np.asarray(y_pred.detach().cpu().numpy().ravel(), dtype=np.float64)
    ss_res = float(np.sum((ref - pred) ** 2))
    ss_tot = float(np.sum((ref - np.mean(ref)) ** 2))
    if ss_tot <= 1e-30 * max(ref.size, 1):
        return float("nan")
    return 1.0 - ss_res / ss_tot


@dataclass
class TestCase:
    label: str
    cu_seqlens_list: list[int] | None
    T: int


def _cu_from_seqlens(seqlens: list[int]) -> list[int]:
    cu = [0]
    for slen in seqlens:
        cu.append(cu[-1] + slen)
    return cu


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
    c.append(TestCase("fixed T=1024 (8 chunks)", None, 1024))
    c.append(TestCase("varlen 1×128", [0, 128], 128))
    c.append(TestCase("varlen 1×256", [0, 256], 256))
    c.append(TestCase("varlen [256,256]", [0, 256, 512], 512))
    c.append(TestCase("varlen [128,256]", [0, 128, 384], 384))
    c.append(TestCase("varlen 1×200 (tail 72)", [0, 200], 200))
    rng = random.Random(42)
    for n_seq, total in [(3, 768), (7, 1792)]:
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
    v = torch.randn(1, T, H, D, device=dev, dtype=torch.float16)
    beta = torch.rand(1, T, H, device=dev, dtype=torch.float16)
    A = torch.randn(1, T, H, C, device=dev, dtype=torch.float16)
    cu_cpu = cu.cpu() if cu is not None else None
    g_in = F.logsigmoid(torch.randn(1, T, H, device=dev, dtype=torch.float32))
    # Chunk-local cumulative gates (same as upstream verifiers).
    g32 = g_in.float().cpu()
    g_sum = torch.zeros(1, T, H, dtype=torch.float32)
    for bos, eos in _seq_ranges(T, cu_cpu):
        for j in range(0, eos - bos, C):
            s, e = bos + j, min(bos + j + C, eos)
            g_sum[0, s:e, :] = g32[0, s:e, :].cumsum(dim=1)
    g_sum = g_sum.to(device=dev)
    stream = torch.npu.current_stream()._as_parameter_
    g_t = _transpose_g(g_sum)
    beta_t = _transpose_beta(beta)

    w_out = torch.empty(1, T, H, D, device=dev, dtype=torch.float16)
    u_out = torch.empty(1, T, H, D, device=dev, dtype=torch.float16)

    torch.npu.synchronize()
    run_wy_fast(
        k, v, beta, g_sum, A, w_out, u_out,
        stream=stream,
        g_t=g_t,
        beta_t=beta_t,
        chunk_size=C,
        cu_seqlens=cu,
        batch_size_override=N_seq,
        key_heads=HG,
    )
    torch.npu.synchronize()

    w_ref, u_ref = ref_wy_group(
        k.cpu(), v.cpu(), beta.cpu(), A.cpu(), g_sum.cpu(), C, cu_cpu,
    )

    def _chk(name, actual, expected):
        diff = (actual - expected).abs()
        mx = diff.max().item()
        exp_abs = expected.abs()
        bound = ATOL_CHECK + RTOL_CHECK * exp_abs
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

    _chk("wy_w", w_out.float().cpu(), w_ref.float())
    _chk("wy_u", u_out.float().cpu(), u_ref.float())
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
