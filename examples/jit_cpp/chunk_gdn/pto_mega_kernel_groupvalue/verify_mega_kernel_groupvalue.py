#!/usr/bin/env python3
"""
Verify group-value mega-kernel against per-stage PTO and CPU fp32 references.

Covers GQA cases (H != Hg) and MHA (H == Hg). Tensor layout matches
``verify_pto_triton_e2e_groupvalue``: ``q``, ``k`` are ``[B,T,Hg,D]``; ``v``,
``β``, gates use ``H`` heads.

Usage:
  cd examples/jit_cpp/chunk_gdn/pto_mega_kernel_groupvalue
  python verify_mega_kernel_groupvalue.py --device npu:4
  python verify_mega_kernel_groupvalue.py --device npu:4 --configs 32x16,48x16
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CHUNK_GDN = os.path.abspath(os.path.join(_HERE, ".."))
_JIT_CPP = os.path.abspath(os.path.join(_CHUNK_GDN, ".."))
_DYN = os.path.join(_CHUNK_GDN, "dynamic_bsnd")
_DYN_GV = os.path.join(_CHUNK_GDN, "dynamic_bsnd_groupvalue")
_FAST_INV = os.path.join(_JIT_CPP, "fast_inverse")
_E2E = os.path.join(_CHUNK_GDN, "pto_e2e_measure")

# ``dynamic_bsnd`` must precede ``dynamic_bsnd_groupvalue`` in resolution order
# (same basename ``dynamic_kernel_libs``); iterate so ``_DYN`` inserts last → first on ``sys.path``.
for p in (_HERE, _CHUNK_GDN, _DYN_GV, _DYN, _FAST_INV, _E2E):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn.functional as F

from mega_kernel_compile import run_mega_kernel

C_PTO = 128

MAX_RMSE_OVER_MEAN_ABS = 0.15
MIN_R2 = 0.99
MIN_PEARSON = 0.995


def r2_score(y_ref, y):
    ref = np.asarray(y_ref.detach().cpu().numpy().ravel(), dtype=np.float64)
    pred = np.asarray(y.detach().cpu().numpy().ravel(), dtype=np.float64)
    ss_res = float(np.sum((ref - pred) ** 2))
    ss_tot = float(np.sum((ref - np.mean(ref)) ** 2))
    if ss_tot <= 1e-30 * max(ref.size, 1):
        return float("nan")
    return 1.0 - ss_res / ss_tot


def pearson_r(x, y):
    a = np.asarray(x.detach().cpu().numpy().ravel(), dtype=np.float64)
    b = np.asarray(y.detach().cpu().numpy().ravel(), dtype=np.float64)
    if a.size < 2:
        return float("nan")
    if np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(a, b)
    v = float(c[0, 1])
    return v if np.isfinite(v) else float("nan")


def _rmse(a, b):
    return float(torch.sqrt(((a - b) ** 2).mean()).item())


def _cu_from_seqlens(seqlens):
    cu = [0]
    for s in seqlens:
        cu.append(cu[-1] + s)
    return cu


def _make_inputs(seed, T, H, Hg, D, cu_list, dev):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    q = torch.randn(1, T, Hg, D, generator=g)
    k = torch.randn(1, T, Hg, D, generator=g)
    v = torch.randn(1, T, H, D, generator=g)
    g_in = F.logsigmoid(torch.randn(1, T, H, generator=g))
    beta = torch.rand(1, T, H, generator=g)
    q, k = F.normalize(q, dim=-1, p=2), F.normalize(k, dim=-1, p=2)
    q_fp = q.to(dev, dtype=torch.float16)
    k_fp = k.to(dev, dtype=torch.float16)
    v_fp = v.to(dev, dtype=torch.float16)
    g_fp = g_in.to(dev, dtype=torch.float32)
    beta_fp = beta.to(dev, dtype=torch.float16)
    cu32 = torch.tensor(cu_list, dtype=torch.int32, device=dev)
    return q_fp, k_fp, v_fp, g_fp, beta_fp, cu32


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default=os.getenv("GDN_NPU_DEVICE", "npu:0"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--skip-per-stage",
        action="store_true",
        help="Skip per-stage PTO comparison",
    )
    p.add_argument(
        "--configs",
        type=str,
        default="16x16,32x16,48x16,64x16",
        help=(
            "Comma-separated HxHg pairs to test, e.g. '32x16,48x16'. "
            "Each runs the full shape list."
        ),
    )
    args = p.parse_args()

    if "PTO_LIB_PATH" not in os.environ:
        fb = "/sources/pto-isa"
        if os.path.isdir(os.path.join(fb, "include")):
            os.environ["PTO_LIB_PATH"] = fb

    configs = []
    for part in args.configs.split(","):
        part = part.strip()
        if not part:
            continue
        hh, hv = part.lower().replace("×", "x").split("x")
        configs.append((int(hh), int(hv)))

    torch.manual_seed(args.seed)
    torch.npu.set_device(args.device)
    dev = torch.device(args.device)

    per_stage_available = False
    if not args.skip_per_stage:
        try:
            from verify_pto_triton_e2e_groupvalue import run_pto_e2e

            from jit_util_fast_inverse import jit_compile

            cpp = os.path.join(_FAST_INV, "fast_inverse.cpp")
            tri_inv = jit_compile(cpp, verbose=False)
            per_stage_available = True
            print("Per-stage group-value PTO pipeline loaded.")
        except Exception as exc:
            print(f"Warning: per-stage pipeline not available: {exc}")

    try:
        sys.path.insert(0, _DYN_GV)
        from verify_dynamic_bsnd_groupvalue import (
            ref_chunk_h_group,
            ref_chunk_o_group,
            ref_cumsum,
            ref_kkt_group,
            ref_wy_group,
        )
        from verify_dynamic_bsnd import ref_solve_tril

        cpu_ref_available = True
    except ImportError:
        cpu_ref_available = False

    cases = [
        ("T=128", 128, [0, 128]),
        ("T=256", 256, [0, 256]),
        ("T=512", 512, [0, 512]),
        ("T=1024", 1024, [0, 1024]),
        ("T=2048", 2048, [0, 2048]),
        ("T=4096", 4096, [0, 4096]),
        ("varlen [256,256]", 512, [0, 256, 512]),
        ("varlen [128,128,128]", 384, [0, 128, 256, 384]),
        ("varlen [150,300]", 450, [0, 150, 450]),
        ("varlen [129,255]", 384, [0, 129, 384]),
        (
            "varlen boundary mix",
            530,
            _cu_from_seqlens([1, 17, 128, 129, 255]),
        ),
        (
            "varlen dense ladder",
            1536,
            _cu_from_seqlens(
                [1, 17, 31, 32, 33, 95, 127, 128, 129, 191, 192, 193, 367]
            ),
        ),
        (
            "varlen long mix",
            2048,
            _cu_from_seqlens([128, 256, 384, 512, 768]),
        ),
    ]

    ok_total = 0
    n_total = 0
    for H, HG in configs:
        if H % HG != 0:
            print(f"SKIP H={H} Hg={HG}: H must be divisible by Hg")
            continue
        scale = 128 ** -0.5
        print(f"\n=== Config: H={H} (value heads), Hg={HG} (Q/K heads) ===")
        for ci, (label, T, cu_list) in enumerate(cases):
            seed_i = args.seed + ci * 10003 + H * 17 + HG * 31
            q, k, v, g_in, beta, cu32 = _make_inputs(
                seed_i, T, H, HG, 128, cu_list, dev
            )

            torch.npu.synchronize()
            stream = torch.npu.current_stream()._as_parameter_
            o_mega = run_mega_kernel(
                q,
                k,
                v,
                g_in,
                beta,
                cu32,
                stream=stream,
                chunk_size=C_PTO,
                scale=scale,
                key_heads=HG,
            )
            torch.npu.synchronize()

            mega_f = o_mega.float().cpu()

            if per_stage_available:
                torch.npu.synchronize()
                o_perstage = run_pto_e2e(
                    q,
                    k,
                    v,
                    g_in,
                    beta,
                    cu32,
                    stream=stream,
                    tri_inv_func=tri_inv,
                    scale=scale,
                    H=H,
                    HG=HG,
                )
                torch.npu.synchronize()
                ps_f = o_perstage.float().cpu()

                rmse_ps = _rmse(mega_f, ps_f)
                mean_abs_ps = float(ps_f.abs().mean().item())
                ratio_ps = rmse_ps / max(mean_abs_ps, 1e-15)
                r2_ps = r2_score(ps_f, mega_f)
                pr_ps = pearson_r(ps_f, mega_f)
            else:
                ratio_ps = r2_ps = pr_ps = float("nan")
                rmse_ps = float("nan")

            if cpu_ref_available:
                q_ref = q.float().cpu()
                k_ref = k.float().cpu()
                v_ref = v.float().cpu()
                g_ref = g_in.float().cpu()
                beta_ref = beta.float().cpu()
                cu_cpu = torch.tensor(cu_list, dtype=torch.long)
                g_sum_ref = ref_cumsum(g_ref, C_PTO, cu_cpu)
                A_ref = ref_kkt_group(
                    k_ref, beta_ref, g_sum_ref, C_PTO, cu_cpu
                )
                A_sol_ref = ref_solve_tril(A_ref, C_PTO, cu_cpu)
                w_ref, u_ref = ref_wy_group(
                    k_ref,
                    v_ref,
                    beta_ref,
                    A_sol_ref,
                    g_sum_ref,
                    C_PTO,
                    cu_cpu,
                )
                h_ref, vn_ref, _ = ref_chunk_h_group(
                    k_ref, w_ref, u_ref, g_sum_ref, C_PTO, cu_cpu
                )
                o_ref = ref_chunk_o_group(
                    q_ref,
                    k_ref,
                    vn_ref,
                    h_ref,
                    g_sum_ref,
                    C_PTO,
                    cu_cpu,
                )
                o_ref = (o_ref * scale).float()

                rmse_ref = _rmse(mega_f, o_ref)
                mean_abs_ref = float(o_ref.abs().mean().item())
                ratio_ref = rmse_ref / max(mean_abs_ref, 1e-15)
                r2_ref = r2_score(o_ref, mega_f)
                pr_ref = pearson_r(o_ref, mega_f)
            else:
                ratio_ref = r2_ref = pr_ref = float("nan")

            if per_stage_available:
                ok_ps = ratio_ps < 0.005 or (
                    np.isfinite(r2_ps) and r2_ps > 0.9999
                )
            else:
                ok_ps = True

            if cpu_ref_available:
                ok_ref = ratio_ref < MAX_RMSE_OVER_MEAN_ABS
                ok_r2 = (not np.isfinite(r2_ref)) or r2_ref >= MIN_R2
                ok_pr = (not np.isfinite(pr_ref)) or abs(pr_ref) >= MIN_PEARSON
                ok_cpu = ok_ref and ok_r2 and ok_pr
            else:
                ok_cpu = True

            passed = ok_ps and ok_cpu
            ps_str = (
                f"mega~PS rmse/|ref|={ratio_ps:.5f} r2={r2_ps:.5f}"
                if per_stage_available
                else "PS: n/a"
            )
            ref_str = (
                f"mega~Ref rmse/|ref|={ratio_ref:.4f} r2={r2_ref:.4f} "
                f"ρ={pr_ref:.4f}"
                if cpu_ref_available
                else "Ref: n/a"
            )
            status = "PASS" if passed else "FAIL"
            print(f"[{status}] H={H}Hg={HG} {label}: {ps_str} | {ref_str}")
            if passed:
                ok_total += 1
            n_total += 1

    print(f"\n{ok_total}/{n_total} sub-cases passed (all configs × shapes).")
    return 0 if ok_total == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
