#!/usr/bin/env python3
"""
Benchmark group-value mega-kernel vs aggregated per-stage PTO kernels.

Default ``--configs``: ``16x16,32x16,48x16,64x16`` (see README).

Usage:
  cd examples/jit_cpp/chunk_gdn/pto_mega_kernel_groupvalue
  python bench_mega_kernel_groupvalue.py --device npu:4
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_CHUNK_GDN = os.path.abspath(os.path.join(_HERE, ".."))
_JIT_CPP = os.path.abspath(os.path.join(_CHUNK_GDN, ".."))
_DYN = os.path.join(_CHUNK_GDN, "dynamic_bsnd")
_FAST_INV = os.path.join(_JIT_CPP, "fast_inverse")
_E2E = os.path.join(_CHUNK_GDN, "pto_e2e_measure")

_DYN_BSND_GV = os.path.join(_CHUNK_GDN, "dynamic_bsnd_groupvalue")
# Standard ``dynamic_kernel_libs`` shadows groupvalue unless ``dynamic_bsnd`` is first on path.
for p in (_HERE, _CHUNK_GDN, _DYN_BSND_GV, _DYN, _FAST_INV, _E2E):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn.functional as F

from mega_kernel_compile import run_mega_kernel

C_PTO = 128


def _cu_from_seqlens(seqlens):
    cu = [0]
    for s in seqlens:
        cu.append(cu[-1] + s)
    return cu


def _make_inputs(seed, T, H, Hg, D, cu_list, dev):
    torch.manual_seed(seed)
    q = torch.randn(1, T, Hg, D, device=dev, dtype=torch.float16)
    k = torch.randn(1, T, Hg, D, device=dev, dtype=torch.float16)
    v = torch.randn(1, T, H, D, device=dev, dtype=torch.float16)
    g_in = torch.randn(1, T, H, device=dev, dtype=torch.float32).sigmoid().log()
    beta = torch.rand(1, T, H, device=dev, dtype=torch.float16)
    q = F.normalize(q.float(), dim=-1, p=2).half()
    k = F.normalize(k.float(), dim=-1, p=2).half()
    cu32 = torch.tensor(cu_list, dtype=torch.int32, device=dev)
    return q, k, v, g_in, beta, cu32


def bench_fn(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=os.getenv("GDN_NPU_DEVICE", "npu:0"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument(
        "--configs",
        type=str,
        default="16x16,32x16,48x16,64x16",
        help="Comma-separated HxHg pairs.",
    )
    args = ap.parse_args()

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

    try:
        from verify_pto_triton_e2e_groupvalue import run_pto_e2e

        from jit_util_fast_inverse import jit_compile

        cpp = os.path.join(_FAST_INV, "fast_inverse.cpp")
        tri_inv = jit_compile(cpp, verbose=False)
        per_stage_ok = True
    except Exception as exc:
        print(f"Per-stage PTO not available: {exc}")
        per_stage_ok = False

    D_DEF = 128
    scale = D_DEF ** -0.5

    cases = [
        ("T=128", 128, [0, 128]),
        ("T=256", 256, [0, 256]),
        ("T=512", 512, [0, 512]),
        ("T=1024", 1024, [0, 1024]),
        ("T=2048", 2048, [0, 2048]),
        ("T=4096", 4096, [0, 4096]),
        ("T=8192", 8192, [0, 8192]),
        ("T=16384", 16384, [0, 16384]),
        ("T=32768", 32768, [0, 32768]),
        ("T=65536", 65536, [0, 65536]),
        ("varlen [256,256]", 512, [0, 256, 512]),
        (
            "varlen long mix (T=2048)",
            2048,
            _cu_from_seqlens([128, 256, 384, 512, 768]),
        ),
        ("16x16384 (T=262144)", 262144, _cu_from_seqlens([16384] * 16)),
    ]

    for H, HG in configs:
        if H % HG != 0:
            print(f"SKIP H={H} Hg={HG}: H must divide by Hg")
            continue

        hdr = (
            f"\nH={H} Hg={HG}: "
            f"{'Case':<30}  {'Mega (ms)':>10}  {'PerStage (ms)':>14}  Speedup\n"
            + "-" * 70
        )
        print(hdr)

        for ci, (label, T, cu_list) in enumerate(cases):
            seed_i = args.seed + ci * 10003 + H * 17 + HG * 31
            q, k, v, g_in, beta, cu32 = _make_inputs(
                seed_i, T, H, HG, D_DEF, cu_list, dev
            )

            stream = torch.npu.current_stream()._as_parameter_

            def run_mega():
                run_mega_kernel(
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

            t_mega = bench_fn(
                run_mega, warmup=args.warmup, iters=args.iters
            )

            if per_stage_ok:

                def run_ps():
                    run_pto_e2e(
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

                t_ps = bench_fn(
                    run_ps, warmup=args.warmup, iters=args.iters
                )
                speedup = t_ps / t_mega if t_mega > 0 else float("inf")
                print(
                    f"{label:<30s}  {t_mega:10.3f}  {t_ps:14.3f}  {speedup:7.2f}x"
                )
            else:
                print(f"{label:<30s}  {t_mega:10.3f}  {'n/a':>14s}  {'n/a':>8s}")


if __name__ == "__main__":
    main()
