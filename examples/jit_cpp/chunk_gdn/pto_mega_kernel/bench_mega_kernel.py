#!/usr/bin/env python3
"""
Benchmark mega-kernel vs aggregated per-stage PTO kernels.

Usage:
  cd examples/jit_cpp/chunk_gdn/pto_mega_kernel
  python bench_mega_kernel.py --device npu:4
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

for p in (_HERE, _CHUNK_GDN, _DYN, _FAST_INV, _E2E):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn.functional as F

from mega_kernel_compile import run_mega_kernel

C_PTO = 128
H_DEFAULT, D_DEFAULT = 16, 128


def _cu_from_seqlens(seqlens):
    cu = [0]
    for s in seqlens:
        cu.append(cu[-1] + s)
    return cu


def _make_inputs(seed, T, H, D, cu_list, dev):
    torch.manual_seed(seed)
    q = torch.randn(1, T, H, D, device=dev, dtype=torch.float16)
    k = torch.randn(1, T, H, D, device=dev, dtype=torch.float16)
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
    return (time.perf_counter() - t0) / iters * 1000.0  # ms


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default=os.getenv("GDN_NPU_DEVICE", "npu:0"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()

    if "PTO_LIB_PATH" not in os.environ:
        fb = "/sources/pto-isa"
        if os.path.isdir(os.path.join(fb, "include")):
            os.environ["PTO_LIB_PATH"] = fb

    torch.manual_seed(args.seed)
    torch.npu.set_device(args.device)
    dev = torch.device(args.device)

    # Try loading per-stage pipeline
    try:
        from verify_pto_triton_e2e import run_pto_e2e
        from jit_util_fast_inverse import jit_compile
        cpp = os.path.join(_FAST_INV, "fast_inverse.cpp")
        tri_inv = jit_compile(cpp, verbose=False)
        per_stage_ok = True
    except Exception as exc:
        print(f"Per-stage PTO not available: {exc}")
        per_stage_ok = False

    scale = D_DEFAULT ** -0.5

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
        ("T=131072", 131072, [0, 131072]),
        ("varlen [256,256]", 512, [0, 256, 512]),
        ("varlen long mix (T=2048)", 2048,
         _cu_from_seqlens([128, 256, 384, 512, 768])),
        ("16x16384 (T=262144)", 262144,
         _cu_from_seqlens([16384] * 16)),
    ]

    print(f"{'Case':<30s}  {'Mega (ms)':>10s}  {'PerStage (ms)':>14s}  {'Speedup':>8s}")
    print("-" * 70)

    for ci, (label, T, cu_list) in enumerate(cases):
        seed_i = args.seed + ci * 10003
        q, k, v, g_in, beta, cu32 = _make_inputs(
            seed_i, T, H_DEFAULT, D_DEFAULT, cu_list, dev)

        def run_mega():
            run_mega_kernel(q, k, v, g_in, beta, cu32,
                            chunk_size=C_PTO, scale=scale)

        t_mega = bench_fn(run_mega, warmup=args.warmup, iters=args.iters)

        if per_stage_ok:
            def run_ps():
                run_pto_e2e(q, k, v, g_in, beta, cu32,
                            tri_inv_func=tri_inv, scale=scale)

            t_ps = bench_fn(run_ps, warmup=args.warmup, iters=args.iters)
            speedup = t_ps / t_mega if t_mega > 0 else float("inf")
            print(f"{label:<30s}  {t_mega:10.3f}  {t_ps:14.3f}  {speedup:7.2f}x")
        else:
            print(f"{label:<30s}  {t_mega:10.3f}  {'n/a':>14s}  {'n/a':>8s}")

    print()


if __name__ == "__main__":
    main()
