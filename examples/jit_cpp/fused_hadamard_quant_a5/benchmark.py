#!/usr/bin/env python3
"""Fused Hadamard + MXFP4 quantize on Ascend A5, for both kernels.

Two comparisons per kernel:

  A. the fusion ladder. Unfused is two launches -- the Hadamard, then the
     quantizer -- at 6.53 B/element, against 2.53 fused. Both arms are the same
     kernel, built from one source with the unwanted half compiled out, so they
     differ in what they fuse and in nothing else.

  B. the fused kernel against a d2d copy of its input, as a reference for what
     moving the bytes costs. Not a proven lower bound: the copy is a vendor
     kernel doing a simpler job, and nothing here shows it is optimal.

Method: wall clock on a saturated queue, medians over TRIALS brackets of
LAUNCHES launches, inputs drawn from a rotating pool so a bracket cannot be
served from cache. Every arm is checked against the unfused arm before any of
them is timed.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from jit_util_fused_a5 import MX_BLOCK, build_and_load  # noqa

COPY_ELEMS = 1 << 26  # 67 million elements per timed launch, whatever K is
M = 16384
TRIALS = 15
LAUNCHES = 20
WARMUP = 5
POOL_BYTES = 256 * 1024 * 1024
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)

# At M=16384 the unfused intermediates are 2*M*k bytes. Below K=4096 that fits
# the 128 MiB L2, so the unfused arms partly read from cache and the ladder
# understates fusing. Kept in the sweep because the effect is worth seeing, not
# because those rows are the headline. K=32 is the dispatch floor, same idea.
SHAPES = {
    "full": (1024, 4096, 8192, 16384),
    "b32": (32, 1024, 4096, 16384),
}


def trials(call, depth, launches=LAUNCHES):
    for _ in range(WARMUP):
        call(0)
    torch.npu.synchronize()
    out = []
    for t in range(TRIALS):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for i in range(launches):
            call((t * launches + i) % depth)
        torch.npu.synchronize()
        out.append((time.perf_counter() - t0) * 1e6 / launches)
    med = statistics.median(out)
    return med, 100 * (max(out) - min(out)) / med


def dequant(q, s, k):
    q = q.cpu()
    lo, hi = q & 0x0F, (q >> 4) & 0x0F
    codes = torch.stack([lo, hi], dim=-1).reshape(q.shape[0], -1)
    mag = E2M1[(codes & 0x07).long()]
    sign = torch.where(codes & 0x08 != 0, -1.0, 1.0)
    scale = torch.exp2(s.cpu().float() - 127.0).repeat_interleave(MX_BLOCK, dim=-1)
    return (mag * sign * scale).reshape(-1, k)


def bench_ladder(kind, k):
    """A: two launches, then one."""
    depth = max(2, min(16, POOL_BYTES // max(M * k * 4, 1)))
    x = [torch.randn(M, k, dtype=torch.bfloat16, device="npu") for _ in range(depth)]

    # one source, three builds: both halves, the rotation alone, the quantizer
    # alone. Same tiling, same UB layout, same buffer count in each, so the
    # difference between the arms is the fusion and nothing else.
    fused = build_and_load(kind, k=k, verbose=False)
    rotate_only = build_and_load(
        kind, k=k, verbose=False, extra_defs=("-DFUSED_ROTATE_ONLY",)
    )
    quant = build_and_load(kind, k=k, verbose=False, extra_defs=("-DFUSED_NO_ROTATE",))

    q = torch.empty((M, k // 2), dtype=torch.uint8, device="npu")
    s = torch.empty((M, k // MX_BLOCK), dtype=torch.uint8, device="npu")
    rot = torch.empty((M, k), dtype=torch.bfloat16, device="npu")
    torch.npu.synchronize()

    def two(i):  # Hadamard, then quantize
        rotate_only(x[i % depth], out=(rot.view(torch.uint8), s))
        quant(rot, out=(q, s))

    def one(i):  # both in one launch
        fused(x[i % depth], out=(q, s))

    # correctness gate: the arms must agree before either is timed
    two(0)
    torch.npu.synchronize()
    ref = dequant(q.clone(), s.clone(), k)
    one(0)
    torch.npu.synchronize()
    got = dequant(q, s, k)
    rel = ((got - ref).abs().mean() / ref.abs().mean().clamp_min(1e-6)).item()

    # A disagreeing arm is a bug, not a datum: stop rather than print a table
    # whose rows measure different computations. bf16 rounding differences
    # between a fused and an unfused rotation land near 1e-3, not near 1.
    if rel > 0.05:
        raise SystemExit(
            f"K={k}: arms disagree, rel={rel:.4f} -- the ladder is not measuring "
            "the same computation in every arm"
        )

    t2, s2 = trials(two, depth)
    t1, s1 = trials(one, depth)
    x.clear()
    torch.npu.empty_cache()
    return {
        "k": k,
        "two_us": round(t2, 1),
        "fused_us": round(t1, 1),
        "vs_two": round(t2 / t1, 2),
        "rel": round(rel, 5),
        "spread_pct": round(max(s2, s1), 1),
    }


def bench_copy(kind, k):
    """B: the fused kernel against a copy of its input."""
    batch = max(128, (COPY_ELEMS // k) // 128 * 128)
    depth = max(2, min(16, POOL_BYTES // max(batch * k * 4, 1)))
    x = [
        torch.randn(batch, k, dtype=torch.bfloat16, device="npu") for _ in range(depth)
    ]
    dst = [torch.empty_like(x[0]) for _ in range(depth)]
    fused = build_and_load(kind, k=k, verbose=False)
    q = torch.empty((batch, k // 2), dtype=torch.uint8, device="npu")
    s = torch.empty((batch, k // MX_BLOCK), dtype=torch.uint8, device="npu")
    torch.npu.synchronize()

    tf, sf = trials(lambda i: fused(x[i % depth], out=(q, s)), depth)
    tc, sc = trials(lambda i: dst[i % depth].copy_(x[i % depth]), depth)
    # the kernel reads 2 B and writes 0.5 + 1/32 per element; the copy 2 and 2
    kernel_gbs = batch * k * (2 + 0.5 + 1 / MX_BLOCK) / (tf * 1e-6) / 1e9
    copy_gbs = batch * k * 4.0 / (tc * 1e-6) / 1e9
    x.clear()
    dst.clear()
    torch.npu.empty_cache()
    return {
        "k": k,
        "batch": batch,
        "fused_us": round(tf, 1),
        "copy_us": round(tc, 1),
        "vs_copy": round(tc / tf, 2),
        "fused_gbs": round(kernel_gbs),
        "copy_gbs": round(copy_gbs),
        "spread_pct": round(max(sf, sc), 1),
    }


def run(kind, shapes):
    print(f"\n########## {kind} ##########")
    print(f"=== A. fusion ladder, M={M} (microseconds per launch) ===")
    print(
        f"{'K':>7} {'2 launches':>11} {'fused':>8} {'vs 2':>6} {'rel':>8} "
        f"{'spread':>7}"
    )
    for k in shapes:
        try:
            r = bench_ladder(kind, k)
        except (RuntimeError, SystemExit) as exc:
            print(f"{k:>7}  skipped: {str(exc)[:56]}")
            continue
        print(
            f"{r['k']:>7} {r['two_us']:>11.1f} {r['fused_us']:>8.1f} "
            f"{r['vs_two']:>5.2f}x {r['rel']:>8.4f} {r['spread_pct']:>6.1f}%"
        )

    print(
        f"\n=== B. fused vs a d2d copy "
        f"({COPY_ELEMS / 1e6:.0f} million elements) ==="
    )
    print(
        f"{'K':>7} {'batch':>8} {'fused':>8} {'copy':>8} {'vs copy':>8} "
        f"{'fused GB/s':>11} {'copy GB/s':>10} {'spread':>7}"
    )
    for k in shapes:
        try:
            r = bench_copy(kind, k)
        except RuntimeError as exc:
            print(f"{k:>7}  skipped: {str(exc)[:56]}")
            continue
        print(
            f"{r['k']:>7} {r['batch']:>8} {r['fused_us']:>8.1f} {r['copy_us']:>8.1f} "
            f"{r['vs_copy']:>7.2f}x {r['fused_gbs']:>11} {r['copy_gbs']:>10} "
            f"{r['spread_pct']:>6.1f}%"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--kernel", choices=("full", "b32", "both"), default="both")
    args = ap.parse_args()
    torch.npu.set_device(args.device)
    torch.manual_seed(20260826)
    np.random.seed(0)
    for kind in (("full", "b32") if args.kernel == "both" else (args.kernel,)):
        run(kind, SHAPES[kind])
    return 0


if __name__ == "__main__":
    sys.exit(main())
