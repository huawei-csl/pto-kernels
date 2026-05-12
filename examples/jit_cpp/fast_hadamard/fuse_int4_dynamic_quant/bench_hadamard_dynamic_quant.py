"""Benchmark fused Hadamard + dynamic int4 quant vs static fused and a traffic-matched copy baseline."""

import math
from pathlib import Path

import torch
import torch_npu  # noqa

from bench_common import (
    add_common_benchmark_args,
    benchmark_hidden_dims,
    benchmark_npu_us,
    bandwidth_gbs,
    ensure_output_dir,
    resolve_dir_arg,
    validate_benchmark_args,
    write_csv_records,
)
from fuse_int4_dynamic_quant.jit_util_hadamard_dynamic_quant import (
    jit_compile as jit_compile_dynamic,
)
from fuse_int4_dynamic_quant.jit_util_traffic_copy import (
    jit_compile as jit_compile_traffic_copy,
)
from fuse_int4_quant.jit_util_hadamard_quant import (
    jit_compile as jit_compile_static,
)

DEFAULT_WARMUP = 20
DEFAULT_REPEATS = 200
DEFAULT_HADAMARD_N = 128
DEFAULT_BATCHES = [1, 5, 8, 10, 16, 20, 32, 40, 64]
CSV_HEADER = (
    "batch,N,hadamard_n,traffic_us,static_us,dynamic_us,"
    "traffic_bw_gbs,static_bw_gbs,dynamic_bw_gbs,"
    "speedup_vs_static,speedup_vs_traffic\n"
)


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark fused Hadamard+dynamic_quant against static fused "
            "and a single-launch traffic-matched copy baseline."
        )
    )
    parser = add_common_benchmark_args(
        parser,
        default_warmup=DEFAULT_WARMUP,
        default_repeats=DEFAULT_REPEATS,
    )
    parser.add_argument(
        "--hadamard-n",
        type=int,
        default=DEFAULT_HADAMARD_N,
        help=f"Blockwise Hadamard dimension (default: {DEFAULT_HADAMARD_N}).",
    )
    return parser.parse_args()


def _effective_bytes(batch, n):
    return batch * n * 2 + batch * n // 2 + batch * 4


def main():
    args = _parse_args()
    validate_benchmark_args(args)

    torch.npu.set_device(args.npu)
    base = Path(__file__).resolve().parent
    csv_dir = resolve_dir_arg(base, getattr(args, "csv_dir", None))
    if csv_dir is not None:
        ensure_output_dir(csv_dir)

    hadamard_n = args.hadamard_n
    batches = args.batches if args.batches else DEFAULT_BATCHES
    hidden_dims = benchmark_hidden_dims(args)

    print(f"Using device: {args.npu}")
    print("Compiling fast_hadamard_dynamic_quant.cpp ...")
    dynamic_func = jit_compile_dynamic(
        str(base / "fast_hadamard_dynamic_quant.cpp"),
        verbose=True,
        device=args.npu,
    )
    print("Compiling traffic_copy.cpp (copy baseline) ...")
    traffic_func = jit_compile_traffic_copy(
        str(base / "traffic_copy.cpp"),
        verbose=True,
        device=args.npu,
    )
    print("Compiling fast_hadamard_quant.cpp (static reference) ...")
    static_func = jit_compile_static(
        str(base.parent / "fuse_int4_quant" / "fast_hadamard_quant.cpp"),
        verbose=True,
        device=args.npu,
    )

    warmup = args.warmup
    repeats = args.repeats
    header = (
        f"{'batch':>6s}  {'N':>6s}  {'had_n':>5s}"
        f"  {'traffic':>8s}  {'static':>8s}  {'dynamic':>8s}"
        f"  {'tr_bw':>8s}  {'st_bw':>8s}  {'dy_bw':>8s}"
        f"  {'sp/st':>6s}  {'sp/tr':>6s}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    records = []
    for n in hidden_dims:
        hn = min(hadamard_n, n)
        log2_hn = int(math.log2(hn))
        num_blocks = n // hn
        for batch in batches:
            eff = _effective_bytes(batch, n)

            # Single-launch byte copy with total GM traffic matched to the
            # dynamic kernel's effective bytes.
            traffic_bytes = eff // 2
            traffic_src = torch.empty(traffic_bytes, device=args.npu, dtype=torch.int8)
            traffic_dst = torch.empty_like(traffic_src)
            traffic_us = benchmark_npu_us(
                warmup,
                repeats,
                lambda i, traffic_src=traffic_src, traffic_dst=traffic_dst: traffic_func(
                    traffic_src, traffic_dst
                ),
            )
            traffic_bw = bandwidth_gbs(eff, traffic_us)

            # Static fused (PTO reference, hadamard_n-sized rows)
            st_batch = batch * num_blocks
            sx = torch.randn(st_batch, hn, device=args.npu, dtype=torch.float16)
            sy = torch.empty(st_batch, hn // 2, device=args.npu, dtype=torch.int8)
            st_us = benchmark_npu_us(
                warmup,
                repeats,
                lambda i, sx=sx, sy=sy, st_batch=st_batch, hn=hn, log2_hn=log2_hn: static_func(
                    sx, sy, st_batch, hn, log2_hn, 9.0
                ),
            )
            st_bw = bandwidth_gbs(eff, st_us)

            # Dynamic fused (our kernel)
            dx = torch.randn(batch, n, device=args.npu, dtype=torch.float16)
            dy = torch.empty(batch, n // 2, device=args.npu, dtype=torch.int8)
            ds = torch.empty(batch, dtype=torch.float32, device=args.npu)
            dx_scratch = dx.clone()
            dy_us = benchmark_npu_us(
                warmup,
                repeats,
                lambda i, dx_scratch=dx_scratch, dy=dy, ds=ds, batch=batch, n=n, hn=hn: dynamic_func(
                    dx_scratch, dy, ds, batch, n, hn
                ),
            )
            dy_bw = bandwidth_gbs(eff, dy_us)

            speedup_st = st_us / dy_us if dy_us > 0 else 0
            speedup_tr = traffic_us / dy_us if dy_us > 0 else 0

            print(
                f"{batch:>6d}  {n:>6d}  {hn:>5d}"
                f"  {traffic_us:>8.2f}  {st_us:>8.2f}  {dy_us:>8.2f}"
                f"  {traffic_bw:>8.2f}  {st_bw:>8.2f}  {dy_bw:>8.2f}"
                f"  {speedup_st:>5.2f}x  {speedup_tr:>5.2f}x"
            )

            records.append(
                f"{batch},{n},{hn},{traffic_us:.4f},{st_us:.4f},{dy_us:.4f},"
                f"{traffic_bw:.4f},{st_bw:.4f},{dy_bw:.4f},"
                f"{speedup_st:.4f},{speedup_tr:.4f}"
            )

    if csv_dir is not None:
        write_csv_records(
            csv_dir / "hadamard_dynamic_quant_int4.csv", CSV_HEADER, records
        )


if __name__ == "__main__":
    main()
