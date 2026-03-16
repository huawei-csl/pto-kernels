import argparse
from pathlib import Path

import torch
import torch_npu  # noqa

from bench_common import (
    BENCH_BATCHES,
    BENCH_HIDDEN_DIMS,
    DEFAULT_SCALE,
    QUANTIZE_POOL_KINDS,
    add_common_benchmark_args,
    bandwidth_gbs,
    benchmark_quantize_us,
    benchmark_torch_quantize_us,
    ensure_output_dir,
    make_scale_tensor,
    make_shape_pools,
    resolve_dir_arg,
    torch_npu_quantize,
    validate_benchmark_args,
    write_csv_records,
)
from jit_util_quantize import jit_compile

DEFAULT_WARMUP = 10
DEFAULT_REPEATS = 100


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark PTO quantize kernel against torch_npu.npu_quantize."
    )
    return add_common_benchmark_args(
        parser,
        default_warmup=DEFAULT_WARMUP,
        default_repeats=DEFAULT_REPEATS,
        include_scale=True,
        default_scale=DEFAULT_SCALE,
    ).parse_args()


def _benchmark_torch_npu(x_list, scale_tensor, warmup, repeats):
    return benchmark_torch_quantize_us(
        x_list,
        scale_tensor,
        warmup=warmup,
        repeats=repeats,
        torch_quantize_fn=torch_npu_quantize,
    )


def benchmark(
    quantize_func,
    scale: float,
    warmup: int,
    repeats: int,
    output_dir: Path,
    device: str,
):
    ensure_output_dir(output_dir)

    block_dim = int(
        getattr(
            torch.npu.get_device_properties(device),
            "cube_core_num",
            quantize_func.block_dim,
        )
    )
    scale_tensor = make_scale_tensor(scale, device)

    print(f"\n{'=' * 92}")
    print(f"QUANTIZE BENCHMARK (BLOCK_DIM={block_dim}, scale={scale})")
    print(f"{'=' * 92}")
    header = (
        f"{'batch':>6s}  {'N':>6s}"
        f"  {'pto_us':>10s}  {'torch_npu_us':>14s}"
        f"  {'pto_bw_gbs':>12s}  {'torch_npu_bw_gbs':>18s}  {'pto_speedup':>11s}"
    )
    print(header)
    print("-" * len(header))

    records = []

    for batch in BENCH_BATCHES:
        for n in BENCH_HIDDEN_DIMS:
            pools = make_shape_pools(
                batch,
                n,
                warmup,
                repeats,
                device=device,
                pool_kinds=QUANTIZE_POOL_KINDS,
            )
            x_list = pools["x"]
            y_list = pools["y"]

            pto_us = benchmark_quantize_us(
                quantize_func,
                x_list,
                y_list,
                scale,
                block_dim=block_dim,
                warmup=warmup,
                repeats=repeats,
            )
            torch_npu_us = _benchmark_torch_npu(x_list, scale_tensor, warmup, repeats)

            data_bytes = batch * n * (2 + 1)
            pto_bw = bandwidth_gbs(data_bytes, pto_us)
            torch_npu_bw = bandwidth_gbs(data_bytes, torch_npu_us)
            pto_speedup = torch_npu_us / pto_us if pto_us > 0 else 0.0

            print(
                f"{batch:>6d}  {n:>6d}"
                f"  {pto_us:>10.2f}  {torch_npu_us:>14.2f}"
                f"  {pto_bw:>12.2f}  {torch_npu_bw:>18.2f}  {pto_speedup:>11.3f}"
            )

            records.append(
                (
                    f"{batch},{n},{scale:.4f},{pto_us:.4f},{torch_npu_us:.4f},"
                    f"{pto_bw:.4f},{torch_npu_bw:.4f},{pto_speedup:.4f}"
                )
            )

    csv_path = output_dir / f"quantize_compare_bd{block_dim}.csv"
    write_csv_records(
        csv_path,
        "batch,N,scale,pto_duration_us,torch_npu_duration_us,"
        "pto_bandwidth_gbs,torch_npu_bandwidth_gbs,pto_speedup_vs_torch_npu\n",
        records,
    )
    print(f"\nSaved to {csv_path}")


def main():
    args = _parse_args()
    validate_benchmark_args(args)

    torch.npu.set_device(args.npu)
    base = Path(__file__).resolve().parent

    kernel_path = base / "quantize.cpp"
    csv_dir = resolve_dir_arg(base, args.csv_dir)

    print(f"Using device: {args.npu}")
    print("Compiling quantize.cpp ...")
    quantize_func = jit_compile(str(kernel_path), verbose=True, device=args.npu)
    print()

    benchmark(
        quantize_func,
        scale=args.scale,
        warmup=args.warmup,
        repeats=args.repeats,
        output_dir=csv_dir,
        device=args.npu,
    )


if __name__ == "__main__":
    main()
