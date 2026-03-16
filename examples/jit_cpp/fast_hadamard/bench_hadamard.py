import argparse
from pathlib import Path

import torch
import torch_npu  # noqa

from bench_common import (
    BENCH_BATCHES,
    BENCH_HIDDEN_DIMS,
    HADAMARD_POOL_KINDS,
    add_common_benchmark_args,
    bandwidth_gbs,
    benchmark_hadamard_us,
    ensure_output_dir,
    make_shape_pools,
    resolve_dir_arg,
    validate_benchmark_args,
    write_csv_records,
)
from jit_util_hadamard import jit_compile

DEFAULT_WARMUP = 2
DEFAULT_REPEATS = 20


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Fast Hadamard PTO kernel and save CSV outputs."
    )
    return add_common_benchmark_args(
        parser,
        default_warmup=DEFAULT_WARMUP,
        default_repeats=DEFAULT_REPEATS,
    ).parse_args()


def benchmark(hadamard_func, warmup: int, repeats: int, output_dir: Path, device: str):
    ensure_output_dir(output_dir)
    block_dim = hadamard_func.block_dim

    print(f"\n{'=' * 60}")
    print(f"BENCHMARK (BLOCK_DIM={block_dim})")
    print(f"{'=' * 60}")
    header = (
        f"{'batch':>6s}  {'N':>6s}" f"  {'duration_us':>12s}  {'bandwidth_gbs':>14s}"
    )
    print(header)
    print("-" * len(header))

    records = []

    for batch in BENCH_BATCHES:
        for n in BENCH_HIDDEN_DIMS:
            x_list = make_shape_pools(
                batch,
                n,
                warmup,
                repeats,
                device=device,
                pool_kinds=HADAMARD_POOL_KINDS,
            )["x"]
            dur_us = benchmark_hadamard_us(
                hadamard_func,
                x_list,
                block_dim=block_dim,
                warmup=warmup,
                repeats=repeats,
            )

            data_bytes = 2 * batch * n * 2
            bw_gbs = bandwidth_gbs(data_bytes, dur_us)

            print(f"{batch:>6d}  {n:>6d}  {dur_us:>12.2f}  {bw_gbs:>14.2f}")
            records.append(f"{batch},{n},{dur_us:.4f},{bw_gbs:.4f}")

    csv_path = output_dir / f"fht_pto_bd{block_dim}.csv"
    write_csv_records(csv_path, "batch,N,duration_us,bandwidth_gbs\n", records)
    print(f"\nSaved to {csv_path}")


def main():
    args = _parse_args()
    validate_benchmark_args(args)

    torch.npu.set_device(args.npu)
    base = Path(__file__).resolve().parent

    kernel_path = base / "fast_hadamard.cpp"
    csv_dir = resolve_dir_arg(base, args.csv_dir)

    print(f"Using device: {args.npu}")
    print("Compiling fast_hadamard.cpp ...")
    hadamard_func = jit_compile(str(kernel_path), verbose=True, device=args.npu)
    print()

    benchmark(
        hadamard_func,
        warmup=args.warmup,
        repeats=args.repeats,
        output_dir=csv_dir,
        device=args.npu,
    )


if __name__ == "__main__":
    main()
