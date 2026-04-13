# pylint: disable=wrong-import-position
import argparse
import sys
from pathlib import Path

import torch
import torch_npu  # noqa

THIS_DIR = Path(__file__).resolve().parent
FAST_HADAMARD_DIR = THIS_DIR.parent / "fast_hadamard"
if str(FAST_HADAMARD_DIR) not in sys.path:
    sys.path.insert(0, str(FAST_HADAMARD_DIR))

from bench_common import (  # noqa: E402
    add_common_benchmark_args,
    benchmark_batches,
    benchmark_hidden_dims,
    benchmark_npu_us,
    benchmark_trials_us,
    ensure_output_dir,
    make_buffer_pool,
    pool_item,
    resolve_dir_arg,
    validate_benchmark_args,
    write_csv_records,
)

from jit_util_swiglu import jit_compile  # noqa: E402

DEFAULT_WARMUP = 10
DEFAULT_REPEATS = 100
EFFECTIVE_OPS_PER_OUTPUT_ELEMENT = 5.0
CSV_HEADER = (
    "batch,N,pto_duration_us,torch_npu_duration_us,"
    "pto_effective_tops,torch_npu_effective_tops,pto_speedup_vs_torch_npu,"
    "trials,pto_duration_mean_us,pto_duration_std_us,pto_duration_min_us,"
    "pto_duration_max_us,pto_duration_cv_pct,torch_npu_duration_mean_us,"
    "torch_npu_duration_std_us,torch_npu_duration_min_us,"
    "torch_npu_duration_max_us,torch_npu_duration_cv_pct\n"
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark PTO SwiGLU against torch_npu.npu_swiglu."
    )
    return add_common_benchmark_args(
        parser,
        default_warmup=DEFAULT_WARMUP,
        default_repeats=DEFAULT_REPEATS,
    ).parse_args()


def _effective_tops(batch, n, duration_us):
    if duration_us <= 0:
        return 0.0
    total_ops = batch * n * EFFECTIVE_OPS_PER_OUTPUT_ELEMENT
    return total_ops / (duration_us * 1e6)


def _make_shape_pools(batch, n, warmup, repeats, device):
    return {
        "x": make_buffer_pool(
            warmup,
            repeats,
            lambda: torch.randn(batch, 2 * n, device=device, dtype=torch.float16),
        ),
        "y": make_buffer_pool(
            warmup,
            repeats,
            lambda: torch.empty(batch, n, device=device, dtype=torch.float16),
        ),
    }


def benchmark(
    swiglu_func,
    *,
    warmup: int,
    repeats: int,
    trials: int,
    output_dir: Path,
    device: str,
    batches,
    hidden_dims,
):
    ensure_output_dir(output_dir)
    block_dim = swiglu_func.block_dim

    print(f"\n{'=' * 96}")
    print(f"SWIGLU BENCHMARK (BLOCK_DIM={block_dim})")
    print(f"{'=' * 96}")
    header = (
        f"{'batch':>6s}  {'N':>6s}"
        f"  {'pto_us':>10s}  {'torch_npu_us':>13s}"
        f"  {'pto_eff_tops':>12s}  {'torch_eff_tops':>16s}  {'pto_speedup':>11s}"
    )
    print(header)
    print("-" * len(header))

    records = []
    for batch in batches:
        for n in hidden_dims:
            pools = _make_shape_pools(batch, n, warmup, repeats, device)
            x_list = pools["x"]
            y_list = pools["y"]

            pto_stats = benchmark_trials_us(
                trials,
                lambda x_list=x_list, y_list=y_list: benchmark_npu_us(
                    warmup,
                    repeats,
                    lambda i: swiglu_func(
                        pool_item(x_list, i),
                        pool_item(y_list, i),
                        block_dim=block_dim,
                    ),
                ),
            )
            torch_npu_stats = benchmark_trials_us(
                trials,
                lambda x_list=x_list: benchmark_npu_us(
                    warmup,
                    repeats,
                    lambda i: torch_npu.npu_swiglu(pool_item(x_list, i), dim=-1),
                ),
            )

            pto_us = pto_stats["median_us"]
            torch_npu_us = torch_npu_stats["median_us"]
            pto_effective_tops = _effective_tops(batch, n, pto_us)
            torch_npu_effective_tops = _effective_tops(batch, n, torch_npu_us)
            pto_speedup = torch_npu_us / pto_us if pto_us > 0 else 0.0

            print(
                f"{batch:>6d}  {n:>6d}"
                f"  {pto_us:>10.2f}  {torch_npu_us:>13.2f}"
                f"  {pto_effective_tops:>12.4f}  {torch_npu_effective_tops:>16.4f}"
                f"  {pto_speedup:>11.3f}"
            )

            records.append(
                f"{batch},{n},{pto_us:.4f},{torch_npu_us:.4f},"
                f"{pto_effective_tops:.6f},{torch_npu_effective_tops:.6f},"
                f"{pto_speedup:.4f},"
                f"{trials},{pto_stats['mean_us']:.4f},{pto_stats['std_us']:.4f},"
                f"{pto_stats['min_us']:.4f},{pto_stats['max_us']:.4f},"
                f"{pto_stats['cv_pct']:.4f},{torch_npu_stats['mean_us']:.4f},"
                f"{torch_npu_stats['std_us']:.4f},"
                f"{torch_npu_stats['min_us']:.4f},"
                f"{torch_npu_stats['max_us']:.4f},"
                f"{torch_npu_stats['cv_pct']:.4f}"
            )

    csv_path = output_dir / f"swiglu_compare_bd{block_dim}.csv"
    write_csv_records(csv_path, CSV_HEADER, records)
    print(f"\nSaved to {csv_path}")


def main():
    args = _parse_args()
    validate_benchmark_args(args)

    torch.npu.set_device(args.npu)
    base = Path(__file__).resolve().parent
    kernel_path = base / "swiglu.cpp"
    csv_dir = resolve_dir_arg(base, args.csv_dir)

    print(f"Using device: {args.npu}")
    print("Compiling swiglu.cpp ...")
    swiglu_func = jit_compile(
        str(kernel_path),
        verbose=True,
        device=args.npu,
    )

    benchmark(
        swiglu_func,
        warmup=args.warmup,
        repeats=args.repeats,
        trials=args.trials,
        output_dir=csv_dir,
        device=args.npu,
        batches=benchmark_batches(args),
        hidden_dims=benchmark_hidden_dims(args),
    )


if __name__ == "__main__":
    main()
