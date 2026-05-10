# pylint: disable=wrong-import-position
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
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

from jit_util_common import get_current_stream_ptr  # noqa: E402
from jit_util_layernorm import jit_compile  # noqa: E402

DEFAULT_WARMUP = 10
DEFAULT_REPEATS = 100
EPS = 1e-5

# Read x + write y dominates; gamma/beta reads are small (hidden-only, not rows*hidden).
# Effective bytes = (read x + write y) = 2 * rows * hidden * sizeof(fp16)
BYTES_PER_ELEMENT = 2  # fp16

CSV_HEADER = (
    "rows,N,pto_duration_us,torch_duration_us,"
    "pto_bandwidth_gbs,torch_bandwidth_gbs,pto_speedup_vs_torch,"
    "trials,pto_duration_mean_us,pto_duration_std_us,pto_duration_min_us,"
    "pto_duration_max_us,pto_duration_cv_pct,torch_duration_mean_us,"
    "torch_duration_std_us,torch_duration_min_us,"
    "torch_duration_max_us,torch_duration_cv_pct\n"
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark PTO LayerNorm against torch.nn.functional.layer_norm."
    )
    parser.add_argument(
        "--no-cache-stream",
        dest="cache_stream",
        action="store_false",
        help="Disable cached stream pointer reuse for PTO launches.",
    )
    parser.set_defaults(cache_stream=True)
    return add_common_benchmark_args(
        parser,
        default_warmup=DEFAULT_WARMUP,
        default_repeats=DEFAULT_REPEATS,
    ).parse_args()


def _effective_bandwidth_gbs(rows, hidden, duration_us):
    if duration_us <= 0:
        return 0.0
    # read x + write y, both fp16; gamma/beta amortized
    data_bytes = 2 * rows * hidden * BYTES_PER_ELEMENT
    return (data_bytes / 1e9) / (duration_us / 1e6)


def _make_shape_pools(rows, hidden, warmup, repeats, device):
    return {
        "x": make_buffer_pool(
            warmup,
            repeats,
            lambda: torch.randn(rows, hidden, device=device, dtype=torch.float16),
        ),
        "y": make_buffer_pool(
            warmup,
            repeats,
            lambda: torch.empty(rows, hidden, device=device, dtype=torch.float16),
        ),
    }


def benchmark(
    layernorm_func,
    *,
    warmup: int,
    repeats: int,
    trials: int,
    output_dir: Path,
    device: str,
    batches,
    hidden_dims,
    stream_ptr=None,
):
    ensure_output_dir(output_dir)
    block_dim = layernorm_func.block_dim

    gamma_cache = {}
    beta_cache = {}

    def get_params(hidden):
        if hidden not in gamma_cache:
            gamma_cache[hidden] = torch.ones(hidden, device=device, dtype=torch.float16)
            beta_cache[hidden] = torch.zeros(hidden, device=device, dtype=torch.float16)
        return gamma_cache[hidden], beta_cache[hidden]

    print(f"\n{'=' * 92}")
    print(f"LAYERNORM BENCHMARK (BLOCK_DIM={block_dim})")
    print(f"{'=' * 92}")
    header = (
        f"{'rows':>6s}  {'N':>7s}"
        f"  {'pto_us':>10s}  {'torch_us':>10s}"
        f"  {'pto_bw(GB/s)':>12s}  {'torch_bw(GB/s)':>14s}  {'pto_speedup':>11s}"
    )
    print(header)
    print("-" * len(header))

    records = []
    for rows in batches:
        for hidden in hidden_dims:
            gamma, beta = get_params(hidden)
            pools = _make_shape_pools(rows, hidden, warmup, repeats, device)
            x_list = pools["x"]
            y_list = pools["y"]
            normalized_shape = (hidden,)

            pto_stats = benchmark_trials_us(
                trials,
                lambda x_list=x_list, y_list=y_list, gamma=gamma, beta=beta: benchmark_npu_us(
                    warmup,
                    repeats,
                    lambda i, x_list=x_list, y_list=y_list, gamma=gamma, beta=beta: layernorm_func(
                        pool_item(x_list, i),
                        gamma,
                        beta,
                        pool_item(y_list, i),
                        eps=EPS,
                        block_dim=block_dim,
                        stream_ptr=stream_ptr,
                    ),
                ),
            )
            torch_stats = benchmark_trials_us(
                trials,
                lambda x_list=x_list, normalized_shape=normalized_shape, gamma=gamma, beta=beta: benchmark_npu_us(
                    warmup,
                    repeats,
                    lambda i, x_list=x_list, normalized_shape=normalized_shape, gamma=gamma, beta=beta: F.layer_norm(
                        pool_item(x_list, i),
                        normalized_shape,
                        gamma,
                        beta,
                        eps=EPS,
                    ),
                ),
            )

            pto_us = pto_stats["median_us"]
            torch_us = torch_stats["median_us"]
            pto_bw = _effective_bandwidth_gbs(rows, hidden, pto_us)
            torch_bw = _effective_bandwidth_gbs(rows, hidden, torch_us)
            pto_speedup = torch_us / pto_us if pto_us > 0 else 0.0

            print(
                f"{rows:>6d}  {hidden:>7d}"
                f"  {pto_us:>10.2f}  {torch_us:>10.2f}"
                f"  {pto_bw:>12.4f}  {torch_bw:>14.4f}"
                f"  {pto_speedup:>11.3f}"
            )

            records.append(
                f"{rows},{hidden},{pto_us:.4f},{torch_us:.4f},"
                f"{pto_bw:.6f},{torch_bw:.6f},"
                f"{pto_speedup:.4f},"
                f"{trials},{pto_stats['mean_us']:.4f},{pto_stats['std_us']:.4f},"
                f"{pto_stats['min_us']:.4f},{pto_stats['max_us']:.4f},"
                f"{pto_stats['cv_pct']:.4f},{torch_stats['mean_us']:.4f},"
                f"{torch_stats['std_us']:.4f},"
                f"{torch_stats['min_us']:.4f},"
                f"{torch_stats['max_us']:.4f},"
                f"{torch_stats['cv_pct']:.4f}"
            )

    csv_path = output_dir / f"layernorm_compare_bd{block_dim}.csv"
    write_csv_records(csv_path, CSV_HEADER, records)
    print(f"\nSaved to {csv_path}")


def main():
    args = _parse_args()
    validate_benchmark_args(args)

    torch.npu.set_device(args.npu)
    base = Path(__file__).resolve().parent
    kernel_path = base / "kernel_layernorm.cpp"
    csv_dir = resolve_dir_arg(base, args.csv_dir)

    print(f"Using device: {args.npu}")
    print("Compiling kernel_layernorm.cpp ...")
    layernorm_func = jit_compile(
        str(kernel_path),
        verbose=True,
        device=args.npu,
    )
    stream_ptr = get_current_stream_ptr() if args.cache_stream else None
    if stream_ptr is not None:
        print("Using cached NPU stream pointer for PTO launches.")

    benchmark(
        layernorm_func,
        warmup=args.warmup,
        repeats=args.repeats,
        trials=args.trials,
        output_dir=csv_dir,
        device=args.npu,
        batches=benchmark_batches(args),
        hidden_dims=benchmark_hidden_dims(args),
        stream_ptr=stream_ptr,
    )


if __name__ == "__main__":
    main()
