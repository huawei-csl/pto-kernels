# pylint: disable=wrong-import-position
"""
Benchmark PTO doubly-stochastic Sinkhorn against PyTorch reference.

Writes:
  outputs/csv/sinkhorn_compare_bd{block_dim}.csv
  outputs/plots/  (via plot_sinkhorn.py)
"""
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
from jit_util_sinkhorn import jit_compile  # noqa: E402

DEFAULT_WARMUP = 10
DEFAULT_REPEATS = 100
SINKHORN_REPEAT = 10
SINKHORN_EPS = 1e-6
BYTES_PER_ELEMENT = 2  # fp16

CSV_HEADER = (
    "batch,N,pto_duration_us,torch_duration_us,"
    "pto_bandwidth_gbs,torch_bandwidth_gbs,pto_speedup_vs_torch,"
    "trials,pto_duration_mean_us,pto_duration_std_us,pto_duration_min_us,"
    "pto_duration_max_us,pto_duration_cv_pct,torch_duration_mean_us,"
    "torch_duration_std_us,torch_duration_min_us,"
    "torch_duration_max_us,torch_duration_cv_pct\n"
)


def sinkhorn_ref(x, repeat=10, eps=1e-6):
    """PyTorch reference (runs on NPU via torch ops)."""
    x = x.float()
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x.half()


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark PTO Sinkhorn (doubly-stochastic) against PyTorch reference."
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


def _effective_bandwidth_gbs(batch, K, duration_us):
    if duration_us <= 0:
        return 0.0
    # read K*K + write K*K
    data_bytes = batch * 2 * K * K * BYTES_PER_ELEMENT
    return (data_bytes / 1e9) / (duration_us / 1e6)


def _make_shape_pools(batch, K, warmup, repeats, device):
    return {
        "x": make_buffer_pool(
            warmup,
            repeats,
            lambda: torch.randn(batch, K, K, device=device, dtype=torch.float16),
        ),
        "y": make_buffer_pool(
            warmup,
            repeats,
            lambda: torch.empty(batch, K, K, device=device, dtype=torch.float16),
        ),
    }


def benchmark(
    sinq_func,
    *,
    warmup,
    repeats,
    trials,
    output_dir,
    device,
    batches,
    hidden_dims,
    stream_ptr=None,
):
    ensure_output_dir(output_dir)
    block_dim = sinq_func.block_dim

    print(f"\n{'=' * 92}")
    print(f"SINKHORN DS BENCHMARK (BLOCK_DIM={block_dim}, repeat={SINKHORN_REPEAT})")
    print(f"{'=' * 92}")
    header = (
        f"{'batch':>6s}  {'K':>6s}"
        f"  {'pto_us':>10s}  {'torch_us':>10s}"
        f"  {'pto_bw(GB/s)':>12s}  {'torch_bw(GB/s)':>14s}  {'pto_speedup':>11s}"
    )
    print(header)
    print("-" * len(header))

    records = []
    for batch in batches:
        for K in hidden_dims:
            pools = _make_shape_pools(batch, K, warmup, repeats, device)
            x_list = pools["x"]
            y_list = pools["y"]

            pto_stats = benchmark_trials_us(
                trials,
                lambda x_list=x_list, y_list=y_list: benchmark_npu_us(
                    warmup,
                    repeats,
                    lambda i: sinq_func(
                        pool_item(x_list, i),
                        pool_item(y_list, i),
                        repeat=SINKHORN_REPEAT,
                        eps=SINKHORN_EPS,
                        stream_ptr=stream_ptr,
                    ),
                ),
            )
            torch_stats = benchmark_trials_us(
                trials,
                lambda x_list=x_list: benchmark_npu_us(
                    warmup,
                    repeats,
                    lambda i: sinkhorn_ref(
                        pool_item(x_list, i),
                        repeat=SINKHORN_REPEAT,
                        eps=SINKHORN_EPS,
                    ),
                ),
            )

            pto_us = pto_stats["median_us"]
            torch_us = torch_stats["median_us"]
            pto_bw = _effective_bandwidth_gbs(batch, K, pto_us)
            torch_bw = _effective_bandwidth_gbs(batch, K, torch_us)
            pto_speedup = torch_us / pto_us if pto_us > 0 else 0.0

            print(
                f"{batch:>6d}  {K:>6d}"
                f"  {pto_us:>10.2f}  {torch_us:>10.2f}"
                f"  {pto_bw:>12.4f}  {torch_bw:>14.4f}"
                f"  {pto_speedup:>11.3f}"
            )

            records.append(
                f"{batch},{K},{pto_us:.4f},{torch_us:.4f},"
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

    csv_path = output_dir / f"sinkhorn_compare_bd{block_dim}.csv"
    write_csv_records(csv_path, CSV_HEADER, records)
    print(f"\nSaved to {csv_path}")


def main():
    args = _parse_args()
    validate_benchmark_args(args)

    torch.npu.set_device(args.npu)
    base = THIS_DIR
    kernel_path = base / "kernel_sinkhorn.cpp"
    csv_dir = resolve_dir_arg(base, args.csv_dir)

    print(f"Using device: {args.npu}")
    print("Compiling kernel_sinkhorn.cpp ...")
    sinq_func = jit_compile(str(kernel_path), verbose=True, device=args.npu)
    stream_ptr = get_current_stream_ptr() if args.cache_stream else None
    if stream_ptr is not None:
        print("Using cached NPU stream pointer for PTO launches.")

    # Override default grids for sinkhorn: batch=N (matrices), K=dim
    batches = args.batches if args.batches else [1, 4, 8, 16, 32, 64]
    dims = args.hidden_dims if args.hidden_dims else [4, 8, 16, 32, 64, 128]

    benchmark(
        sinq_func,
        warmup=args.warmup,
        repeats=args.repeats,
        trials=args.trials,
        output_dir=csv_dir,
        device=args.npu,
        batches=batches,
        hidden_dims=dims,
        stream_ptr=stream_ptr,
    )


if __name__ == "__main__":
    main()
