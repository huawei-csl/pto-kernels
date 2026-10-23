import argparse
import math
from pathlib import Path

import torch
import torch_npu  # noqa

from bench_common import (
    POOL_EMPTY_FP16,
    POOL_RANDN_FP16,
    add_common_benchmark_args,
    bandwidth_gbs,
    benchmark_batches,
    benchmark_npu_us,
    benchmark_trials_us,
    ensure_output_dir,
    make_shape_pools,
    pool_item,
    resolve_dir_arg,
    validate_benchmark_args,
    write_csv_records,
)
from cube.jit_util_hadamard_twosided import twosided_block_dim_schedule
from cube.jit_util_hadamard_twosided import jit_compile as jit_compile_twosided
from standard.jit_util_hadamard import jit_compile as jit_compile_vector

DEFAULT_WARMUP = 10
DEFAULT_REPEATS = 100

# The two-sided path needs N = R * C with both factors 16-aligned, so its grid
# starts at 256 rather than the 128 the vector kernel supports.
BENCH_CUBE_HIDDEN_DIMS = [256, 512, 1024, 2048, 4096, 8192, 16384]

# The two-sided kernel is out of place; the vector kernel is in place.
CUBE_COMPARE_POOL_KINDS = {
    "twosided_x": POOL_RANDN_FP16,
    "twosided_y": POOL_EMPTY_FP16,
    "vector_x": POOL_RANDN_FP16,
}

CSV_HEADER = (
    "batch,N,twosided_block_dim,twosided_duration_us,vector_duration_us,"
    "twosided_bandwidth_gbs,vector_bandwidth_gbs,twosided_speedup_vs_vector,trials,"
    "twosided_duration_mean_us,twosided_duration_std_us,twosided_duration_min_us,"
    "twosided_duration_max_us,twosided_duration_cv_pct,vector_duration_mean_us,"
    "vector_duration_std_us,vector_duration_min_us,vector_duration_max_us,"
    "vector_duration_cv_pct\n"
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the two-sided cube Hadamard PTO kernel against the vector "
            "butterfly kernel and save CSV outputs."
        )
    )
    return add_common_benchmark_args(
        parser,
        default_warmup=DEFAULT_WARMUP,
        default_repeats=DEFAULT_REPEATS,
    ).parse_args()


def twosided_hidden_dims(args):
    return args.hidden_dims if args.hidden_dims else BENCH_CUBE_HIDDEN_DIMS


def benchmark_twosided_hadamard_us(
    twosided_func,
    x_list,
    y_list,
    *,
    block_dim,
    warmup,
    repeats,
) -> float:
    return benchmark_npu_us(
        warmup,
        repeats,
        lambda i: twosided_func(
            pool_item(x_list, i),
            pool_item(y_list, i),
            block_dim=block_dim,
        ),
    )


def benchmark_vector_hadamard_us(
    vector_func,
    x_list,
    *,
    batch,
    n,
    log2_n,
    block_dim,
    warmup,
    repeats,
) -> float:
    return benchmark_npu_us(
        warmup,
        repeats,
        lambda i: vector_func(
            pool_item(x_list, i),
            batch,
            n,
            log2_n,
            block_dim=block_dim,
        ),
    )


def measure_twosided_hadamard_shape(
    twosided_func,
    vector_func,
    *,
    batch,
    n,
    twosided_block_dim,
    vector_block_dim,
    warmup,
    repeats,
    trials,
    device,
):
    pools = make_shape_pools(
        batch,
        n,
        warmup,
        repeats,
        device=device,
        pool_kinds=CUBE_COMPARE_POOL_KINDS,
    )
    twosided_stats = benchmark_trials_us(
        trials,
        lambda: benchmark_twosided_hadamard_us(
            twosided_func,
            pools["twosided_x"],
            pools["twosided_y"],
            block_dim=twosided_block_dim,
            warmup=warmup,
            repeats=repeats,
        ),
    )
    vector_stats = benchmark_trials_us(
        trials,
        lambda: benchmark_vector_hadamard_us(
            vector_func,
            pools["vector_x"],
            batch=batch,
            n=n,
            log2_n=int(math.log2(n)),
            block_dim=vector_block_dim,
            warmup=warmup,
            repeats=repeats,
        ),
    )

    twosided_us = twosided_stats["median_us"]
    vector_us = vector_stats["median_us"]
    data_bytes = 2 * batch * n * 2
    result = {
        "batch": batch,
        "n": n,
        # What this shape actually launched with: None means the wrapper's
        # per-shape schedule is in charge, so resolve it for the record.
        "twosided_block_dim": (
            twosided_block_dim
            if twosided_block_dim is not None
            else twosided_block_dim_schedule(batch, n, twosided_func.block_dim)
        ),
        "twosided_us": twosided_us,
        "vector_us": vector_us,
        "twosided_bw": bandwidth_gbs(data_bytes, twosided_us),
        "vector_bw": bandwidth_gbs(data_bytes, vector_us),
        "twosided_speedup_vs_vector": (
            vector_us / twosided_us if twosided_us > 0 else 0.0
        ),
        "trials": trials,
    }
    for prefix, stats in (("twosided", twosided_stats), ("vector", vector_stats)):
        result.update(
            {
                f"{prefix}_mean_us": stats["mean_us"],
                f"{prefix}_std_us": stats["std_us"],
                f"{prefix}_min_us": stats["min_us"],
                f"{prefix}_max_us": stats["max_us"],
                f"{prefix}_cv_pct": stats["cv_pct"],
            }
        )
    return result


def print_twosided_hadamard_shape_summary(result) -> None:
    print(
        f"{result['batch']:>6d}  {result['n']:>6d}"
        f"  {result['twosided_us']:>10.2f}  {result['twosided_cv_pct']:>8.2f}"
        f"  {result['vector_us']:>10.2f}  {result['vector_cv_pct']:>8.2f}"
        f"  {result['twosided_bw']:>12.2f}  {result['vector_bw']:>14.2f}"
        f"  {result['twosided_speedup_vs_vector']:>15.3f}"
    )


def format_twosided_hadamard_csv_record(result) -> str:
    return (
        f"{result['batch']},{result['n']},{result['twosided_block_dim']},"
        f"{result['twosided_us']:.4f},"
        f"{result['vector_us']:.4f},{result['twosided_bw']:.4f},"
        f"{result['vector_bw']:.4f},{result['twosided_speedup_vs_vector']:.4f},"
        f"{result['trials']},{result['twosided_mean_us']:.4f},"
        f"{result['twosided_std_us']:.4f},{result['twosided_min_us']:.4f},"
        f"{result['twosided_max_us']:.4f},{result['twosided_cv_pct']:.4f},"
        f"{result['vector_mean_us']:.4f},{result['vector_std_us']:.4f},"
        f"{result['vector_min_us']:.4f},{result['vector_max_us']:.4f},"
        f"{result['vector_cv_pct']:.4f}"
    )


def benchmark(
    twosided_func,
    vector_func,
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
    # None lets the wrapper apply its per-shape block_dim schedule, which is
    # what ships; a constant here would benchmark a configuration nobody runs.
    twosided_block_dim = None
    vector_block_dim = vector_func.block_dim

    # Both arms leave stream_ptr at None so that each call resolves it through
    # torch.npu.current_stream(). That is not a cacheable lookup: resolving the
    # stream drains torch_npu's async task queue, and a raw ctypes launch goes
    # straight to the ACL stream -- so hoisting the pointer out of the loop lets
    # the kernel run before the torch ops that fill its inputs and before the
    # events meant to bracket it. The pointer value is identical either way.
    #
    # Both arms also pay that drain, which is more than the device time of the
    # smaller shapes: separating the kernels there needs a profiler
    # (msprof op_summary), not this harness.

    print(f"\n{'=' * 96}")
    print(
        f"TWO-SIDED vs VECTOR HADAMARD BENCHMARK "
        f"(BLOCK_DIM=scheduled per shape, max {twosided_func.block_dim}, "
        f"vector BLOCK_DIM={vector_block_dim})"
    )
    print(f"{'=' * 96}")
    header = (
        f"{'batch':>6s}  {'N':>6s}"
        f"  {'twosided_us':>10s}  {'ts_cv%':>8s}"
        f"  {'vector_us':>10s}  {'vec_cv%':>8s}"
        f"  {'twosided_bw_gbs':>12s}  {'vector_bw_gbs':>14s}"
        f"  {'twosided_vs_vector':>15s}"
    )
    print(header)
    print("-" * len(header))

    records = []

    for batch in batches:
        for n in hidden_dims:
            result = measure_twosided_hadamard_shape(
                twosided_func,
                vector_func,
                batch=batch,
                n=n,
                twosided_block_dim=twosided_block_dim,
                vector_block_dim=vector_block_dim,
                warmup=warmup,
                repeats=repeats,
                trials=trials,
                device=device,
            )
            print_twosided_hadamard_shape_summary(result)
            records.append(format_twosided_hadamard_csv_record(result))

    # Keyed on the cap, not the value used: the schedule picks per shape and the
    # twosided_block_dim column records what each row ran at. The stem has to
    # stay prefix + an int for plot_common.block_dim_from_path.
    csv_path = output_dir / f"fht_twosided_compare_bd{twosided_func.block_dim}.csv"
    write_csv_records(csv_path, CSV_HEADER, records)
    print(f"\nSaved to {csv_path}")


def main():
    args = _parse_args()
    validate_benchmark_args(args)

    torch.npu.set_device(args.npu)
    base = Path(__file__).resolve().parent
    standard_base = base.parent / "standard"

    twosided_kernel_path = base / "fast_hadamard_twosided.cpp"
    vector_kernel_path = standard_base / "fast_hadamard.cpp"
    csv_dir = resolve_dir_arg(base, args.csv_dir)

    print(f"Using device: {args.npu}")
    print("Compiling fast_hadamard_twosided.cpp ...")
    twosided_func = jit_compile_twosided(
        str(twosided_kernel_path), verbose=True, device=args.npu
    )
    print("Compiling fast_hadamard.cpp ...")
    vector_func = jit_compile_vector(
        str(vector_kernel_path), verbose=True, device=args.npu
    )

    benchmark(
        twosided_func,
        vector_func,
        warmup=args.warmup,
        repeats=args.repeats,
        trials=args.trials,
        output_dir=csv_dir,
        device=args.npu,
        batches=benchmark_batches(args),
        hidden_dims=twosided_hidden_dims(args),
    )


if __name__ == "__main__":
    main()
