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
from cube.jit_util_hadamard_onesided import (
    MAX_ONESIDED_N,
    MIN_ONESIDED_BATCH,
    MIN_ONESIDED_N,
    onesided_block_dim_schedule,
)
from cube.jit_util_hadamard_onesided import jit_compile as jit_compile_onesided
from cube.jit_util_hadamard_twosided import MIN_TWOSIDED_N
from cube.jit_util_hadamard_twosided import jit_compile as jit_compile_twosided
from standard.jit_util_hadamard import jit_compile as jit_compile_vector

DEFAULT_WARMUP = 10
DEFAULT_REPEATS = 100

# The one-sided kernel reads the whole of H per tile, so its case table is
# exactly these four sizes -- see MIN/MAX_ONESIDED_N and the kernel header.
BENCH_ONESIDED_HIDDEN_DIMS = [64, 128, 256, 512]

ONESIDED_POOL_KINDS = {
    "onesided_x": POOL_RANDN_FP16,
    "onesided_y": POOL_EMPTY_FP16,
    "vector_x": POOL_RANDN_FP16,
}
# The two-sided kernel needs N = R * C with both factors 16-aligned, so it has
# no arm at the bottom two sizes and those rows report blanks for it.
TWOSIDED_POOL_KINDS = {
    "twosided_x": POOL_RANDN_FP16,
    "twosided_y": POOL_EMPTY_FP16,
}

CSV_HEADER = (
    "batch,N,onesided_block_dim,onesided_duration_us,vector_duration_us,"
    "twosided_duration_us,onesided_bandwidth_gbs,vector_bandwidth_gbs,"
    "twosided_bandwidth_gbs,onesided_speedup_vs_vector,"
    "onesided_speedup_vs_twosided,"
    "trials,onesided_duration_mean_us,onesided_duration_std_us,"
    "onesided_duration_min_us,onesided_duration_max_us,"
    "onesided_duration_cv_pct,vector_duration_mean_us,vector_duration_std_us,"
    "vector_duration_min_us,vector_duration_max_us,vector_duration_cv_pct,"
    "twosided_duration_mean_us,twosided_duration_std_us,twosided_duration_min_us,"
    "twosided_duration_max_us,twosided_duration_cv_pct\n"
)

ARM_STAT_KEYS = ("mean_us", "std_us", "min_us", "max_us", "cv_pct")


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the one-sided (single dense matmul) Hadamard PTO kernel "
            "against the vector butterfly kernel and the two-sided "
            "kernel, and save CSV outputs."
        )
    )
    return add_common_benchmark_args(
        parser,
        default_warmup=DEFAULT_WARMUP,
        default_repeats=DEFAULT_REPEATS,
    ).parse_args()


def onesided_hidden_dims(args):
    return args.hidden_dims if args.hidden_dims else BENCH_ONESIDED_HIDDEN_DIMS


def supports_onesided(n: int) -> bool:
    return MIN_ONESIDED_N <= n <= MAX_ONESIDED_N and not (n & (n - 1))


def supports_two_sided(n: int) -> bool:
    return n >= MIN_TWOSIDED_N


def benchmark_out_of_place_us(func, x_list, y_list, *, block_dim, warmup, repeats):
    return benchmark_npu_us(
        warmup,
        repeats,
        lambda i: func(
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


def _add_arm_stats(result, prefix, stats):
    if stats is None:
        result.update({f"{prefix}_{key}": None for key in ARM_STAT_KEYS})
        return
    result.update({f"{prefix}_{key}": stats[key] for key in ARM_STAT_KEYS})


def measure_onesided_hadamard_shape(
    onesided_func,
    vector_func,
    twosided_func,
    *,
    batch,
    n,
    onesided_block_dim,
    vector_block_dim,
    twosided_block_dim,
    warmup,
    repeats,
    trials,
    device,
):
    pool_kinds = dict(ONESIDED_POOL_KINDS)
    if supports_two_sided(n):
        pool_kinds.update(TWOSIDED_POOL_KINDS)
    pools = make_shape_pools(
        batch,
        n,
        warmup,
        repeats,
        device=device,
        pool_kinds=pool_kinds,
    )

    onesided_stats = benchmark_trials_us(
        trials,
        lambda: benchmark_out_of_place_us(
            onesided_func,
            pools["onesided_x"],
            pools["onesided_y"],
            block_dim=onesided_block_dim,
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
    twosided_stats = (
        benchmark_trials_us(
            trials,
            lambda: benchmark_out_of_place_us(
                twosided_func,
                pools["twosided_x"],
                pools["twosided_y"],
                block_dim=twosided_block_dim,
                warmup=warmup,
                repeats=repeats,
            ),
        )
        if supports_two_sided(n)
        else None
    )

    onesided_us = onesided_stats["median_us"]
    vector_us = vector_stats["median_us"]
    twosided_us = twosided_stats["median_us"] if twosided_stats is not None else None
    data_bytes = 2 * batch * n * 2

    result = {
        "batch": batch,
        "n": n,
        # What this shape actually launched with: None means the wrapper's
        # per-shape schedule is in charge, so resolve it for the record. Below
        # MIN_ONESIDED_BATCH the schedule sees 16 rather than batch.
        "onesided_block_dim": (
            onesided_block_dim
            if onesided_block_dim is not None
            else onesided_block_dim_schedule(
                max(batch, MIN_ONESIDED_BATCH), n, onesided_func.block_dim
            )
        ),
        "onesided_us": onesided_us,
        "vector_us": vector_us,
        "twosided_us": twosided_us,
        "onesided_bw": bandwidth_gbs(data_bytes, onesided_us),
        "vector_bw": bandwidth_gbs(data_bytes, vector_us),
        "twosided_bw": (
            bandwidth_gbs(data_bytes, twosided_us) if twosided_us is not None else None
        ),
        "onesided_speedup_vs_vector": (
            vector_us / onesided_us if onesided_us > 0 else 0.0
        ),
        "onesided_speedup_vs_twosided": (
            (twosided_us / onesided_us if onesided_us > 0 else 0.0)
            if twosided_us is not None
            else None
        ),
        "trials": trials,
    }
    for prefix, stats in (
        ("onesided", onesided_stats),
        ("vector", vector_stats),
        ("twosided", twosided_stats),
    ):
        _add_arm_stats(result, prefix, stats)
    return result


def _fmt(value, width, precision=2):
    """Right-aligned cell, or a dash where an arm does not support the shape."""
    if value is None:
        return f"{'-':>{width}s}"
    return f"{value:>{width}.{precision}f}"


def print_onesided_hadamard_shape_summary(result) -> None:
    print(
        f"{result['batch']:>6d}  {result['n']:>6d}"
        f"  {result['onesided_block_dim']:>6d}"
        f"  {_fmt(result['onesided_us'], 12)}"
        f"  {_fmt(result['onesided_cv_pct'], 8)}"
        f"  {_fmt(result['vector_us'], 10)}  {_fmt(result['vector_cv_pct'], 8)}"
        f"  {_fmt(result['twosided_us'], 10)}  {_fmt(result['twosided_cv_pct'], 8)}"
        f"  {_fmt(result['onesided_bw'], 14)}"
        f"  {_fmt(result['onesided_speedup_vs_vector'], 11, 3)}"
        f"  {_fmt(result['onesided_speedup_vs_twosided'], 11, 3)}"
    )


def _csv_float(value) -> str:
    """Blank for an arm that does not support this shape, so the CSV keeps its
    rectangular shape and the plot layer can filter on the empty cell."""
    return "" if value is None else f"{value:.4f}"


def format_onesided_hadamard_csv_record(result) -> str:
    fields = [
        str(result["batch"]),
        str(result["n"]),
        str(result["onesided_block_dim"]),
        _csv_float(result["onesided_us"]),
        _csv_float(result["vector_us"]),
        _csv_float(result["twosided_us"]),
        _csv_float(result["onesided_bw"]),
        _csv_float(result["vector_bw"]),
        _csv_float(result["twosided_bw"]),
        _csv_float(result["onesided_speedup_vs_vector"]),
        _csv_float(result["onesided_speedup_vs_twosided"]),
        str(result["trials"]),
    ]
    for prefix in ("onesided", "vector", "twosided"):
        fields.extend(_csv_float(result[f"{prefix}_{key}"]) for key in ARM_STAT_KEYS)
    return ",".join(fields)


def benchmark(
    onesided_func,
    vector_func,
    twosided_func,
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
    # None lets each wrapper apply its own per-shape block_dim schedule, which
    # is what ships; a constant here would benchmark a configuration nobody
    # runs. The vector kernel has no schedule, so it keeps its default.
    onesided_block_dim = None
    twosided_block_dim = None
    vector_block_dim = vector_func.block_dim

    # Every arm leaves stream_ptr at None so that each call resolves it through
    # torch.npu.current_stream(). That is not a cacheable lookup: resolving the
    # stream drains torch_npu's async task queue, and a raw ctypes launch goes
    # straight to the ACL stream -- so hoisting the pointer out of the loop lets
    # the kernel run before the torch ops that fill its inputs and before the
    # events meant to bracket it. The pointer value is identical either way.
    #
    # All three arms also pay that drain, which is more than the device time of
    # the smaller shapes: separating the kernels there needs a profiler
    # (msprof op_summary), not this harness.

    unsupported = [n for n in hidden_dims if not supports_onesided(n)]
    if unsupported:
        raise ValueError(
            f"the one-sided kernel supports N in {BENCH_ONESIDED_HIDDEN_DIMS}; "
            f"got unsupported {unsupported}."
        )
    blank_two_sided = [n for n in hidden_dims if not supports_two_sided(n)]
    if blank_two_sided:
        print(
            f"Note: the two-sided kernel needs N >= {MIN_TWOSIDED_N}, so "
            f"N in {blank_two_sided} has no two-sided arm; those cells are blank."
        )

    print(f"\n{'=' * 128}")
    print(
        f"ONE-SIDED vs VECTOR vs TWO-SIDED HADAMARD BENCHMARK "
        f"(one-sided and two-sided BLOCK_DIM=scheduled per shape, max "
        f"{onesided_func.block_dim}, vector BLOCK_DIM={vector_block_dim})"
    )
    print(f"{'=' * 128}")
    header = (
        f"{'batch':>6s}  {'N':>6s}  {'os_bd':>6s}"
        f"  {'onesided_us':>12s}  {'os_cv%':>8s}"
        f"  {'vector_us':>10s}  {'vec_cv%':>8s}"
        f"  {'twosided_us':>10s}  {'ts_cv%':>8s}"
        f"  {'os_bw_gbs':>14s}"
        f"  {'os_vs_vec':>11s}  {'os_vs_twosided':>11s}"
    )
    print(header)
    print("-" * len(header))

    records = []

    for batch in batches:
        for n in hidden_dims:
            result = measure_onesided_hadamard_shape(
                onesided_func,
                vector_func,
                twosided_func,
                batch=batch,
                n=n,
                onesided_block_dim=onesided_block_dim,
                vector_block_dim=vector_block_dim,
                twosided_block_dim=twosided_block_dim,
                warmup=warmup,
                repeats=repeats,
                trials=trials,
                device=device,
            )
            print_onesided_hadamard_shape_summary(result)
            records.append(format_onesided_hadamard_csv_record(result))

    # Keyed on the cap, not the value used: the schedule picks per shape, and the
    # onesided_block_dim column records what each row actually ran at. The stem
    # has to stay prefix + an int for plot_common.block_dim_from_path.
    csv_path = output_dir / f"fht_onesided_compare_bd{onesided_func.block_dim}.csv"
    write_csv_records(csv_path, CSV_HEADER, records)
    print(f"\nSaved to {csv_path}")


def main():
    args = _parse_args()
    validate_benchmark_args(args)

    torch.npu.set_device(args.npu)
    base = Path(__file__).resolve().parent
    repo_base = base.parent

    onesided_kernel_path = base / "fast_hadamard_onesided.cpp"
    twosided_kernel_path = repo_base / "cube" / "fast_hadamard_twosided.cpp"
    vector_kernel_path = repo_base / "standard" / "fast_hadamard.cpp"
    csv_dir = resolve_dir_arg(base, args.csv_dir)

    print(f"Using device: {args.npu}")
    print("Compiling fast_hadamard_onesided.cpp ...")
    onesided_func = jit_compile_onesided(
        str(onesided_kernel_path), verbose=True, device=args.npu
    )
    print("Compiling fast_hadamard_twosided.cpp ...")
    twosided_func = jit_compile_twosided(
        str(twosided_kernel_path), verbose=True, device=args.npu
    )
    print("Compiling fast_hadamard.cpp ...")
    vector_func = jit_compile_vector(
        str(vector_kernel_path), verbose=True, device=args.npu
    )

    benchmark(
        onesided_func,
        vector_func,
        twosided_func,
        warmup=args.warmup,
        repeats=args.repeats,
        trials=args.trials,
        output_dir=csv_dir,
        device=args.npu,
        batches=benchmark_batches(args),
        hidden_dims=onesided_hidden_dims(args),
    )


if __name__ == "__main__":
    main()
