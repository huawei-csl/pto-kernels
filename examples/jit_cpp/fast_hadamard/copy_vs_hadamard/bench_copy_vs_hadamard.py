import argparse
import math
from pathlib import Path

import torch
import torch_npu  # noqa

from bench_common import (
    DTYPE,
    POOL_EMPTY_FP16,
    POOL_RANDN_FP16,
    add_common_benchmark_args,
    bandwidth_gbs,
    benchmark_copy_us,
    benchmark_hadamard_us,
    benchmark_torch_clone_us,
    benchmark_trials_us,
    ensure_output_dir,
    make_shape_pools,
    resolve_dir_arg,
    validate_benchmark_args,
    write_csv_records,
)
from copy_vs_hadamard.jit_util_copy_pto import jit_compile as jit_compile_copy_pto
from copy_vs_hadamard.jit_util_copy_raw_cce import (
    jit_compile as jit_compile_copy_raw_cce,
)
from standard.jit_util_hadamard import jit_compile as jit_compile_hadamard

DEFAULT_WARMUP = 2
DEFAULT_REPEATS = 20
DEFAULT_MAX_POOL_ITEMS = 32
DEFAULT_POOL_ROLE_BYTES_MB = 1024
DEFAULT_BATCHES = [1 << exponent for exponent in range(15)]
DEFAULT_HIDDEN_DIMS = [1 << exponent for exponent in range(15)]
DEFAULT_OUTPUT_STEM = "copy_vs_hadamard"
BYTES_PER_FP16 = torch.finfo(DTYPE).bits // 8
POOL_KINDS = {
    "copy_x": POOL_RANDN_FP16,
    "copy_y": POOL_EMPTY_FP16,
    "hadamard_x": POOL_RANDN_FP16,
}
STATIC_BATCH = 4096
STATIC_N = 4096
MIN_SAFE_HADAMARD_N = 128


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark PTO/raw-CCE copy kernels against torch.clone and fast_hadamard."
    )
    parser = add_common_benchmark_args(
        parser,
        default_warmup=DEFAULT_WARMUP,
        default_repeats=DEFAULT_REPEATS,
    )
    parser.add_argument(
        "--max-pool-items",
        type=int,
        default=DEFAULT_MAX_POOL_ITEMS,
        help=(
            "Upper bound for the per-shape rotating input/output pool size used to "
            "limit data reuse (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--pool-role-bytes-mb",
        type=int,
        default=DEFAULT_POOL_ROLE_BYTES_MB,
        help=(
            "Per-pool memory budget in MiB used to choose how many distinct tensors "
            "to rotate through for a shape (default: %(default)s)."
        ),
    )
    return parser.parse_args()


def benchmark_batches(args):
    return args.batches if args.batches else DEFAULT_BATCHES


def benchmark_hidden_dims(args):
    return args.hidden_dims if args.hidden_dims else DEFAULT_HIDDEN_DIMS


def _pool_items_for_shape(batch, n, *, warmup, repeats, max_pool_items, pool_role_bytes):
    shape_bytes = batch * n * BYTES_PER_FP16
    by_bytes = max(1, pool_role_bytes // max(shape_bytes, 1))
    return max(1, min(max_pool_items, warmup + repeats, by_bytes))


def _build_shape_pools(
    batch,
    n,
    *,
    warmup,
    repeats,
    device,
    pool_items,
):
    return make_shape_pools(
        batch,
        n,
        warmup,
        repeats,
        device=device,
        pool_kinds=POOL_KINDS,
        buffer_pool=pool_items,
    )


def _benchmark_candidate(
    benchmark_fn,
    *,
    batch,
    n,
    warmup,
    repeats,
    device,
    pool_items,
):
    pools = _build_shape_pools(
        batch,
        n,
        warmup=warmup,
        repeats=repeats,
        device=device,
        pool_items=pool_items,
    )
    try:
        return benchmark_fn(pools)
    finally:
        del pools
        torch.npu.empty_cache()


def _copy_bandwidth(batch, n, duration_us):
    data_bytes = 2 * batch * n * BYTES_PER_FP16
    return bandwidth_gbs(data_bytes, duration_us)


def _optional_bandwidth(batch, n, duration_us):
    return None if duration_us is None else _copy_bandwidth(batch, n, duration_us)


def _format_optional(value):
    return "" if value is None else f"{value:.4f}"


def benchmark(
    copy_pto_func,
    copy_raw_cce_func,
    copy_raw_cce_static_func,
    hadamard_func,
    *,
    warmup: int,
    repeats: int,
    trials: int,
    output_dir: Path,
    device: str,
    batches,
    hidden_dims,
    max_pool_items: int,
    pool_role_bytes: int,
):
    ensure_output_dir(output_dir)
    block_dim = hadamard_func.block_dim

    print(f"\n{'=' * 108}")
    print(f"COPY VS FAST_HADAMARD (BLOCK_DIM={block_dim})")
    print(f"{'=' * 108}")
    print(
        f"Note: fast_hadamard measurements are left blank for N < {MIN_SAFE_HADAMARD_N} "
        "because the current kernel is only validated on aligned widths from that point up."
    )
    header = (
        f"{'batch':>6s}  {'N':>6s}"
        f"  {'pool':>6s}  {'pto_copy_us':>12s}  {'raw_cce_us':>12s}"
        f"  {'raw_cce_4096_us':>16s}  {'torch_clone_us':>15s}  {'hadamard_us':>12s}"
    )
    print(header)
    print("-" * len(header))

    records = []

    for batch in batches:
        for n in hidden_dims:
            pool_items = _pool_items_for_shape(
                batch,
                n,
                warmup=warmup,
                repeats=repeats,
                max_pool_items=max_pool_items,
                pool_role_bytes=pool_role_bytes,
            )

            copy_pto_stats = benchmark_trials_us(
                trials,
                lambda: _benchmark_candidate(
                    lambda pools: benchmark_copy_us(
                        copy_pto_func,
                        pools["copy_x"],
                        pools["copy_y"],
                        block_dim=copy_pto_func.block_dim,
                        warmup=warmup,
                        repeats=repeats,
                    ),
                    batch=batch,
                    n=n,
                    warmup=warmup,
                    repeats=repeats,
                    device=device,
                    pool_items=pool_items,
                ),
            )
            copy_raw_cce_stats = benchmark_trials_us(
                trials,
                lambda: _benchmark_candidate(
                    lambda pools: benchmark_copy_us(
                        copy_raw_cce_func,
                        pools["copy_x"],
                        pools["copy_y"],
                        block_dim=copy_raw_cce_func.block_dim,
                        warmup=warmup,
                        repeats=repeats,
                    ),
                    batch=batch,
                    n=n,
                    warmup=warmup,
                    repeats=repeats,
                    device=device,
                    pool_items=pool_items,
                ),
            )
            clone_stats = benchmark_trials_us(
                trials,
                lambda: _benchmark_candidate(
                    lambda pools: benchmark_torch_clone_us(
                        pools["copy_x"],
                        warmup=warmup,
                        repeats=repeats,
                    ),
                    batch=batch,
                    n=n,
                    warmup=warmup,
                    repeats=repeats,
                    device=device,
                    pool_items=pool_items,
                ),
            )
            hadamard_stats = None
            if n >= MIN_SAFE_HADAMARD_N:
                hadamard_stats = benchmark_trials_us(
                    trials,
                    lambda: _benchmark_candidate(
                        lambda pools: benchmark_hadamard_us(
                            hadamard_func,
                            pools["hadamard_x"],
                            block_dim=hadamard_func.block_dim,
                            warmup=warmup,
                            repeats=repeats,
                        ),
                        batch=batch,
                        n=n,
                        warmup=warmup,
                        repeats=repeats,
                        device=device,
                        pool_items=pool_items,
                    ),
                )

            static_stats = None
            if batch == STATIC_BATCH and n == STATIC_N:
                static_stats = benchmark_trials_us(
                    trials,
                    lambda: _benchmark_candidate(
                        lambda pools: benchmark_copy_us(
                            copy_raw_cce_static_func,
                            pools["copy_x"],
                            pools["copy_y"],
                            block_dim=copy_raw_cce_static_func.block_dim,
                            warmup=warmup,
                            repeats=repeats,
                        ),
                        batch=batch,
                        n=n,
                        warmup=warmup,
                        repeats=repeats,
                        device=device,
                        pool_items=pool_items,
                    ),
                )

            copy_pto_us = copy_pto_stats["median_us"]
            copy_raw_cce_us = copy_raw_cce_stats["median_us"]
            clone_us = clone_stats["median_us"]
            hadamard_us = (
                None if hadamard_stats is None else hadamard_stats["median_us"]
            )
            static_us = None if static_stats is None else static_stats["median_us"]

            print(
                f"{batch:>6d}  {n:>6d}"
                f"  {pool_items:>6d}  {copy_pto_us:>12.2f}  {copy_raw_cce_us:>12.2f}"
                f"  {_format_optional(static_us):>16s}  {clone_us:>15.2f}  {_format_optional(hadamard_us):>12s}"
            )

            records.append(
                ",".join(
                    [
                        str(batch),
                        str(n),
                        str(pool_items),
                        f"{copy_pto_us:.4f}",
                        f"{copy_raw_cce_us:.4f}",
                        _format_optional(static_us),
                        f"{clone_us:.4f}",
                        _format_optional(hadamard_us),
                        f"{_copy_bandwidth(batch, n, copy_pto_us):.4f}",
                        f"{_copy_bandwidth(batch, n, copy_raw_cce_us):.4f}",
                        _format_optional(
                            None
                            if static_us is None
                            else _copy_bandwidth(batch, n, static_us)
                        ),
                        f"{_copy_bandwidth(batch, n, clone_us):.4f}",
                        _format_optional(_optional_bandwidth(batch, n, hadamard_us)),
                        str(trials),
                        f"{copy_pto_stats['mean_us']:.4f}",
                        f"{copy_pto_stats['std_us']:.4f}",
                        f"{copy_pto_stats['min_us']:.4f}",
                        f"{copy_pto_stats['max_us']:.4f}",
                        f"{copy_pto_stats['cv_pct']:.4f}",
                        f"{copy_raw_cce_stats['mean_us']:.4f}",
                        f"{copy_raw_cce_stats['std_us']:.4f}",
                        f"{copy_raw_cce_stats['min_us']:.4f}",
                        f"{copy_raw_cce_stats['max_us']:.4f}",
                        f"{copy_raw_cce_stats['cv_pct']:.4f}",
                        _format_optional(
                            None if static_stats is None else static_stats["mean_us"]
                        ),
                        _format_optional(
                            None if static_stats is None else static_stats["std_us"]
                        ),
                        _format_optional(
                            None if static_stats is None else static_stats["min_us"]
                        ),
                        _format_optional(
                            None if static_stats is None else static_stats["max_us"]
                        ),
                        _format_optional(
                            None if static_stats is None else static_stats["cv_pct"]
                        ),
                        f"{clone_stats['mean_us']:.4f}",
                        f"{clone_stats['std_us']:.4f}",
                        f"{clone_stats['min_us']:.4f}",
                        f"{clone_stats['max_us']:.4f}",
                        f"{clone_stats['cv_pct']:.4f}",
                        _format_optional(
                            None if hadamard_stats is None else hadamard_stats["mean_us"]
                        ),
                        _format_optional(
                            None if hadamard_stats is None else hadamard_stats["std_us"]
                        ),
                        _format_optional(
                            None if hadamard_stats is None else hadamard_stats["min_us"]
                        ),
                        _format_optional(
                            None if hadamard_stats is None else hadamard_stats["max_us"]
                        ),
                        _format_optional(
                            None if hadamard_stats is None else hadamard_stats["cv_pct"]
                        ),
                    ]
                )
            )

    csv_path = output_dir / f"{DEFAULT_OUTPUT_STEM}_bd{block_dim}.csv"
    write_csv_records(
        csv_path,
        "batch,N,pool_items,copy_pto_us,copy_raw_cce_us,copy_raw_cce_static_4096_us,"
        "torch_clone_us,hadamard_us,copy_pto_bw_gbs,copy_raw_cce_bw_gbs,"
        "copy_raw_cce_static_4096_bw_gbs,torch_clone_bw_gbs,hadamard_bw_gbs,"
        "trials,copy_pto_mean_us,copy_pto_std_us,copy_pto_min_us,copy_pto_max_us,"
        "copy_pto_cv_pct,copy_raw_cce_mean_us,copy_raw_cce_std_us,copy_raw_cce_min_us,"
        "copy_raw_cce_max_us,copy_raw_cce_cv_pct,copy_raw_cce_static_4096_mean_us,"
        "copy_raw_cce_static_4096_std_us,copy_raw_cce_static_4096_min_us,"
        "copy_raw_cce_static_4096_max_us,copy_raw_cce_static_4096_cv_pct,"
        "torch_clone_mean_us,torch_clone_std_us,torch_clone_min_us,"
        "torch_clone_max_us,torch_clone_cv_pct,hadamard_mean_us,hadamard_std_us,"
        "hadamard_min_us,hadamard_max_us,hadamard_cv_pct\n",
        records,
    )
    print(f"\nSaved to {csv_path}")


def main():
    args = _parse_args()
    validate_benchmark_args(args)
    if args.max_pool_items <= 0:
        raise ValueError("--max-pool-items must be > 0")
    if args.pool_role_bytes_mb <= 0:
        raise ValueError("--pool-role-bytes-mb must be > 0")

    torch.npu.set_device(args.npu)
    base = Path(__file__).resolve().parent
    csv_dir = resolve_dir_arg(base, args.csv_dir)

    print(f"Using device: {args.npu}")
    print("Compiling copy_pto.cpp ...")
    copy_pto_func = jit_compile_copy_pto(
        str(base / "copy_pto.cpp"),
        verbose=True,
        device=args.npu,
    )
    print("Compiling copy_raw_cce.cpp ...")
    copy_raw_cce_func = jit_compile_copy_raw_cce(
        str(base / "copy_raw_cce.cpp"),
        verbose=True,
        device=args.npu,
        block_dim=copy_pto_func.block_dim,
    )
    print("Compiling copy_raw_cce_static_4096_4096.cpp ...")
    copy_raw_cce_static_func = jit_compile_copy_raw_cce(
        str(base / "copy_raw_cce_static_4096_4096.cpp"),
        verbose=True,
        device=args.npu,
        block_dim=copy_pto_func.block_dim,
    )
    print("Compiling ../standard/fast_hadamard.cpp ...")
    hadamard_func = jit_compile_hadamard(
        str(base.parent / "standard" / "fast_hadamard.cpp"),
        verbose=True,
        device=args.npu,
    )
    print()

    benchmark(
        copy_pto_func,
        copy_raw_cce_func,
        copy_raw_cce_static_func,
        hadamard_func,
        warmup=args.warmup,
        repeats=args.repeats,
        trials=args.trials,
        output_dir=csv_dir,
        device=args.npu,
        batches=benchmark_batches(args),
        hidden_dims=benchmark_hidden_dims(args),
        max_pool_items=args.max_pool_items,
        pool_role_bytes=args.pool_role_bytes_mb * 1024 * 1024,
    )


if __name__ == "__main__":
    main()
