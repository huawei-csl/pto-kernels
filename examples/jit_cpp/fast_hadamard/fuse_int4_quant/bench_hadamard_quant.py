import argparse
from pathlib import Path

import torch
import torch_npu  # noqa

from bench_common import (
    DEFAULT_SCALE,
    add_common_benchmark_args,
    benchmark_batches,
    bandwidth_gbs,
    benchmark_fused_hadamard_quant_us,
    benchmark_hidden_dims,
    benchmark_npu_us,
    benchmark_separate_hadamard_quant_us,
    benchmark_trials_us,
    ensure_output_dir,
    hadamard_torch_stagewise,
    make_buffer_pool,
    make_scale_tensor,
    pool_item,
    resolve_dir_arg,
    validate_benchmark_args,
    write_csv_records,
)
from fuse_int4_quant.jit_util_hadamard_quant import jit_compile as jit_compile_fused
from fuse_int4_quant.jit_util_quantize import jit_compile as jit_compile_quantize
from standard.jit_util_hadamard import jit_compile as jit_compile_hadamard

DEFAULT_WARMUP = 10
DEFAULT_REPEATS = 100
CSV_HEADER = (
    "batch,N,scale,fused_duration_us,separate_duration_us,torch_unfused_duration_us,"
    "fused_effective_bandwidth_gbs,separate_effective_bandwidth_gbs,"
    "fused_speedup_vs_separate,fused_speedup_vs_torch_unfused,trials,"
    "fused_duration_mean_us,fused_duration_std_us,fused_duration_min_us,"
    "fused_duration_max_us,fused_duration_cv_pct,separate_duration_mean_us,"
    "separate_duration_std_us,separate_duration_min_us,separate_duration_max_us,"
    "separate_duration_cv_pct,torch_unfused_duration_mean_us,"
    "torch_unfused_duration_std_us,torch_unfused_duration_min_us,"
    "torch_unfused_duration_max_us,torch_unfused_duration_cv_pct\n"
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark fused Hadamard+int4 quantize against separate PTO kernels "
            "and a torch packed-int4 unfused baseline."
        )
    )
    return add_common_benchmark_args(
        parser,
        default_warmup=DEFAULT_WARMUP,
        default_repeats=DEFAULT_REPEATS,
        include_scale=True,
        default_scale=DEFAULT_SCALE,
    ).parse_args()


def _torch_int4_quantize_packed(x, scale_tensor):
    q = torch.round((x * scale_tensor).float()).to(torch.int32)
    q = torch.clamp(q, -8, 7)
    low = torch.bitwise_and(q[:, 0::2], 0xF)
    high = torch.bitwise_and(q[:, 1::2], 0xF)
    packed = torch.bitwise_or(low, high * 16)
    packed = packed.to(torch.int16)
    packed = torch.where(packed >= 128, packed - 256, packed)
    return packed.to(torch.int8)


def _make_pools(batch, n, warmup, repeats, device):
    return {
        "fused_x": make_buffer_pool(
            warmup,
            repeats,
            lambda: torch.randn(batch, n, device=device, dtype=torch.float16),
        ),
        "fused_y": make_buffer_pool(
            warmup,
            repeats,
            lambda: torch.empty(batch, n // 2, device=device, dtype=torch.int8),
        ),
        "separate_x": make_buffer_pool(
            warmup,
            repeats,
            lambda: torch.randn(batch, n, device=device, dtype=torch.float16),
        ),
        "separate_y": make_buffer_pool(
            warmup,
            repeats,
            lambda: torch.empty(batch, n // 2, device=device, dtype=torch.int8),
        ),
        "torch_x": make_buffer_pool(
            warmup,
            repeats,
            lambda: torch.randn(batch, n, device=device, dtype=torch.float16),
        ),
        "torch_work": make_buffer_pool(
            warmup,
            repeats,
            lambda: torch.empty(batch, n, device=device, dtype=torch.float16),
        ),
    }


def _benchmark_torch_unfused(x_list, work_list, scale_tensor, warmup, repeats):
    return benchmark_npu_us(
        warmup,
        repeats,
        lambda i: _torch_int4_quantize_packed(
            hadamard_torch_stagewise(pool_item(x_list, i), pool_item(work_list, i)),
            scale_tensor,
        ),
    )


def _summarize_shape(batch, n, fused_us, separate_us, torch_unfused_us):
    effective_bytes = batch * n * (2.0 + 0.5)
    return {
        "batch": batch,
        "n": n,
        "fused_us": fused_us,
        "separate_us": separate_us,
        "torch_unfused_us": torch_unfused_us,
        "fused_bw": bandwidth_gbs(effective_bytes, fused_us),
        "separate_bw": bandwidth_gbs(effective_bytes, separate_us),
        "fused_speedup_vs_separate": separate_us / fused_us if fused_us > 0 else 0.0,
        "fused_speedup_vs_unfused": (
            torch_unfused_us / fused_us if fused_us > 0 else 0.0
        ),
    }


def _attach_trial_stats(result, fused_stats, separate_stats, torch_stats, trials):
    result.update(
        {
            "trials": trials,
            "fused_mean_us": fused_stats["mean_us"],
            "fused_std_us": fused_stats["std_us"],
            "fused_min_us": fused_stats["min_us"],
            "fused_max_us": fused_stats["max_us"],
            "fused_cv_pct": fused_stats["cv_pct"],
            "separate_mean_us": separate_stats["mean_us"],
            "separate_std_us": separate_stats["std_us"],
            "separate_min_us": separate_stats["min_us"],
            "separate_max_us": separate_stats["max_us"],
            "separate_cv_pct": separate_stats["cv_pct"],
            "torch_unfused_mean_us": torch_stats["mean_us"],
            "torch_unfused_std_us": torch_stats["std_us"],
            "torch_unfused_min_us": torch_stats["min_us"],
            "torch_unfused_max_us": torch_stats["max_us"],
            "torch_unfused_cv_pct": torch_stats["cv_pct"],
        }
    )
    return result


def _print_shape_summary(result):
    print(
        f"{result['batch']:>6d}  {result['n']:>6d}"
        f"  {result['fused_us']:>10.2f}  {result['fused_cv_pct']:>8.2f}"
        f"  {result['separate_us']:>12.2f}  {result['separate_cv_pct']:>8.2f}"
        f"  {result['torch_unfused_us']:>16.2f}  {result['torch_unfused_cv_pct']:>8.2f}"
        f"  {result['fused_bw']:>14.2f}  {result['separate_bw']:>17.2f}"
        f"  {result['fused_speedup_vs_separate']:>12.3f}"
        f"  {result['fused_speedup_vs_unfused']:>17.3f}"
    )


def _csv_record(scale, result):
    return (
        f"{result['batch']},{result['n']},{scale:.4f},{result['fused_us']:.4f},"
        f"{result['separate_us']:.4f},{result['torch_unfused_us']:.4f},"
        f"{result['fused_bw']:.4f},{result['separate_bw']:.4f},"
        f"{result['fused_speedup_vs_separate']:.4f},"
        f"{result['fused_speedup_vs_unfused']:.4f},{result['trials']},"
        f"{result['fused_mean_us']:.4f},{result['fused_std_us']:.4f},"
        f"{result['fused_min_us']:.4f},{result['fused_max_us']:.4f},"
        f"{result['fused_cv_pct']:.4f},{result['separate_mean_us']:.4f},"
        f"{result['separate_std_us']:.4f},{result['separate_min_us']:.4f},"
        f"{result['separate_max_us']:.4f},{result['separate_cv_pct']:.4f},"
        f"{result['torch_unfused_mean_us']:.4f},"
        f"{result['torch_unfused_std_us']:.4f},{result['torch_unfused_min_us']:.4f},"
        f"{result['torch_unfused_max_us']:.4f},{result['torch_unfused_cv_pct']:.4f}"
    )


def benchmark(
    fused_func,
    hadamard_func,
    quantize_func,
    *,
    scale: float,
    warmup: int,
    repeats: int,
    trials: int,
    output_dir: Path,
    device: str,
    batches,
    hidden_dims,
):
    ensure_output_dir(output_dir)
    block_dim = fused_func.block_dim
    scale_tensor = make_scale_tensor(scale, device)

    print(f"\n{'=' * 108}")
    print(f"FUSED HADAMARD+INT4 QUANT BENCHMARK (BLOCK_DIM={block_dim}, scale={scale})")
    print(f"{'=' * 108}")
    header = (
        f"{'batch':>6s}  {'N':>6s}"
        f"  {'fused_us':>10s}  {'fused_cv%':>8s}"
        f"  {'separate_us':>12s}  {'sep_cv%':>8s}"
        f"  {'torch_unfused_us':>16s}  {'torch_cv%':>8s}"
        f"  {'fused_eff_bw':>14s}  {'separate_eff_bw':>17s}"
        f"  {'fused_vs_sep':>12s}  {'fused_vs_unfused':>17s}"
    )
    print(header)
    print("-" * len(header))

    records = []
    for batch in batches:
        for n in hidden_dims:
            pools = _make_pools(batch, n, warmup, repeats, device)
            fused_x = pools["fused_x"]
            fused_y = pools["fused_y"]
            separate_x = pools["separate_x"]
            separate_y = pools["separate_y"]
            torch_x = pools["torch_x"]
            torch_work = pools["torch_work"]
            fused_stats = benchmark_trials_us(
                trials,
                lambda fused_x=fused_x, fused_y=fused_y: benchmark_fused_hadamard_quant_us(
                    fused_func,
                    fused_x,
                    fused_y,
                    scale,
                    block_dim=block_dim,
                    warmup=warmup,
                    repeats=repeats,
                ),
            )
            separate_stats = benchmark_trials_us(
                trials,
                lambda separate_x=separate_x, separate_y=separate_y: benchmark_separate_hadamard_quant_us(
                    hadamard_func,
                    quantize_func,
                    separate_x,
                    separate_y,
                    scale,
                    block_dim=block_dim,
                    warmup=warmup,
                    repeats=repeats,
                ),
            )
            torch_stats = benchmark_trials_us(
                trials,
                lambda torch_x=torch_x, torch_work=torch_work: _benchmark_torch_unfused(
                    torch_x,
                    torch_work,
                    scale_tensor,
                    warmup,
                    repeats,
                ),
            )
            fused_us = fused_stats["median_us"]
            separate_us = separate_stats["median_us"]
            torch_unfused_us = torch_stats["median_us"]
            result = _summarize_shape(
                batch,
                n,
                fused_us,
                separate_us,
                torch_unfused_us,
            )
            result = _attach_trial_stats(
                result,
                fused_stats,
                separate_stats,
                torch_stats,
                trials,
            )
            _print_shape_summary(result)
            records.append(_csv_record(scale, result))

    csv_path = output_dir / f"fht_quant_compare_bd{block_dim}.csv"
    write_csv_records(csv_path, CSV_HEADER, records)
    print(f"\nSaved to {csv_path}")


def main():
    args = _parse_args()
    validate_benchmark_args(args)

    torch.npu.set_device(args.npu)
    base = Path(__file__).resolve().parent
    standard_base = base.parent / "standard"

    fused_kernel_path = base / "fast_hadamard_quant.cpp"
    hadamard_kernel_path = standard_base / "fast_hadamard.cpp"
    quantize_kernel_path = base / "quantize.cpp"
    csv_dir = resolve_dir_arg(base, args.csv_dir)

    print(f"Using device: {args.npu}")
    print("Compiling fast_hadamard_quant.cpp ...")
    fused_func = jit_compile_fused(
        str(fused_kernel_path), verbose=True, device=args.npu
    )
    print("Compiling fast_hadamard.cpp ...")
    hadamard_func = jit_compile_hadamard(
        str(hadamard_kernel_path), verbose=True, device=args.npu
    )
    print("Compiling quantize.cpp ...")
    quantize_func = jit_compile_quantize(
        str(quantize_kernel_path), verbose=True, device=args.npu
    )

    benchmark(
        fused_func,
        hadamard_func,
        quantize_func,
        scale=args.scale,
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
