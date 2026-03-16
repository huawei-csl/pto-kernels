import argparse
from functools import partial
from pathlib import Path

import torch
import torch_npu  # noqa

from bench_common import (
    BENCH_BATCHES,
    BENCH_HIDDEN_DIMS,
    DEFAULT_SCALE,
    add_common_benchmark_args,
    ensure_output_dir,
    format_fused_hadamard_quant_csv_record,
    hadamard_torch_stagewise,
    make_scale_tensor,
    measure_fused_hadamard_quant_shape,
    print_fused_hadamard_quant_shape_summary,
    resolve_dir_arg,
    torch_npu_quantize,
    validate_benchmark_args,
    write_csv_records,
)
from jit_util_hadamard import jit_compile as jit_compile_hadamard
from jit_util_hadamard_quant import jit_compile as jit_compile_fused
from jit_util_quantize import jit_compile as jit_compile_quantize

DEFAULT_WARMUP = 10
DEFAULT_REPEATS = 100
CSV_HEADER = (
    "batch,N,scale,fused_duration_us,separate_duration_us,"
    "torch_unfused_duration_us,"
    "fused_effective_bandwidth_gbs,separate_effective_bandwidth_gbs,"
    "fused_speedup_vs_separate,fused_speedup_vs_torch_unfused\n"
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark fused Hadamard+quantize against separate PTO kernels and "
            "a torch/torch_npu unfused baseline."
        )
    )
    return add_common_benchmark_args(
        parser,
        default_warmup=DEFAULT_WARMUP,
        default_repeats=DEFAULT_REPEATS,
        include_scale=True,
        default_scale=DEFAULT_SCALE,
    ).parse_args()


def benchmark(
    fused_func,
    hadamard_func,
    quantize_func,
    scale: float,
    warmup: int,
    repeats: int,
    output_dir: Path,
    device: str,
):
    ensure_output_dir(output_dir)
    block_dim = fused_func.block_dim
    scale_tensor = make_scale_tensor(scale, device)
    measure_shape = partial(
        measure_fused_hadamard_quant_shape,
        hadamard_stagewise_fn=hadamard_torch_stagewise,
        torch_quantize_fn=torch_npu_quantize,
    )

    print(f"\n{'=' * 108}")
    print(f"FUSED HADAMARD+QUANT BENCHMARK (BLOCK_DIM={block_dim}, scale={scale})")
    print(f"{'=' * 108}")
    header = (
        f"{'batch':>6s}  {'N':>6s}"
        f"  {'fused_us':>10s}  {'separate_us':>12s}  {'torch_unfused_us':>16s}"
        f"  {'fused_eff_bw':>14s}  {'separate_eff_bw':>17s}"
        f"  {'fused_vs_sep':>12s}  {'fused_vs_unfused':>17s}"
    )
    print(header)
    print("-" * len(header))

    records = []

    for batch in BENCH_BATCHES:
        for n in BENCH_HIDDEN_DIMS:
            result = measure_shape(
                fused_func,
                hadamard_func,
                quantize_func,
                batch=batch,
                n=n,
                scale=scale,
                scale_tensor=scale_tensor,
                block_dim=block_dim,
                warmup=warmup,
                repeats=repeats,
                device=device,
            )
            print_fused_hadamard_quant_shape_summary(result)
            records.append(format_fused_hadamard_quant_csv_record(scale, result))

    csv_path = output_dir / f"fht_quant_compare_bd{block_dim}.csv"
    write_csv_records(csv_path, CSV_HEADER, records)
    print(f"\nSaved to {csv_path}")


def main():
    args = _parse_args()
    validate_benchmark_args(args)

    torch.npu.set_device(args.npu)
    base = Path(__file__).resolve().parent

    fused_kernel_path = base / "fast_hadamard_quant.cpp"
    hadamard_kernel_path = base / "fast_hadamard.cpp"
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
        output_dir=csv_dir,
        device=args.npu,
    )


if __name__ == "__main__":
    main()
