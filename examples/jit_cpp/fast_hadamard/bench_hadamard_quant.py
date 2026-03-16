import argparse
import math
from pathlib import Path

import torch
import torch_npu  # noqa

from jit_util_hadamard import chmod_output_path, normalize_npu_device
from jit_util_hadamard import jit_compile as jit_compile_hadamard
from jit_util_hadamard_quant import jit_compile as jit_compile_fused
from jit_util_quantize import jit_compile as jit_compile_quantize

BENCH_BATCHES = [1, 5, 8, 10, 16, 20, 32, 40, 64, 128, 256, 512, 1024]
BENCH_HIDDEN_DIMS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]

DEFAULT_DEVICE = "npu:0"
DTYPE = torch.float16
DEFAULT_SCALE = 9.0
DEFAULT_WARMUP = 10
DEFAULT_REPEATS = 100
DEFAULT_CSV_DIR = Path("outputs") / "csv"


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark fused Hadamard+quantize against separate PTO kernels and "
            "a torch/torch_npu unfused baseline."
        )
    )
    parser.add_argument(
        "--npu",
        type=normalize_npu_device,
        default=DEFAULT_DEVICE,
        help="NPU device (examples: 0, npu:0, '0', 'npu:0').",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=DEFAULT_SCALE,
        help=f"Quantization scale multiplier (default: {DEFAULT_SCALE}).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"Warmup iterations per shape (default: {DEFAULT_WARMUP}).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"Timed iterations per shape (default: {DEFAULT_REPEATS}).",
    )
    parser.add_argument(
        "--csv-dir",
        type=str,
        default=str(DEFAULT_CSV_DIR),
        help=f"Output CSV directory (default: {DEFAULT_CSV_DIR}).",
    )
    return parser.parse_args()


def hadamard_torch_stagewise(x, work):
    n = x.shape[-1]
    n_half = n // 2
    log2_n = int(math.log2(n))
    src = x
    dst = work

    for _ in range(log2_n):
        pairs = src.reshape(x.shape[0], n_half, 2)
        torch.add(pairs[..., 0], pairs[..., 1], out=dst[:, :n_half])
        torch.sub(pairs[..., 0], pairs[..., 1], out=dst[:, n_half:])
        src, dst = dst, src
    return src


def _make_scale_tensor(scale: float, device: str):
    return torch.tensor([scale], device=device, dtype=DTYPE)


def _torch_npu_quantize(x, scale_tensor):
    return torch_npu.npu_quantize(x, scale_tensor, None, torch.qint8, -1, False)


def _benchmark_fused(fused_func, x_list, y_list, scale, block_dim, warmup, repeats):
    for i in range(warmup):
        n = x_list[i].shape[1]
        fused_func(
            x_list[i],
            y_list[i],
            x_list[i].shape[0],
            n,
            int(math.log2(n)),
            scale,
            block_dim=block_dim,
        )
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)

    start.record()
    for i in range(repeats):
        n = x_list[warmup + i].shape[1]
        fused_func(
            x_list[warmup + i],
            y_list[warmup + i],
            x_list[warmup + i].shape[0],
            n,
            int(math.log2(n)),
            scale,
            block_dim=block_dim,
        )
    end.record()
    torch.npu.synchronize()

    return start.elapsed_time(end) / repeats * 1e3


def _benchmark_separate(
    hadamard_func,
    quantize_func,
    x_list,
    y_list,
    scale,
    block_dim,
    warmup,
    repeats,
):
    for i in range(warmup):
        batch, n = x_list[i].shape
        log2_n = int(math.log2(n))
        hadamard_func(x_list[i], batch, n, log2_n, block_dim=block_dim)
        quantize_func(x_list[i], y_list[i], scale, block_dim=block_dim)
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)

    start.record()
    for i in range(repeats):
        batch, n = x_list[warmup + i].shape
        log2_n = int(math.log2(n))
        hadamard_func(x_list[warmup + i], batch, n, log2_n, block_dim=block_dim)
        quantize_func(
            x_list[warmup + i], y_list[warmup + i], scale, block_dim=block_dim
        )
    end.record()
    torch.npu.synchronize()

    return start.elapsed_time(end) / repeats * 1e3


def _benchmark_torch_unfused(x_list, work_list, scale_tensor, warmup, repeats):
    for i in range(warmup):
        y = hadamard_torch_stagewise(x_list[i], work_list[i])
        _torch_npu_quantize(y, scale_tensor)
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)

    start.record()
    for i in range(repeats):
        y = hadamard_torch_stagewise(x_list[warmup + i], work_list[warmup + i])
        _torch_npu_quantize(y, scale_tensor)
    end.record()
    torch.npu.synchronize()

    return start.elapsed_time(end) / repeats * 1e3


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
    output_dir.mkdir(parents=True, exist_ok=True)
    chmod_output_path(output_dir)
    block_dim = fused_func.block_dim
    scale_tensor = _make_scale_tensor(scale, device)

    print(f"\n{'=' * 108}")
    print(f"FUSED HADAMARD+QUANT BENCHMARK (BLOCK_DIM={block_dim}, scale={scale})")
    print(f"{'=' * 108}")
    header = (
        f"{'batch':>6s}  {'N':>6s}"
        f"  {'fused_us':>10s}  {'separate_us':>12s}  {'torch_unfused_us':>16s}"
        f"  {'fused_bw_gbs':>14s}  {'separate_bw_gbs':>17s}"
        f"  {'fused_vs_sep':>12s}  {'fused_vs_unfused':>17s}"
    )
    print(header)
    print("-" * len(header))

    records = []

    for batch in BENCH_BATCHES:
        for n in BENCH_HIDDEN_DIMS:
            allocated = warmup + repeats
            fused_x_list = [
                torch.randn(batch, n, device=device, dtype=DTYPE)
                for _ in range(allocated)
            ]
            fused_y_list = [
                torch.empty(batch, n, device=device, dtype=torch.int8)
                for _ in range(allocated)
            ]
            separate_x_list = [
                torch.randn(batch, n, device=device, dtype=DTYPE)
                for _ in range(allocated)
            ]
            separate_y_list = [
                torch.empty(batch, n, device=device, dtype=torch.int8)
                for _ in range(allocated)
            ]
            torch_x_list = [
                torch.randn(batch, n, device=device, dtype=DTYPE)
                for _ in range(allocated)
            ]
            torch_work_list = [
                torch.empty(batch, n, device=device, dtype=DTYPE)
                for _ in range(allocated)
            ]

            fused_us = _benchmark_fused(
                fused_func,
                fused_x_list,
                fused_y_list,
                scale,
                block_dim,
                warmup,
                repeats,
            )
            separate_us = _benchmark_separate(
                hadamard_func,
                quantize_func,
                separate_x_list,
                separate_y_list,
                scale,
                block_dim,
                warmup,
                repeats,
            )
            torch_unfused_us = _benchmark_torch_unfused(
                torch_x_list,
                torch_work_list,
                scale_tensor,
                warmup,
                repeats,
            )

            fused_bytes = batch * n * (2 + 1)
            separate_bytes = batch * n * (4 + 3)
            fused_bw = (fused_bytes / 1e9) / (fused_us / 1e6) if fused_us > 0 else 0.0
            separate_bw = (
                (separate_bytes / 1e9) / (separate_us / 1e6) if separate_us > 0 else 0.0
            )
            fused_speedup_vs_separate = separate_us / fused_us if fused_us > 0 else 0.0
            fused_speedup_vs_unfused = (
                torch_unfused_us / fused_us if fused_us > 0 else 0.0
            )

            print(
                f"{batch:>6d}  {n:>6d}"
                f"  {fused_us:>10.2f}  {separate_us:>12.2f}  {torch_unfused_us:>16.2f}"
                f"  {fused_bw:>14.2f}  {separate_bw:>17.2f}"
                f"  {fused_speedup_vs_separate:>12.3f}  {fused_speedup_vs_unfused:>17.3f}"
            )

            records.append(
                (
                    f"{batch},{n},{scale:.4f},{fused_us:.4f},{separate_us:.4f},{torch_unfused_us:.4f},"
                    f"{fused_bw:.4f},{separate_bw:.4f},{fused_speedup_vs_separate:.4f},"
                    f"{fused_speedup_vs_unfused:.4f}"
                )
            )

    csv_path = output_dir / f"fht_quant_compare_bd{block_dim}.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(
            "batch,N,scale,fused_duration_us,separate_duration_us,torch_unfused_duration_us,"
            "fused_bandwidth_gbs,separate_bandwidth_gbs,"
            "fused_speedup_vs_separate,fused_speedup_vs_torch_unfused\n"
        )
        f.write("\n".join(records) + "\n")
    chmod_output_path(csv_path)
    print(f"\nSaved to {csv_path}")


def main():
    args = _parse_args()

    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.repeats <= 0:
        raise ValueError("--repeats must be > 0")

    torch.npu.set_device(args.npu)
    base = Path(__file__).resolve().parent

    fused_kernel_path = base / "fast_hadamard_quant.cpp"
    hadamard_kernel_path = base / "fast_hadamard.cpp"
    quantize_kernel_path = base / "quantize.cpp"
    csv_dir = Path(args.csv_dir)
    if not csv_dir.is_absolute():
        csv_dir = base / csv_dir

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
    print()

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
