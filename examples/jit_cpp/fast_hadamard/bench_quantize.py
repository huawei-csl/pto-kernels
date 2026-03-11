import argparse
from pathlib import Path

import torch
import torch_npu  # noqa

from jit_util_hadamard import chmod_output_path, normalize_npu_device
from jit_util_quantize import jit_compile

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
        description="Benchmark PTO quantize kernel against torch_npu.npu_quantize."
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


def _make_scale_tensor(scale: float, device: str):
    return torch.tensor([scale], device=device, dtype=DTYPE)


def _torch_npu_quantize(x, scale_tensor):
    return torch_npu.npu_quantize(x, scale_tensor, None, torch.qint8, -1, False)


def _benchmark_pto(quantize_func, x_list, y_list, scale, block_dim, warmup, repeats):
    for i in range(warmup):
        quantize_func(x_list[i], y_list[i], scale, block_dim=block_dim)
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)

    start.record()
    for i in range(repeats):
        quantize_func(
            x_list[warmup + i],
            y_list[warmup + i],
            scale,
            block_dim=block_dim,
        )
    end.record()
    torch.npu.synchronize()

    return start.elapsed_time(end) / repeats * 1e3


def _benchmark_torch_npu(x_list, scale_tensor, warmup, repeats):
    for i in range(warmup):
        _torch_npu_quantize(x_list[i], scale_tensor)
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)

    start.record()
    for i in range(repeats):
        _torch_npu_quantize(x_list[warmup + i], scale_tensor)
    end.record()
    torch.npu.synchronize()

    return start.elapsed_time(end) / repeats * 1e3


def benchmark(
    quantize_func,
    scale: float,
    warmup: int,
    repeats: int,
    output_dir: Path,
    device: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    chmod_output_path(output_dir)

    block_dim = int(
        getattr(
            torch.npu.get_device_properties(device),
            "cube_core_num",
            quantize_func.block_dim,
        )
    )
    scale_tensor = _make_scale_tensor(scale, device)

    print(f"\n{'=' * 92}")
    print(f"QUANTIZE BENCHMARK (BLOCK_DIM={block_dim}, scale={scale})")
    print(f"{'=' * 92}")
    header = (
        f"{'batch':>6s}  {'N':>6s}"
        f"  {'pto_us':>10s}  {'torch_npu_us':>14s}"
        f"  {'pto_bw_gbs':>12s}  {'torch_npu_bw_gbs':>18s}  {'pto_speedup':>11s}"
    )
    print(header)
    print("-" * len(header))

    records = []

    for batch in BENCH_BATCHES:
        for n in BENCH_HIDDEN_DIMS:
            allocated = warmup + repeats
            x_list = [
                torch.randn(batch, n, device=device, dtype=DTYPE)
                for _ in range(allocated)
            ]
            y_list = [
                torch.empty(batch, n, device=device, dtype=torch.int8)
                for _ in range(allocated)
            ]

            pto_us = _benchmark_pto(
                quantize_func,
                x_list,
                y_list,
                scale,
                block_dim,
                warmup,
                repeats,
            )
            torch_npu_us = _benchmark_torch_npu(x_list, scale_tensor, warmup, repeats)

            data_bytes = batch * n * (2 + 1)
            pto_bw = (data_bytes / 1e9) / (pto_us / 1e6) if pto_us > 0 else 0.0
            torch_npu_bw = (
                (data_bytes / 1e9) / (torch_npu_us / 1e6) if torch_npu_us > 0 else 0.0
            )
            pto_speedup = torch_npu_us / pto_us if pto_us > 0 else 0.0

            print(
                f"{batch:>6d}  {n:>6d}"
                f"  {pto_us:>10.2f}  {torch_npu_us:>14.2f}"
                f"  {pto_bw:>12.2f}  {torch_npu_bw:>18.2f}  {pto_speedup:>11.3f}"
            )

            records.append(
                (
                    f"{batch},{n},{scale:.4f},{pto_us:.4f},{torch_npu_us:.4f},"
                    f"{pto_bw:.4f},{torch_npu_bw:.4f},{pto_speedup:.4f}"
                )
            )

    csv_path = output_dir / f"quantize_compare_bd{block_dim}.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(
            "batch,N,scale,pto_duration_us,torch_npu_duration_us,"
            "pto_bandwidth_gbs,torch_npu_bandwidth_gbs,pto_speedup_vs_torch_npu\n"
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

    kernel_path = base / "quantize.cpp"
    csv_dir = Path(args.csv_dir)
    if not csv_dir.is_absolute():
        csv_dir = base / csv_dir

    print(f"Using device: {args.npu}")
    print("Compiling quantize.cpp ...")
    quantize_func = jit_compile(str(kernel_path), verbose=True)
    print()

    benchmark(
        quantize_func,
        scale=args.scale,
        warmup=args.warmup,
        repeats=args.repeats,
        output_dir=csv_dir,
        device=args.npu,
    )


if __name__ == "__main__":
    main()
