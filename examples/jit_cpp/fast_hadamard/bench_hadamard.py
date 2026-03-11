import argparse
import math
from pathlib import Path

import torch
import torch_npu  # noqa

from jit_util_hadamard import chmod_output_path, jit_compile, normalize_npu_device

BENCH_BATCHES = [1, 5, 8, 10, 16, 20, 32, 40, 64, 128, 256, 512, 1024]
BENCH_HIDDEN_DIMS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]

DEFAULT_DEVICE = "npu:0"
DTYPE = torch.float16
DEFAULT_WARMUP = 2
DEFAULT_REPEATS = 20
DEFAULT_CSV_DIR = Path("outputs") / "csv"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Fast Hadamard PTO kernel and save CSV outputs."
    )
    parser.add_argument(
        "--npu",
        type=normalize_npu_device,
        default=DEFAULT_DEVICE,
        help="NPU device (examples: 0, npu:0, '0', 'npu:0').",
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


def benchmark(hadamard_func, warmup: int, repeats: int, output_dir: Path, device: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    chmod_output_path(output_dir)
    block_dim = hadamard_func.block_dim

    print(f"\n{'=' * 60}")
    print(f"BENCHMARK (BLOCK_DIM={block_dim})")
    print(f"{'=' * 60}")
    header = (
        f"{'batch':>6s}  {'N':>6s}" f"  {'duration_us':>12s}  {'bandwidth_gbs':>14s}"
    )
    print(header)
    print("-" * len(header))

    records = []

    for batch in BENCH_BATCHES:
        for n in BENCH_HIDDEN_DIMS:
            log2_n = int(math.log2(n))
            allocated = warmup + repeats

            # Use separate tensors per launch to reduce cache reuse artifacts.
            x_list = [
                torch.randn(batch, n, device=device, dtype=DTYPE)
                for _ in range(allocated)
            ]

            for i in range(warmup):
                hadamard_func(x_list[i], batch, n, log2_n)
            torch.npu.synchronize()

            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)

            start.record()
            for i in range(repeats):
                hadamard_func(x_list[warmup + i], batch, n, log2_n)
            end.record()
            torch.npu.synchronize()

            duration_ms = start.elapsed_time(end) / repeats
            dur_us = duration_ms * 1e3

            data_bytes = 2 * batch * n * 2
            bw_gbs = (data_bytes / 1e9) / (dur_us / 1e6) if dur_us > 0 else 0.0

            print(f"{batch:>6d}  {n:>6d}  {dur_us:>12.2f}  {bw_gbs:>14.2f}")
            records.append(f"{batch},{n},{dur_us:.4f},{bw_gbs:.4f}")

    csv_path = output_dir / f"fht_pto_bd{block_dim}.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("batch,N,duration_us,bandwidth_gbs\n")
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

    kernel_path = base / "fast_hadamard.cpp"
    csv_dir = Path(args.csv_dir)
    if not csv_dir.is_absolute():
        csv_dir = base / csv_dir

    print(f"Using device: {args.npu}")
    print("Compiling fast_hadamard.cpp ...")
    hadamard_func = jit_compile(str(kernel_path), verbose=True, device=args.npu)
    print()

    benchmark(
        hadamard_func,
        warmup=args.warmup,
        repeats=args.repeats,
        output_dir=csv_dir,
        device=args.npu,
    )


if __name__ == "__main__":
    main()
