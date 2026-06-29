#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import ctypes
import os
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch
import torch_npu  # noqa: F401

HERE = Path(__file__).resolve().parent
ADD_TILE_ELEMS = 24576  # 48 KiB UB tile / sizeof(fp16), matches add.cpp

sys.path.insert(0, str(HERE.parents[1]))
from pto_demo_utils import (  # noqa: E402
    compile_kernel,
    configure_torch_npu,
    cube_core_count,
    stream_ptr,
    tensor_ptr,
    vector_core_count,
)


def load_add() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(compile_kernel(HERE / "compile.sh", "add")))
    lib.call_add.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    return lib


def load_matmul() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(compile_kernel(HERE / "compile.sh", "matmul")))
    lib.call_matmul.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    return lib


def event_bench(
    fn: Callable[[], None],
    *,
    warmup: int,
    repeats: int,
    flush_cache: bool,
) -> list[float]:
    cache = (
        torch.empty((256 * 1024 * 1024,), dtype=torch.int8, device="npu")
        if flush_cache
        else None
    )
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()

    samples_us: list[float] = []
    for _ in range(repeats):
        if cache is not None:
            cache.zero_()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1e3)
    return samples_us


def summarize(samples_us: list[float]) -> tuple[float, float, float]:
    median = statistics.median(samples_us)
    mean = statistics.mean(samples_us)
    stdev = statistics.pstdev(samples_us) if len(samples_us) > 1 else 0.0
    return median, mean, stdev


def bench_add(args, device: str, block_dim: int) -> list[dict]:
    lib = load_add()
    rows = []
    for n in args.add_sizes:
        torch.manual_seed(0)
        x = torch.randn(n, device=device, dtype=torch.float16)
        z = torch.randn(n, device=device, dtype=torch.float16)
        y = torch.empty_like(x)
        # Do not spread tiny vectors across all physical cores: inactive blocks can
        # still issue with invalid tiny tile sizes on this sample kernel.
        active_blocks = min(
            block_dim, max(1, (n + ADD_TILE_ELEMS - 1) // ADD_TILE_ELEMS)
        )
        stream = stream_ptr()

        def launch() -> None:
            lib.call_add(
                active_blocks, stream, tensor_ptr(y), tensor_ptr(x), tensor_ptr(z), n
            )

        samples = event_bench(
            launch,
            warmup=args.warmup,
            repeats=args.repeats,
            flush_cache=args.flush_cache,
        )
        median, mean, stdev = summarize(samples)
        bytes_moved = n * 3 * 2
        rows.append(
            {
                "kernel": "add",
                "shape": f"n={n}",
                "block_dim": active_blocks,
                "median_us": median,
                "mean_us": mean,
                "stdev_us": stdev,
                "effective_gbs": bytes_moved / median * 1e-3,
            }
        )
    return rows


def bench_matmul(args, device: str, max_block_dim: int) -> list[dict]:
    lib = load_matmul()
    rows = []
    for m in args.matmul_m:
        if m % 128 != 0:
            raise ValueError(f"matmul M must be a multiple of 128, got {m}")
        torch.manual_seed(1)
        a = torch.randn(m, 128, device=device, dtype=torch.float16)
        b = torch.randn(128, 128, device=device, dtype=torch.float16)
        c = torch.empty(m, 128, device=device, dtype=torch.float16)
        block_dim = min(max_block_dim, max(1, m // 128))
        stream = stream_ptr()

        def launch() -> None:
            lib.call_matmul(
                block_dim, stream, tensor_ptr(a), tensor_ptr(b), tensor_ptr(c), m
            )

        samples = event_bench(
            launch,
            warmup=args.warmup,
            repeats=args.repeats,
            flush_cache=args.flush_cache,
        )
        median, mean, stdev = summarize(samples)
        flops = 2 * m * 128 * 128
        bytes_moved = (m * 128 + 128 * 128 + m * 128) * 2
        rows.append(
            {
                "kernel": "simple_matmul",
                "shape": f"m={m},k=128,n=128",
                "block_dim": block_dim,
                "median_us": median,
                "mean_us": mean,
                "stdev_us": stdev,
                "effective_gbs": bytes_moved / median * 1e-3,
                "effective_tflops": flops / median * 1e-6,
            }
        )
    return rows


def print_rows(rows: list[dict]) -> None:
    header = (
        f"{'kernel':<15} {'shape':<20} {'block':>5} {'median_us':>10} "
        f"{'mean_us':>10} {'stdev_us':>10} {'GB/s':>10} {'TFLOP/s':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['kernel']:<15} {row['shape']:<20} {row['block_dim']:>5} "
            f"{row['median_us']:>10.2f} {row['mean_us']:>10.2f} "
            f"{row['stdev_us']:>10.2f} {row['effective_gbs']:>10.2f} "
            f"{row.get('effective_tflops', 0.0):>10.4f}"
        )


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "kernel",
        "shape",
        "block_dim",
        "median_us",
        "mean_us",
        "stdev_us",
        "effective_gbs",
        "effective_tflops",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark local A2A3 PTO demo kernels"
    )
    parser.add_argument("--device", default=os.environ.get("NPU_DEVICE", "npu:0"))
    parser.add_argument("--kernel", choices=("add", "matmul", "all"), default="all")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--add-sizes", type=int, nargs="*", default=[4096, 65536, 1048576]
    )
    parser.add_argument("--matmul-m", type=int, nargs="*", default=[128, 1024, 4096])
    parser.add_argument("--flush-cache", action="store_true")
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("--warmup must be >= 0 and --repeats must be > 0")

    configure_torch_npu()
    torch.npu.set_device(args.device)
    cube_block_dim = cube_core_count(args.device)
    vector_block_dim = vector_core_count(args.device)
    rows: list[dict] = []
    if args.kernel in ("add", "all"):
        rows.extend(bench_add(args, args.device, vector_block_dim))
    if args.kernel in ("matmul", "all"):
        rows.extend(bench_matmul(args, args.device, cube_block_dim))
    print_rows(rows)
    if args.csv:
        write_csv(rows, args.csv)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
