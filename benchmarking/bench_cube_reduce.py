# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved. See LICENSE in the root of the repository.

"""Compare torch.sum(x) with run_cube_reduce (Python: pto_cube_reduce).

Example (with the extension built and installed):
    python benchmarking/bench_cube_reduce.py --device npu:1

The default sweep covers 1,024 through 99,000,000 elements for FP16, FP32 and
INT8. Use --sizes to select specific lengths, for example --sizes 99000000.
Both reductions output FP32 for FP16 and FP32 input, and INT32 for INT8 input.

Timings use NPU events and include all device work in each public call,
including cube-reduce padding, temporary initialization, and final reduction.
Bandwidth counts input bytes only, rather than internal temporary traffic.
"""

import argparse
import csv
import os
from pathlib import Path

import torch
import torch_npu  # noqa: F401 -- registers the NPU backend

from utils import do_bench


DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "int8": torch.int8,
}


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def bench(args):
    from pto_kernels import pto_cube_reduce

    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = False
    torch.manual_seed(args.seed)
    rows = []
    print("Times: NPU event latency; bandwidth: input bytes / latency.")
    print("Speedup = torch.sum latency / pto_cube_reduce latency.")
    print(
        f"{'dtype':>8} {'numel':>10} {'torch out':>10} {'cube out':>10} "
        f"{'torch us':>12} {'cube us':>12} {'speedup':>9}"
    )
    for dtype_name in args.dtypes:
        dtype = DTYPES[dtype_name]
        is_int8 = dtype == torch.int8
        output_dtype = torch.int32 if is_int8 else torch.float32
        for numel in args.sizes:
            # Generate on CPU so correctness has an independent reference.
            if is_int8:
                host = torch.randint(-3, 3, (numel,), dtype=dtype)
            else:
                host = (0.1 * torch.randn(numel)).to(dtype)
            x = host.to(args.device)

            def torch_sum():
                return torch.sum(x, dtype=output_dtype)

            def cube_reduce():
                return pto_cube_reduce(x)

            reference = host.sum(dtype=torch.int64 if is_int8 else torch.float64)
            torch_result = torch_sum()
            cube_result = cube_reduce()
            for result in (torch_result, cube_result):
                tolerance = 0 if is_int8 else 0.01
                torch.testing.assert_close(
                    result.cpu().reshape(()),
                    reference.to(result.dtype),
                    atol=tolerance,
                    rtol=tolerance,
                )

            timing = dict(
                warmup_iters=args.warmup,
                benchmark_iters=args.iters,
                flush_cache=not args.no_flush_cache,
                unit="us",
            )
            torch_us = do_bench(torch_sum, **timing)
            cube_us = do_bench(cube_reduce, **timing)
            if torch_us <= 0 or cube_us <= 0:
                raise RuntimeError("NPU event latency must be positive")
            speedup = torch_us / cube_us
            torch_dtype = str(torch_result.dtype).removeprefix("torch.")
            cube_dtype = str(cube_result.dtype).removeprefix("torch.")
            rows.append(
                dict(
                    dtype=dtype_name,
                    numel=numel,
                    torch_output_dtype=torch_dtype,
                    cube_output_dtype=cube_dtype,
                    torch_time_us=torch_us,
                    cube_time_us=cube_us,
                    speedup=speedup,
                    torch_bandwidth_gbps=x.nbytes / torch_us / 1e3,
                    cube_bandwidth_gbps=x.nbytes / cube_us / 1e3,
                    device=str(x.device),
                    warmup_iters=args.warmup,
                    benchmark_iters=args.iters,
                    flush_cache=not args.no_flush_cache,
                )
            )
            print(
                f"{dtype_name:>8} {numel:>10} {torch_dtype:>10} {cube_dtype:>10} "
                f"{torch_us:>12.3f} {cube_us:>12.3f} {speedup:>8.2f}x"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=os.environ.get("NPU_DEVICE", "npu:1"))
    parser.add_argument("--dtypes", nargs="+", choices=DTYPES, default=list(DTYPES))
    parser.add_argument(
        "--sizes",
        type=positive_int,
        nargs="+",
        default=[
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
            65536,
            131072,
            262144,
            327680,
            327681,
            524288,
            1048576,
            2097152,
            4194304,
            8388608,
            12582912,
            16777216,
            16777217,
            25165824,
            33554432,
            50331648,
            67108864,
            83886080,
            99000000,
        ],
        help="input element counts (default: aligned and padded cases up to 99 million)",
    )
    parser.add_argument("--warmup", type=positive_int, default=10)
    parser.add_argument("--iters", type=positive_int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-flush-cache", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("benchmark_data/cube_reduce.csv")
    )
    bench(parser.parse_args())


if __name__ == "__main__":
    main()
