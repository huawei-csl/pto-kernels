# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved. See LICENSE in the root of the repository.

"""Compare torch.linalg.vector_norm(x) with run_cube_norm (Python: pto_cube_norm).

Example (with the extension built and installed):
    python benchmarking/bench_cube_norm.py --device npu:1

The default sweep covers 1,024 through 99,000,000 elements for FP16 and FP32.
Use --sizes to select specific lengths, for example --sizes 99000000.

int8 is also supported, but two things about its row differ. Its lengths stop
at the 8,388,544 elements the kernel accepts before its int32 accumulator
could wrap, and torch has no int8 vector_norm, so the baseline it is compared
against is vector_norm of an fp32 copy and includes that cast.

Both entry points read every element once and return a scalar, so the
interesting column is GB/s: the kernel does S times more multiply-accumulates
than a plain reduction (an S x S matrix product per S x S tile), which the Cube
unit has to absorb without falling behind HBM.

Timings use NPU events and include all device work in each public call,
including the partials allocation and the second launch that finishes the
reduction. The L2 cache is flushed before every timed iteration, so the numbers
are cold-cache. Bandwidth counts input bytes only, rather than internal
temporary traffic.
"""

import argparse
import csv
import os
from pathlib import Path

import torch
import torch_npu  # noqa: F401  # pylint: disable=unused-import

from utils import do_bench

DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "int8": torch.int8,
}

# Longest int8 input run_cube_norm accepts; see its int32 accumulator bound.
INT8_MAX_NUMEL = 64 * ((2**31 - 1) // (128 * 128))


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def bench(args):
    from pto_kernels import pto_cube_norm

    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = False
    torch.manual_seed(args.seed)
    rows = []
    print("Times: NPU event latency, cold cache; bandwidth: input bytes / latency.")
    print("Speedup = torch.linalg.vector_norm latency / pto_cube_norm latency.")
    print(
        f"{'dtype':>8} {'numel':>10} {'torch us':>12} {'cube us':>12} "
        f"{'torch GB/s':>11} {'cube GB/s':>11} {'speedup':>9}"
    )
    for dtype_name in args.dtypes:
        dtype = DTYPES[dtype_name]
        for numel in args.sizes:
            if dtype == torch.int8 and numel > INT8_MAX_NUMEL:
                continue  # past what the int32 accumulator can hold
            # Generate on CPU so correctness has an independent reference, and
            # keep the scale small so fp16 squares stay far from overflow.
            if dtype == torch.int8:
                host = torch.randint(-128, 128, (numel,), dtype=torch.int8)
            else:
                host = (0.1 * torch.randn(numel)).to(dtype)
            x = host.to(args.device)

            if dtype == torch.int8:
                # torch has no int8 vector_norm, so the baseline is the norm
                # of an fp32 copy and is timed with that cast included.
                def torch_norm(x=x):
                    return torch.linalg.vector_norm(x.to(torch.float32))

            else:

                def torch_norm(x=x):
                    return torch.linalg.vector_norm(x)

            def cube_norm(x=x):
                return pto_cube_norm(x)

            reference = torch.linalg.vector_norm(host.to(torch.float64))
            for result in (torch_norm(), cube_norm()):
                torch.testing.assert_close(
                    result.cpu().to(torch.float64).reshape(()),
                    reference,
                    atol=1e-3 * reference.item(),
                    rtol=1e-3,
                )

            timing = {
                "warmup_iters": args.warmup,
                "benchmark_iters": args.iters,
                "unit": "us",
            }
            torch_us = do_bench(torch_norm, **timing)
            cube_us = do_bench(cube_norm, **timing)
            if torch_us <= 0 or cube_us <= 0:
                raise RuntimeError("NPU event latency must be positive")
            speedup = torch_us / cube_us
            torch_gbps = x.nbytes / torch_us / 1e3
            cube_gbps = x.nbytes / cube_us / 1e3
            rows.append(
                {
                    "dtype": dtype_name,
                    "numel": numel,
                    "input_mib": x.nbytes / 2**20,
                    "torch_time_us": torch_us,
                    "cube_time_us": cube_us,
                    "speedup": speedup,
                    "torch_bandwidth_gbps": torch_gbps,
                    "cube_bandwidth_gbps": cube_gbps,
                    "device": str(x.device),
                    "warmup_iters": args.warmup,
                    "benchmark_iters": args.iters,
                }
            )
            print(
                f"{dtype_name:>8} {numel:>10} {torch_us:>12.3f} {cube_us:>12.3f} "
                f"{torch_gbps:>11.1f} {cube_gbps:>11.1f} {speedup:>8.2f}x"
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
            4096,
            16384,
            65536,
            262144,
            327681,
            1048576,
            4194304,
            16777216,
            16777217,
            33554432,
            67108864,
            99000000,
        ],
        help="input element counts (default: aligned and padded cases up to 99 million)",
    )
    parser.add_argument("--warmup", type=positive_int, default=10)
    parser.add_argument("--iters", type=positive_int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmark_data/cube_norm.csv")
    )
    bench(parser.parse_args())


if __name__ == "__main__":
    main()
