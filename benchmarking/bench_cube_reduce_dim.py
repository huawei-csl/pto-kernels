# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved. See LICENSE in the root of the repository.

"""Compare torch.sum(x, dim) with run_cube_reduce_dim (Python: pto_cube_reduce_dim).

Example (with the extension built and installed):
    python benchmarking/bench_cube_reduce_dim.py --device npu:0

The kernel folds the axes around the reduced one into an (outer, mid, inner)
view that always reduces `mid`, and it tiles two of those extents, so what a
shape costs depends on those extents rather than on which dim was named. The
default sweep therefore varies one extent at a time at a roughly constant
element count:

  kept-inner  dim=1 with mid fixed, varying the kept contiguous extent.
  reduced     dim=1 with the kept extent fixed, varying the reduced extent.
  contiguous  dim=2, where the reduced axis is the contiguous one.
  rank        every axis of a rank-4 and a rank-5 shape, each paired with the
              3D shape its axes fold into.
  shapes      assorted whole shapes across all three dims.

Both reductions output FP32 for FP16 and FP32 input, and INT32 for INT8 input.

Timings use NPU events and include all device work in each public call.
Bandwidth counts input bytes only, rather than internal temporary traffic.
A reduction whose tiles would be cut across an axis longer than the tile
load's 16-bit row pitch is rejected by the host; those rows report no timing.
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

# Element count each sweep aims for, so that rows differ in shape rather than
# in how much memory they move.
SWEEP_NUMEL = 4 * 1024 * 1024


def sweep_shapes():
    """The default (group, shape, dim) sweep."""
    cases = []
    for inner in (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384):
        outer = max(1, SWEEP_NUMEL // (256 * inner))
        cases.append(("kept-inner", (outer, 256, inner), 1))
    for mid in (8, 16, 32, 64, 128, 256, 512, 1024):
        outer = max(1, SWEEP_NUMEL // (mid * 4096))
        cases.append(("reduced", (outer, mid, 4096), 1))
    for mid in (8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384):
        outer = max(1, SWEEP_NUMEL // mid)
        cases.append(("contiguous", (outer, 1, mid), 2))
    # Rank never reaches the kernel: the host folds the axes around the reduced
    # one into (outer, mid, inner) before launching, so a rank-4 or rank-5
    # reduction runs the same tiles as the 3D shape its axes fold into. What
    # the fold does not absorb is the host's own work -- two more loop
    # iterations, a longer output shape, a contiguity check over more strides --
    # which is a real share of the latency on small inputs. Each case is
    # therefore emitted next to its folded equivalent, so the pair can be read
    # as two adjacent rows carrying identical extents: a gap between them is
    # host-side, since the device work is identical by construction.
    for shape in ((512, 64, 16, 8), (256, 32, 16, 8, 4)):
        for dim in range(len(shape)):
            cases.append(("rank", shape, dim))
            cases.append(("rank", extents(shape, dim), 1))
    for shape in ((256, 256, 64), (8, 512, 512), (64, 128, 1024), (1024, 64, 64)):
        for dim in (0, 1, 2):
            cases.append(("shapes", shape, dim))
    return cases


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def extents(shape, dim):
    """The (outer, mid, inner) view the kernel reduces."""
    axis = dim % len(shape)
    outer, inner = 1, 1
    for size in shape[:axis]:
        outer *= size
    for size in shape[axis + 1 :]:
        inner *= size
    return outer, shape[axis], inner


def make_input(shape, dtype):
    # Generate on CPU so correctness has an independent reference.
    if dtype == torch.int8:
        return torch.randint(-3, 3, shape, dtype=dtype)
    return (0.1 * torch.randn(shape)).to(dtype)


def bench(args):
    from pto_kernels import pto_cube_reduce_dim

    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = False
    torch.manual_seed(args.seed)
    cases = [case for case in sweep_shapes() if case[0] in args.groups]
    rows = []
    print("Times: NPU event latency; bandwidth: input bytes / latency.")
    print("Speedup = torch.sum latency / pto_cube_reduce_dim latency.")
    print(
        f"{'group':>11} {'dtype':>8} {'shape':>20} {'dim':>4} "
        f"{'outer':>8} {'mid':>6} {'inner':>6} "
        f"{'torch us':>10} {'pto us':>10} {'speedup':>8} {'pto GB/s':>9}"
    )
    for dtype_name in args.dtypes:
        dtype = DTYPES[dtype_name]
        is_int8 = dtype == torch.int8
        output_dtype = torch.int32 if is_int8 else torch.float32
        for group, shape, dim in cases:
            outer, mid, inner = extents(shape, dim)
            host = make_input(shape, dtype)
            x = host.to(args.device)

            def torch_sum(x=x, dim=dim, output_dtype=output_dtype):
                return torch.sum(x, dim=dim, dtype=output_dtype)

            def pto_reduce(x=x, dim=dim):
                return pto_cube_reduce_dim(x, dim)

            try:
                pto_result = pto_reduce()
            except RuntimeError as error:
                if "row pitch" not in str(error):
                    raise
                print(
                    f"{group:>11} {dtype_name:>8} {str(shape):>20} {dim:>4} "
                    f"{outer:>8} {mid:>6} {inner:>6} "
                    f"{'rejected: row pitch over the 16-bit limit':>50}"
                )
                rows.append(
                    {
                        "group": group,
                        "dtype": dtype_name,
                        "shape": "x".join(str(size) for size in shape),
                        "dim": dim,
                        "outer": outer,
                        "mid": mid,
                        "inner": inner,
                        "torch_time_us": "",
                        "pto_time_us": "",
                        "speedup": "",
                        "torch_bandwidth_gbps": "",
                        "pto_bandwidth_gbps": "",
                        "rejected": "row_pitch",
                        "device": str(x.device),
                        "warmup_iters": args.warmup,
                        "benchmark_iters": args.iters,
                        "flush_cache": not args.no_flush_cache,
                    }
                )
                continue

            reference = host.sum(
                dim=dim, dtype=torch.int64 if is_int8 else torch.float64
            )
            for result in (torch_sum(), pto_result):
                tolerance = 0 if is_int8 else 0.01
                torch.testing.assert_close(
                    result.cpu(),
                    reference.to(result.dtype),
                    atol=tolerance,
                    rtol=tolerance,
                )

            timing = {
                "warmup_iters": args.warmup,
                "benchmark_iters": args.iters,
                "flush_cache": not args.no_flush_cache,
                "unit": "us",
            }
            torch_us = do_bench(torch_sum, **timing)
            pto_us = do_bench(pto_reduce, **timing)
            if torch_us <= 0 or pto_us <= 0:
                raise RuntimeError("NPU event latency must be positive")
            speedup = torch_us / pto_us
            rows.append(
                {
                    "group": group,
                    "dtype": dtype_name,
                    "shape": "x".join(str(size) for size in shape),
                    "dim": dim,
                    "outer": outer,
                    "mid": mid,
                    "inner": inner,
                    "torch_time_us": torch_us,
                    "pto_time_us": pto_us,
                    "speedup": speedup,
                    "torch_bandwidth_gbps": x.nbytes / torch_us / 1e3,
                    "pto_bandwidth_gbps": x.nbytes / pto_us / 1e3,
                    "rejected": "",
                    "device": str(x.device),
                    "warmup_iters": args.warmup,
                    "benchmark_iters": args.iters,
                    "flush_cache": not args.no_flush_cache,
                }
            )
            print(
                f"{group:>11} {dtype_name:>8} {str(shape):>20} {dim:>4} "
                f"{outer:>8} {mid:>6} {inner:>6} "
                f"{torch_us:>10.3f} {pto_us:>10.3f} {speedup:>7.2f}x "
                f"{x.nbytes / pto_us / 1e3:>9.1f}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=os.environ.get("NPU_DEVICE", "npu:0"))
    parser.add_argument("--dtypes", nargs="+", choices=DTYPES, default=list(DTYPES))
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=["kept-inner", "reduced", "contiguous", "rank", "shapes"],
        default=["kept-inner", "reduced", "contiguous", "rank", "shapes"],
        help="which sweeps to run (default: all)",
    )
    parser.add_argument("--warmup", type=positive_int, default=10)
    parser.add_argument("--iters", type=positive_int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-flush-cache", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("benchmark_data/cube_reduce_dim.csv")
    )
    bench(parser.parse_args())


if __name__ == "__main__":
    main()
