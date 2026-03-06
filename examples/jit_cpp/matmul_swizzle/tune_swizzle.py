import argparse
import os
from pathlib import Path

import pandas as pd
import torch

from jit_util_matmul import jit_compile

DEVICE = os.environ.get("NPU_DEVICE", "npu:0")
DTYPE = torch.float16
M_TILE = 128
DEFAULT_MAX_BLOCK_DIM = int(os.environ.get("PTO_MATMUL_MAX_BLOCK_DIM", "20"))


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep swizzle configs for fixed N/K across a range of M and report "
            "the fastest configuration."
        )
    )
    parser.add_argument("--n", type=int, required=True, help="N dimension")
    parser.add_argument("--k", type=int, required=True, help="K dimension")
    parser.add_argument(
        "--m-min",
        type=int,
        default=M_TILE,
        help=f"Minimum M for sweep (default: {M_TILE})",
    )
    parser.add_argument(
        "--m-max",
        type=int,
        default=4096,
        help="Maximum M for sweep (default: 4096)",
    )
    parser.add_argument(
        "--m-step",
        type=int,
        default=M_TILE,
        help=f"M sweep step (default: {M_TILE})",
    )
    parser.add_argument(
        "--m-values",
        type=str,
        default="",
        help="Comma-separated M values. If set, overrides m-min/m-max/m-step.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup iterations per config (default: 5)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=20,
        help="Measured iterations per config (default: 20)",
    )
    parser.add_argument(
        "--max-block-dim",
        type=int,
        default=DEFAULT_MAX_BLOCK_DIM,
        help=f"Kernel max blockDim (default: {DEFAULT_MAX_BLOCK_DIM})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed used for input generation (default: 1234)",
    )
    return parser.parse_args()


def _build_m_values(args) -> list[int]:
    if args.m_values.strip():
        values = [int(v) for v in args.m_values.split(",") if v.strip()]
        values = sorted(set(values))
    else:
        if args.m_step <= 0:
            raise ValueError("--m-step must be > 0")
        if args.m_max < args.m_min:
            raise ValueError("--m-max must be >= --m-min")
        values = list(range(args.m_min, args.m_max + 1, args.m_step))

    if not values:
        raise ValueError("No valid M values to test")

    if any(m <= 0 for m in values):
        raise ValueError("All M values must be > 0")

    aligned_values = sorted(set(((m + M_TILE - 1) // M_TILE) * M_TILE for m in values))
    if aligned_values != values:
        print(
            f"[INFO] Aligning M values to {M_TILE}-tile boundaries: "
            f"{values} -> {aligned_values}"
        )
    return aligned_values


def _measure_us(
    kernel,
    a,
    b,
    swizzle_direction: int,
    swizzle_count: int,
    max_block_dim: int,
    warmup: int,
    repeat: int,
) -> float:
    for _ in range(warmup):
        kernel(
            a,
            b,
            max_block_dim=max_block_dim,
            swizzle_direction=swizzle_direction,
            swizzle_count=swizzle_count,
        )

    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)

    start.record()
    for _ in range(repeat):
        kernel(
            a,
            b,
            max_block_dim=max_block_dim,
            swizzle_direction=swizzle_direction,
            swizzle_count=swizzle_count,
        )
    end.record()
    torch.npu.synchronize()

    return start.elapsed_time(end) / repeat * 1e3


def main():
    args = _parse_args()

    if args.n <= 0 or args.k <= 0:
        raise ValueError("--n and --k must be > 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.repeat <= 0:
        raise ValueError("--repeat must be > 0")
    if args.max_block_dim <= 0:
        raise ValueError("--max-block-dim must be > 0")

    m_values = _build_m_values(args)
    swizzle_configs = [(d, c) for d in (0, 1) for c in range(13)]

    torch.manual_seed(args.seed)
    torch.npu.set_device(DEVICE)

    base_dir = Path(__file__).resolve().parent
    kernel_cpp = str(base_dir / "matmul_custom_pto.cpp")
    print(f"Compiling kernel: {kernel_cpp}")
    kernel = jit_compile(kernel_cpp, verbose=True)

    records = []

    for m in m_values:
        print(f"\n=== Sweeping M={m}, N={args.n}, K={args.k} ===")
        a = torch.randn(m, args.k, dtype=DTYPE, device=DEVICE)
        b = torch.randn(args.n, args.k, dtype=DTYPE, device=DEVICE)

        best_direction = None
    best_count = None
    best_time_us = None

    for direction, count in swizzle_configs:
        t_us = _measure_us(
            kernel,
            a,
            b,
            swizzle_direction=direction,
            swizzle_count=count,
            max_block_dim=int(args.max_block_dim),
            warmup=int(args.warmup),
            repeat=int(args.repeat),
        )
        records.append(
            {
                "M": m,
                "N": args.n,
                "K": args.k,
                "swizzle_direction": direction,
                "swizzle_count": count,
                "time_us": t_us,
            }
        )
        if best_time_us is None or t_us < best_time_us:
            best_direction = direction
            best_count = count
            best_time_us = t_us

    assert best_direction is not None
    assert best_count is not None
    assert best_time_us is not None

    print(
        "best:"
        f" dir={best_direction},"
        f" count={best_count},"
        f" time={best_time_us:.3f} us"
    )

    df = pd.DataFrame.from_records(records)
    df = df.sort_values(["M", "time_us"], kind="mergesort")
    df["rank_for_M"] = df.groupby("M")["time_us"].rank(method="first")
    df["is_best_for_M"] = df["rank_for_M"] == 1

    summary = (
        df.groupby(["swizzle_direction", "swizzle_count"], as_index=False)
        .agg(
            mean_time_us=("time_us", "mean"),
            median_time_us=("time_us", "median"),
            best_wins=("is_best_for_M", "sum"),
        )
        .sort_values(["mean_time_us", "median_time_us"], kind="mergesort")
    )

    best_overall = summary.iloc[0]
    print("\n=== Overall fastest (by mean runtime over all M) ===")
    print(
        f"swizzle_direction={int(best_overall['swizzle_direction'])}, "
        f"swizzle_count={int(best_overall['swizzle_count'])}, "
        f"mean_time={float(best_overall['mean_time_us']):.3f} us, "
        f"median_time={float(best_overall['median_time_us']):.3f} us, "
        f"wins={int(best_overall['best_wins'])}/{len(m_values)}"
    )

    print("\nTop 5 configs:")
    for rank, (_, row) in enumerate(summary.head(5).iterrows(), start=1):
        print(
            f"  {rank:02d}. dir={int(row['swizzle_direction'])}, "
            f"count={int(row['swizzle_count'])}, "
            f"mean={float(row['mean_time_us']):.3f} us, "
            f"median={float(row['median_time_us']):.3f} us, "
            f"wins={int(row['best_wins'])}"
        )

    csv_dir = base_dir / "outputs" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    detail_path = csv_dir / f"swizzle_sweep_n{args.n}_k{args.k}.csv"
    summary_path = csv_dir / f"swizzle_summary_n{args.n}_k{args.k}.csv"
    df.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved detail CSV:  {detail_path}")
    print(f"Saved summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
