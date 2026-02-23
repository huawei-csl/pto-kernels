import argparse
import csv
import math
import os
from typing import Dict, List

import matplotlib.pyplot as plt


DEFAULT_SINGLE_CSV = "./perf_data/block_rotate_fp16.csv"
DEFAULT_DOUBLE_CSV = "./perf_data_double_buffer/block_rotate_fp16_double_buffer.csv"
DEFAULT_OUTPUT_DIR = "./perf_compare"


def _read_csv(path: str) -> Dict[int, Dict[str, float]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    out: Dict[int, Dict[str, float]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = int(row["M"])
            out[m] = {
                "duration_us": float(row["duration_us"]),
                "bandwidth_gbs": float(row["bandwidth_gbs"]),
            }
    return out


def _median(values: List[float]) -> float:
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return vals[n // 2]
    return (vals[n // 2 - 1] + vals[n // 2]) / 2.0


def _geomean(values: List[float]) -> float:
    vals = [v for v in values if v > 0]
    if not vals:
        return 0.0
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def compare(single_csv: str, double_csv: str, output_dir: str) -> None:
    single = _read_csv(single_csv)
    double = _read_csv(double_csv)

    common_m = sorted(set(single.keys()) & set(double.keys()))
    if not common_m:
        raise RuntimeError(
            "No common M values between single-buffer and double-buffer CSVs."
        )

    os.makedirs(output_dir, exist_ok=True)

    rows = []
    speedups = []
    bw_gains = []

    for m in common_m:
        s_dur = single[m]["duration_us"]
        d_dur = double[m]["duration_us"]
        s_bw = single[m]["bandwidth_gbs"]
        d_bw = double[m]["bandwidth_gbs"]

        speedup = (s_dur / d_dur) if d_dur > 0 else 0.0
        bw_gain = (d_bw / s_bw) if s_bw > 0 else 0.0

        speedups.append(speedup)
        bw_gains.append(bw_gain)

        rows.append(
            {
                "M": m,
                "single_duration_us": s_dur,
                "double_duration_us": d_dur,
                "single_bandwidth_gbs": s_bw,
                "double_bandwidth_gbs": d_bw,
                "speedup_x": speedup,
                "bw_gain_x": bw_gain,
            }
        )

    out_csv = os.path.join(output_dir, "single_vs_double_buffer.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "M",
            "single_duration_us",
            "double_duration_us",
            "single_bandwidth_gbs",
            "double_bandwidth_gbs",
            "speedup_x",
            "bw_gain_x",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Plot 1: duration curves
    ms = [r["M"] for r in rows]
    single_dur = [r["single_duration_us"] for r in rows]
    double_dur = [r["double_duration_us"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    ax = axes[0]
    ax.plot(ms, single_dur, "-", color="#2563eb", linewidth=1.8, label="Single buffer")
    ax.plot(ms, double_dur, "-", color="#dc2626", linewidth=1.8, label="Double buffer")
    ax.set_title("Duration vs M", fontsize=12, fontweight="bold")
    ax.set_xlabel("M")
    ax.set_ylabel("Duration (us)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 2: speedup and bandwidth gain
    ax = axes[1]
    ax.plot(
        ms,
        bw_gains,
        "-",
        color="#9333ea",
        linewidth=1.8,
        label="BW gain (double/single)",
    )
    ax.axhline(1.0, color="#9ca3af", linestyle="--", linewidth=1.0)
    ax.set_title("Relative Gain vs M", fontsize=12, fontweight="bold")
    ax.set_xlabel("M")
    ax.set_ylabel("Ratio (x)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle(
        "Block Rotate FP16: Single vs Double Buffer", fontsize=14, fontweight="bold"
    )
    fig.tight_layout()

    out_png = os.path.join(output_dir, "single_vs_double_buffer.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    best_idx = max(range(len(rows)), key=lambda i: rows[i]["speedup_x"])
    best_row = rows[best_idx]

    print("=" * 72)
    print("Single vs Double Buffer Comparison")
    print("=" * 72)
    print(f"Common points      : {len(rows)}")
    print(f"Median speedup     : {_median(speedups):.3f}x")
    print(f"Geomean speedup    : {_geomean(speedups):.3f}x")
    print(f"Median BW gain     : {_median(bw_gains):.3f}x")
    print(f"Geomean BW gain    : {_geomean(bw_gains):.3f}x")
    print(
        f"Best speedup       : {best_row['speedup_x']:.3f}x at M={best_row['M']} "
        f"(single={best_row['single_duration_us']:.2f}us, double={best_row['double_duration_us']:.2f}us)"
    )
    print(f"Saved comparison CSV : {out_csv}")
    print(f"Saved comparison plot: {out_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare block_rotate single-buffer vs double-buffer benchmark CSVs."
    )
    parser.add_argument("--single-csv", default=DEFAULT_SINGLE_CSV)
    parser.add_argument("--double-csv", default=DEFAULT_DOUBLE_CSV)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    compare(args.single_csv, args.double_csv, args.output_dir)
