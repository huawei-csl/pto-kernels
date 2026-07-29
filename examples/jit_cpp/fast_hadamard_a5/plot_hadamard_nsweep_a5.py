#!/usr/bin/env python3
"""Plot the fast_hadamard_256_a5 block-size (N) sweep.

Reads the CSV emitted by ``benchmark.py --nsweep`` (columns n,chunks,rows,batch,
rel_err,had_gbs,copy_gbs,ratio) and renders two panels:

  * achieved bandwidth vs N against the measured copy floor, and
  * the fraction of that floor reached, and
  * the vector-op cost per element, which explains why the ratio is now flat:
    packing keeps all 128 lanes busy at every N, so cost no longer blows up
    as N shrinks.

Three single-scale panels rather than two with a twinned y-axis: the op count and
the bandwidth ratio share no units, and overlaying them would invite reading a
crossing point that means nothing.

Self-contained (matplotlib only): the shared plot_common helper imports
jit_util_common, which pulls in torch, and the matplotlib-only plotting env has
no torch.
"""

import argparse
import csv
import math
import sys
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

DEFAULT_CSV = Path("build") / "nsweep256.csv"
DEFAULT_PLOT_NAME = "hadamard_nsweep.png"
HAD_COLOR = "#2f6df6"
FLOOR_COLOR = "#8b929b"
OPS_COLOR = "#b8620a"
VEC_F16_LANES = 128


def _parse_args():
    parser = argparse.ArgumentParser(description="Plot the N sweep.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"CSV from benchmark.py --nsweep (default: {DEFAULT_CSV}).",
    )
    parser.add_argument(
        "--plot-name",
        type=str,
        default=DEFAULT_PLOT_NAME,
        help=f"Output PNG filename (default: {DEFAULT_PLOT_NAME}).",
    )
    return parser.parse_args()


def _load(csv_path: Path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            if not record.get("n") or not record.get("had_gbs"):
                continue  # skip rows a correctness gate rejected
            rows.append(
                {
                    "n": int(record["n"]),
                    "had": float(record["had_gbs"]),
                    "copy": float(record["copy_gbs"]),
                    "ratio": float(record["ratio"]),
                }
            )
    rows.sort(key=lambda r: r["n"])
    return rows


def _ops_per_element(n: int) -> float:
    """Vector ops per element: 5 per stage (vlds, vadd, vsub, 2x vsts) over a
    256-element window, plus log2(R) vdintlv to undo the packing rotation.

    For N < 256 the kernel packs R = 256/N rows per window, so all 128 lanes are
    driven regardless of N and the per-stage cost is amortised over 256 elements
    rather than over N. For N > 256 a row spans CHUNKS windows, so the stage
    count keeps growing while the per-window cost stays flat.
    """
    log2n = int(math.log2(n))
    pack = max(1, 256 // n)
    return (5 * log2n + int(math.log2(pack))) / 256.0


def _draw_bandwidth(axis, rows):
    x = [r["n"] for r in rows]
    axis.plot(
        x,
        [r["copy"] / 1000 for r in rows],
        "--",
        marker="o",
        color=FLOOR_COLOR,
        linewidth=2,
        markersize=8,
        label="copy floor (DMA ceiling)",
    )
    axis.plot(
        x,
        [r["had"] / 1000 for r in rows],
        "-",
        marker="o",
        color=HAD_COLOR,
        linewidth=2,
        markersize=8,
        label="hadamard",
    )
    axis.set_xscale("log", base=2)
    axis.set_xticks(x)
    axis.set_xticklabels([str(v) for v in x])
    axis.set_xlabel("block size N")
    axis.set_ylabel("bandwidth (TB/s)")
    axis.set_ylim(0, max(r["copy"] for r in rows) / 1000 * 1.12)
    axis.set_title("Bandwidth vs block size")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="lower right", frameon=False)
    # label only the endpoints and the shipped default, not every point
    for r in rows:
        if r["n"] in (min(x), 256, max(x)):
            axis.annotate(
                f"{r['had'] / 1000:.2f}",
                (r["n"], r["had"] / 1000),
                textcoords="offset points",
                xytext=(0, -17),
                ha="center",
                fontsize=9,
                color=HAD_COLOR,
            )


def _draw_ratio(axis, rows):
    x = [r["n"] for r in rows]
    axis.plot(
        x,
        [r["ratio"] for r in rows],
        "-",
        marker="o",
        color=HAD_COLOR,
        linewidth=2,
        markersize=8,
        label="hadamard / copy",
    )
    axis.axhline(1.0, color=FLOOR_COLOR, linestyle="--", linewidth=2)
    axis.set_xscale("log", base=2)
    axis.set_xticks(x)
    axis.set_xticklabels([str(v) for v in x])
    axis.set_xlabel("block size N")
    axis.set_ylabel("fraction of copy floor")
    axis.set_ylim(0, 1.15)
    axis.set_title("Fraction of the DMA ceiling")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="lower right", frameon=False)
    best = max(rows, key=lambda r: r["ratio"])
    axis.annotate(
        f"{best['ratio']:.2f} at N={best['n']}",
        (best["n"], best["ratio"]),
        textcoords="offset points",
        xytext=(0, -18),
        ha="center",
        fontsize=9,
        color=HAD_COLOR,
    )


def _draw_ops(axis, rows):
    """Why the curve has that shape. Separate panel, not a second y-axis."""
    x = [r["n"] for r in rows]
    ops = [_ops_per_element(v) for v in x]
    axis.plot(
        x,
        ops,
        "-",
        marker="s",
        color=OPS_COLOR,
        linewidth=2,
        markersize=8,
        label="vector ops / element",
    )
    axis.set_xscale("log", base=2)
    axis.set_xticks(x)
    axis.set_xticklabels([str(v) for v in x])
    axis.set_xlabel("block size N")
    axis.set_ylabel("vector ops per element")
    axis.set_ylim(0, max(ops) * 1.25)
    axis.set_title("Vector-op cost per element")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="upper right", frameon=False)
    lo = min(range(len(ops)), key=lambda i: ops[i])
    axis.annotate(
        f"cheapest at N={x[lo]}",
        (x[lo], ops[lo]),
        textcoords="offset points",
        xytext=(12, 2),
        ha="center",
        fontsize=9,
        color=OPS_COLOR,
    )


def main():
    args = _parse_args()
    if plt is None:
        print("matplotlib is not installed; skipping plot generation.", file=sys.stderr)
        return
    if not args.csv.exists():
        print(
            f"error: {args.csv} not found (run benchmark.py --nsweep first)",
            file=sys.stderr,
        )
        return
    rows = _load(args.csv)
    if not rows:
        print(f"error: no usable rows in {args.csv}", file=sys.stderr)
        return

    fig, (left, mid, right) = plt.subplots(1, 3, figsize=(18, 5))
    _draw_bandwidth(left, rows)
    _draw_ratio(mid, rows)
    _draw_ops(right, rows)
    fig.suptitle(
        "fast_hadamard_256_a5 on Ascend A5 (dav-c310): block size N vs the copy floor"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path = args.csv.parent / args.plot_name
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
