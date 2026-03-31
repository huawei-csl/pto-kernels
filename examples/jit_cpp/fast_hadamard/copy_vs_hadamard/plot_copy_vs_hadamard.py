import argparse
import math
from pathlib import Path

from plot_common import (
    add_common_plot_args,
    collect_csv_paths,
    ensure_matplotlib,
    load_nonempty_rows,
    resolve_dir_arg,
    save_figure,
)

DEFAULT_PLOT_NAME_TEMPLATE = "copy_vs_hadamard_duration_us_bd{block_dim}.png"
SERIES = [
    ("copy_pto_us", "PTO copy", "#1f77b4"),
    ("copy_raw_cce_us", "Raw CCE copy", "#ff7f0e"),
    ("copy_raw_cce_static_4096_us", "Raw CCE copy static 4096x4096", "#2ca02c"),
    ("torch_clone_us", "torch.clone", "#7f7f7f"),
    ("hadamard_us", "fast_hadamard", "#d62728"),
]


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Plot grouped duration bars for copy-vs-Hadamard benchmark CSV files."
    )
    parser = add_common_plot_args(parser)
    parser.add_argument(
        "--plot-name",
        type=str,
        default=None,
        help="Optional output filename override.",
    )
    return parser.parse_args()


def _parse_optional_float(value):
    if value is None or value == "":
        return None
    return float(value)


def _group_rows(rows):
    grouped = {}
    for row in rows:
        batch = int(row["batch"])
        n = int(row["N"])
        grouped.setdefault(batch, {})[n] = {
            key: _parse_optional_float(row[key]) for key, _, _ in SERIES
        }
    return grouped


def plot_copy_vs_hadamard(csv_path: Path, output_path: Path):
    if not ensure_matplotlib():
        return

    rows = load_nonempty_rows(csv_path)
    if rows is None:
        return

    import matplotlib.pyplot as plt

    grouped = _group_rows(rows)
    batches = sorted(grouped)
    ns = sorted({int(row["N"]) for row in rows})
    ncols = 3
    nrows = math.ceil(len(batches) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(8.6 * ncols, 4.4 * nrows),
        squeeze=False,
    )
    axes = axes.flatten()

    width = 0.15
    x_positions = list(range(len(ns)))
    offset_center = (len(SERIES) - 1) / 2.0

    for axis_idx, batch in enumerate(batches):
        ax = axes[axis_idx]
        values_by_n = grouped[batch]

        for series_idx, (key, label, color) in enumerate(SERIES):
            heights = [
                (
                    math.nan
                    if values_by_n.get(n, {}).get(key) is None
                    else values_by_n.get(n, {}).get(key)
                )
                for n in ns
            ]
            positions = [
                pos + width * (series_idx - offset_center) for pos in x_positions
            ]
            ax.bar(
                positions,
                heights,
                width=width,
                color=color,
                label=label,
            )

        ax.set_title(f"batch = {batch}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Hidden dim")
        ax.set_ylabel("Duration (us)")
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(n) for n in ns], rotation=45, ha="right")
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", alpha=0.3)
        if axis_idx == 0:
            ax.legend(fontsize=8)

    for axis_idx in range(len(batches), len(axes)):
        axes[axis_idx].set_visible(False)

    fig.suptitle(
        "Copy vs fast_hadamard duration by batch and hidden dim",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)
    print(f"Plotted {csv_path.name}")


def main():
    args = _parse_args()
    base = Path(__file__).resolve().parent
    csv_dir = resolve_dir_arg(base, args.csv_dir)
    plot_dir = resolve_dir_arg(base, args.plot_dir)

    csv_paths = collect_csv_paths(
        csv_dir,
        "copy_vs_hadamard_bd*.csv",
        "copy_vs_hadamard_bd",
        "no copy-vs-Hadamard benchmark CSV files found",
    )
    if not csv_paths:
        return

    for csv_path in csv_paths:
        block_dim = int(csv_path.stem.removeprefix("copy_vs_hadamard_bd"))
        plot_name = (
            args.plot_name
            if args.plot_name is not None
            else DEFAULT_PLOT_NAME_TEMPLATE.format(block_dim=block_dim)
        )
        plot_copy_vs_hadamard(csv_path, plot_dir / plot_name)


if __name__ == "__main__":
    main()
