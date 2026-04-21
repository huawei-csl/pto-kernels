# pylint: disable=wrong-import-position
"""
Plot Sinkhorn benchmark comparison from CSV files.

Reads:  outputs/csv/sinkhorn_compare_bd*.csv
Writes: outputs/plots/sinkhorn_*.png
"""
import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
FAST_HADAMARD_DIR = THIS_DIR.parent / "fast_hadamard"
if str(FAST_HADAMARD_DIR) not in sys.path:
    sys.path.insert(0, str(FAST_HADAMARD_DIR))

from plot_common import (  # noqa: E402
    add_common_plot_args,
    block_dim_from_path,
    ensure_matplotlib,
    ensure_plot_dir,
    load_nonempty_rows,
    make_batched_line_plot,
    make_speedup_heatmap,
    plot_csv_collection,
    resolve_dir_arg,
)

CSV_PREFIX = "sinkhorn_compare_bd"
CSV_PATTERN = f"{CSV_PREFIX}*.csv"

DURATION_LINE_PLOT = {
    "filename": "sinkhorn_duration_bd{block_dim}.png",
    "series": (
        ("pto_duration_us", "PTO Sinkhorn", "#dc2626", "s--"),
        ("torch_duration_us", "PyTorch Reference", "#2563eb", "o-"),
    ),
    "y_label": "Duration (us)",
    "title": "Sinkhorn Duration: PTO vs PyTorch Reference",
}

BANDWIDTH_LINE_PLOT = {
    "filename": "sinkhorn_bandwidth_bd{block_dim}.png",
    "series": (
        ("pto_bandwidth_gbs", "PTO Sinkhorn", "#dc2626", "s--"),
        ("torch_bandwidth_gbs", "PyTorch Reference", "#2563eb", "o-"),
    ),
    "y_label": "Effective Bandwidth (GB/s)",
    "title": "Sinkhorn Effective Bandwidth: PTO vs PyTorch Reference",
}

HEATMAPS = (
    {
        "filename": "sinkhorn_speedup_heatmap_bd{block_dim}.png",
        "key": "pto_speedup_vs_torch",
        "title": "Sinkhorn PTO Speedup over PyTorch Reference",
        "colorbar_label": "log2(PTO speedup)",
    },
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Sinkhorn benchmark comparison from CSV files."
    )
    return add_common_plot_args(parser).parse_args()


def _make_2x3_line_plot(
    rows, block_dim, output_path, series, y_label, title, log_y=False
):
    """2x3 subplot grid with optional log y-scale."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import ticker
    from plot_common import (
        group_by_batch,
        normalize_axes,
        save_figure,
        format_log2_ticks,
    )

    batches = sorted({int(row["batch"]) for row in rows})
    grouped = group_by_batch(rows, [key for key, _, _, _ in series])

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))
    axes = normalize_axes(axes)

    for idx, batch in enumerate(batches):
        if idx >= len(axes):
            break
        ax = axes[idx]
        ns = sorted(grouped[batch].keys())
        for key, label, color, style in series:
            ax.plot(
                ns,
                [grouped[batch][n][key] for n in ns],
                style,
                color=color,
                label=label,
                linewidth=2,
                markersize=5,
            )
        ax.set_xscale("log", base=2)
        if log_y:
            ax.set_yscale("log")
        ax.set_title(f"batch = {batch}", fontsize=11, fontweight="bold")
        ax.set_xlabel("N")
        ax.set_ylabel(y_label)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_log2_ticks))
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)

    for idx in range(len(batches), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"{title} (BLOCK_DIM={block_dim})", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def plot_sinkhorn(csv_path: Path, plot_dir: Path):
    if not ensure_matplotlib():
        return

    rows = load_nonempty_rows(csv_path)
    if rows is None:
        return

    block_dim = block_dim_from_path(csv_path, CSV_PREFIX)
    ensure_plot_dir(plot_dir)

    # Duration: log y-scale, 2x3 layout
    _make_2x3_line_plot(
        rows,
        block_dim,
        plot_dir / DURATION_LINE_PLOT["filename"].format(block_dim=block_dim),
        DURATION_LINE_PLOT["series"],
        DURATION_LINE_PLOT["y_label"],
        DURATION_LINE_PLOT["title"],
        log_y=True,
    )

    # Bandwidth: linear y-scale, 2x3 layout
    _make_2x3_line_plot(
        rows,
        block_dim,
        plot_dir / BANDWIDTH_LINE_PLOT["filename"].format(block_dim=block_dim),
        BANDWIDTH_LINE_PLOT["series"],
        BANDWIDTH_LINE_PLOT["y_label"],
        BANDWIDTH_LINE_PLOT["title"],
        log_y=False,
    )

    for heatmap in HEATMAPS:
        make_speedup_heatmap(
            rows,
            block_dim,
            plot_dir / heatmap["filename"].format(block_dim=block_dim),
            heatmap["key"],
            heatmap["title"],
            colorbar_label=heatmap.get("colorbar_label", "log2(speedup)"),
        )

    print(f"Plotted {csv_path.name}")


def main():
    args = _parse_args()
    base = THIS_DIR
    csv_dir = resolve_dir_arg(base, args.csv_dir)
    plot_dir = resolve_dir_arg(base, args.plot_dir)

    plot_csv_collection(
        csv_dir,
        plot_dir,
        pattern=CSV_PATTERN,
        prefix=CSV_PREFIX,
        warning="no Sinkhorn benchmark CSV files found",
        plot_csv_fn=plot_sinkhorn,
    )


if __name__ == "__main__":
    main()
