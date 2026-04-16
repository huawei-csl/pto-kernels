import argparse
import math
from pathlib import Path

from plot_common import (
    add_common_plot_args,
    ensure_matplotlib,
    format_log2_ticks,
    group_by_batch,
    load_nonempty_rows,
    normalize_axes,
    plt,
    resolve_dir_arg,
    save_figure,
    ticker,
)

CSV_NAME = "hadamard_dynamic_quant_int4.csv"

LINE_PLOTS = (
    {
        "filename": "hadamard_dynamic_quant_duration.png",
        "series": (
            ("traffic_us", "Traffic-matched copy", "#64748b", "o-"),
            ("static_us", "Static PTO int4", "#ea580c", "^-"),
            ("dynamic_us", "Dynamic PTO int4", "#dc2626", "s-"),
        ),
        "y_label": "Duration (us)",
        "title": "Hadamard+Dynamic Int4 Quant Duration",
    },
    {
        "filename": "hadamard_dynamic_quant_bandwidth.png",
        "series": (
            ("traffic_bw_gbs", "Traffic-matched copy", "#64748b", "o-"),
            ("static_bw_gbs", "Static PTO int4", "#ea580c", "^-"),
            ("dynamic_bw_gbs", "Dynamic PTO int4", "#dc2626", "s-"),
        ),
        "y_label": "Effective Bandwidth (GB/s)",
        "title": "Hadamard+Dynamic Int4 Quant Effective Bandwidth",
    },
)

HEATMAPS = (
    {
        "filename": "hadamard_dynamic_quant_speedup_vs_static.png",
        "key": "speedup_vs_static",
        "title": "Dynamic PTO int4 Speedup over Static PTO int4",
    },
    {
        "filename": "hadamard_dynamic_quant_speedup_vs_traffic.png",
        "key": "speedup_vs_traffic",
        "title": "Dynamic PTO int4 Speedup over Traffic-matched Copy",
    },
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Plot fused Hadamard+dynamic int4 quant benchmark results."
    )
    return add_common_plot_args(parser).parse_args()


def _make_line_plot(rows, output_path: Path, series, y_label, title):
    batches = sorted({int(row["batch"]) for row in rows})
    ncols = min(5, len(batches))
    nrows = (len(batches) + ncols - 1) // ncols
    grouped = group_by_batch(rows, [key for key, _, _, _ in series])

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.6 * nrows))
    axes = normalize_axes(axes)

    for idx, batch in enumerate(batches):
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

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def _make_heatmap(rows, output_path: Path, key: str, title: str):
    batches = sorted({int(row["batch"]) for row in rows})
    ns = sorted({int(row["N"]) for row in rows})

    speedups = {(int(row["batch"]), int(row["N"])): float(row[key]) for row in rows}
    matrix = [[speedups[(batch, n)] for n in ns] for batch in batches]
    log_matrix = [[math.log2(max(value, 1e-9)) for value in row] for row in matrix]
    vmax = max(max(abs(value) for value in row) for row in log_matrix)
    vmax = max(vmax, 0.5)

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(
        log_matrix,
        cmap="RdBu",
        aspect="auto",
        origin="lower",
        vmin=-vmax,
        vmax=vmax,
    )

    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels(ns, rotation=45)
    ax.set_yticks(range(len(batches)))
    ax.set_yticklabels(batches)
    ax.set_xlabel("N")
    ax.set_ylabel("batch")
    ax.set_title(f"{title} (log2 scale)", fontsize=13, fontweight="bold")

    for i, _batch in enumerate(batches):
        for j, _n in enumerate(ns):
            value = matrix[i][j]
            color = "white" if abs(log_matrix[i][j]) > vmax * 0.6 else "black"
            ax.text(
                j,
                i,
                f"{value:.2f}x",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color=color,
            )

    cbar = fig.colorbar(im, ax=ax, label="log2(speedup)")
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.set_ticklabels(["0.5x", "0.71x", "1x", "1.41x", "2x"])

    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def plot_hadamard_dynamic_quant(csv_path: Path, plot_dir: Path):
    if not ensure_matplotlib():
        return

    rows = load_nonempty_rows(csv_path)
    if rows is None:
        return

    for plot in LINE_PLOTS:
        _make_line_plot(
            rows,
            plot_dir / plot["filename"],
            plot["series"],
            plot["y_label"],
            plot["title"],
        )

    for heatmap in HEATMAPS:
        _make_heatmap(
            rows,
            plot_dir / heatmap["filename"],
            heatmap["key"],
            heatmap["title"],
        )

    print(f"Plotted {csv_path.name}")


def main():
    args = _parse_args()
    base = Path(__file__).resolve().parent

    csv_dir = resolve_dir_arg(base, args.csv_dir)
    plot_dir = resolve_dir_arg(base, args.plot_dir)
    csv_path = csv_dir / CSV_NAME

    if not csv_path.exists():
        print(f"Warning: no dynamic int4 benchmark CSV found at {csv_path}.")
        return

    plot_hadamard_dynamic_quant(csv_path, plot_dir)


if __name__ == "__main__":
    main()
