import argparse
import csv
import math
from pathlib import Path

from jit_util_hadamard import chmod_output_path

try:
    import matplotlib.pyplot as plt
    from matplotlib import ticker
except ImportError:
    plt = None
    ticker = None

DEFAULT_CSV_DIR = Path("outputs") / "csv"
DEFAULT_PLOT_DIR = Path("outputs") / "plots"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Plot fused Hadamard+quant benchmark comparison from CSV files."
    )
    parser.add_argument(
        "--csv-dir",
        type=str,
        default=str(DEFAULT_CSV_DIR),
        help=f"Input CSV directory (default: {DEFAULT_CSV_DIR}).",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=str(DEFAULT_PLOT_DIR),
        help=f"Output plot directory (default: {DEFAULT_PLOT_DIR}).",
    )
    return parser.parse_args()


def _load_rows(csv_path: Path):
    with csv_path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _block_dim_from_path(csv_path: Path):
    return int(csv_path.stem.removeprefix("fht_quant_compare_bd"))


def _format_log2_ticks(value, _):
    return f"{int(value)}"


def _group_by_batch(rows, value_keys):
    grouped = {}
    for row in rows:
        batch = int(row["batch"])
        n = int(row["N"])
        grouped.setdefault(batch, {})[n] = {key: float(row[key]) for key in value_keys}
    return grouped


def _make_line_plot(rows, block_dim, output_path: Path, series, y_label, title):
    batches = sorted({int(row["batch"]) for row in rows})
    ncols = min(5, len(batches))
    nrows = math.ceil(len(batches) / ncols)
    grouped = _group_by_batch(rows, [key for key, _, _, _ in series])

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.6 * nrows))
    if not isinstance(axes, (list, tuple)):
        try:
            axes = axes.flatten()
        except AttributeError:
            axes = [axes]
    else:
        axes = list(axes)

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
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_format_log2_ticks))
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)

    for idx in range(len(batches), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"{title} (BLOCK_DIM={block_dim})", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    chmod_output_path(output_path)
    plt.close(fig)


def _make_speedup_heatmap(rows, block_dim, output_path: Path, key, title):
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
    ax.set_title(
        f"{title} (BLOCK_DIM={block_dim}, log2 scale)",
        fontsize=13,
        fontweight="bold",
    )

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
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    chmod_output_path(output_path)
    plt.close(fig)


def plot_hadamard_quant(csv_path: Path, plot_dir: Path):
    if plt is None:
        print("Warning: matplotlib is not installed; skipping plot generation.")
        return

    rows = _load_rows(csv_path)
    if not rows:
        print(f"Warning: CSV is empty: {csv_path}")
        return

    block_dim = _block_dim_from_path(csv_path)
    plot_dir.mkdir(parents=True, exist_ok=True)
    chmod_output_path(plot_dir)

    _make_line_plot(
        rows,
        block_dim,
        plot_dir / f"hadamard_quant_duration_bd{block_dim}.png",
        (
            ("fused_duration_us", "Fused PTO", "#dc2626", "s-"),
            ("separate_duration_us", "Separate PTO", "#ea580c", "^-"),
            ("torch_unfused_duration_us", "torch + torch_npu unfused", "#2563eb", "o-"),
        ),
        "Duration (us)",
        "Hadamard+Quant Duration",
    )
    _make_line_plot(
        rows,
        block_dim,
        plot_dir / f"hadamard_quant_bandwidth_bd{block_dim}.png",
        (
            ("fused_bandwidth_gbs", "Fused PTO", "#dc2626", "s-"),
            ("separate_bandwidth_gbs", "Separate PTO", "#ea580c", "^-"),
        ),
        "Bandwidth (GB/s)",
        "Hadamard+Quant Effective Bandwidth",
    )
    _make_speedup_heatmap(
        rows,
        block_dim,
        plot_dir / f"hadamard_quant_speedup_vs_separate_bd{block_dim}.png",
        "fused_speedup_vs_separate",
        "Fused PTO Speedup over Separate PTO",
    )
    _make_speedup_heatmap(
        rows,
        block_dim,
        plot_dir / f"hadamard_quant_speedup_vs_torch_bd{block_dim}.png",
        "fused_speedup_vs_torch_unfused",
        "Fused PTO Speedup over torch + torch_npu unfused",
    )

    print(f"Plotted {csv_path.name}")


def main():
    args = _parse_args()
    base = Path(__file__).resolve().parent

    csv_dir = Path(args.csv_dir)
    if not csv_dir.is_absolute():
        csv_dir = base / csv_dir

    plot_dir = Path(args.plot_dir)
    if not plot_dir.is_absolute():
        plot_dir = base / plot_dir

    csv_paths = sorted(
        csv_dir.glob("fht_quant_compare_bd*.csv"),
        key=_block_dim_from_path,
    )
    if not csv_paths:
        print(f"Warning: no Hadamard+quant benchmark CSV files found in {csv_dir}.")
        return

    for csv_path in csv_paths:
        plot_hadamard_quant(csv_path, plot_dir)


if __name__ == "__main__":
    main()
