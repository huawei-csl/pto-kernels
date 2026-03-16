import csv
import math
from pathlib import Path

from jit_util_common import chmod_output_path

try:
    import matplotlib.pyplot as plt
    from matplotlib import ticker
except ImportError:
    plt = None
    ticker = None

DEFAULT_CSV_DIR = Path("outputs") / "csv"
DEFAULT_PLOT_DIR = Path("outputs") / "plots"


def add_common_plot_args(parser):
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
    return parser


def resolve_dir_arg(base: Path, path_arg: str) -> Path:
    path = Path(path_arg)
    if not path.is_absolute():
        path = base / path
    return path


def ensure_matplotlib() -> bool:
    if plt is None:
        print("Warning: matplotlib is not installed; skipping plot generation.")
        return False
    return True


def ensure_plot_dir(plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    chmod_output_path(plot_dir)


def save_figure(fig, output_path: Path, *, dpi: int = 150, bbox_inches="tight") -> None:
    ensure_plot_dir(output_path.parent)
    fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)
    chmod_output_path(output_path)


def load_rows(csv_path: Path):
    with csv_path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_nonempty_rows(csv_path: Path):
    rows = load_rows(csv_path)
    if not rows:
        print(f"Warning: CSV is empty: {csv_path}")
        return None
    return rows


def block_dim_from_path(csv_path: Path, prefix: str) -> int:
    return int(csv_path.stem.removeprefix(prefix))


def collect_csv_paths(csv_dir: Path, pattern: str, prefix: str, warning: str):
    csv_paths = sorted(
        csv_dir.glob(pattern),
        key=lambda path: block_dim_from_path(path, prefix),
    )
    if not csv_paths:
        print(f"Warning: {warning} in {csv_dir}.")
    return csv_paths


def plot_csv_collection(
    csv_dir: Path,
    plot_dir: Path,
    *,
    pattern: str,
    prefix: str,
    warning: str,
    plot_csv_fn,
):
    csv_paths = collect_csv_paths(csv_dir, pattern, prefix, warning)
    if not csv_paths:
        return
    for csv_path in csv_paths:
        plot_csv_fn(csv_path, plot_dir)


def format_log2_ticks(value, _):
    return f"{int(value)}"


def group_by_batch(rows, value_keys):
    grouped = {}
    for row in rows:
        batch = int(row["batch"])
        n = int(row["N"])
        grouped.setdefault(batch, {})[n] = {key: float(row[key]) for key in value_keys}
    return grouped


def normalize_axes(axes):
    if not isinstance(axes, (list, tuple)):
        try:
            return list(axes.flatten())
        except AttributeError:
            return [axes]
    return list(axes)


def make_batched_line_plot(rows, block_dim, output_path: Path, series, y_label, title):
    batches = sorted({int(row["batch"]) for row in rows})
    ncols = min(5, len(batches))
    nrows = math.ceil(len(batches) / ncols)
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

    fig.suptitle(f"{title} (BLOCK_DIM={block_dim})", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def make_speedup_heatmap(
    rows,
    block_dim,
    output_path: Path,
    key: str,
    title: str,
    *,
    colorbar_label: str = "log2(speedup)",
):
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

    cbar = fig.colorbar(im, ax=ax, label=colorbar_label)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.set_ticklabels(["0.5x", "0.71x", "1x", "1.41x", "2x"])

    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def plot_comparison_csv(
    csv_path: Path,
    plot_dir: Path,
    *,
    prefix: str,
    line_plots,
    heatmaps,
):
    if not ensure_matplotlib():
        return

    rows = load_nonempty_rows(csv_path)
    if rows is None:
        return

    block_dim = block_dim_from_path(csv_path, prefix)
    ensure_plot_dir(plot_dir)

    for plot in line_plots:
        make_batched_line_plot(
            rows,
            block_dim,
            plot_dir / plot["filename"].format(block_dim=block_dim),
            plot["series"],
            plot["y_label"],
            plot["title"],
        )

    for heatmap in heatmaps:
        make_speedup_heatmap(
            rows,
            block_dim,
            plot_dir / heatmap["filename"].format(block_dim=block_dim),
            heatmap["key"],
            heatmap["title"],
            colorbar_label=heatmap.get("colorbar_label", "log2(speedup)"),
        )

    print(f"Plotted {csv_path.name}")
