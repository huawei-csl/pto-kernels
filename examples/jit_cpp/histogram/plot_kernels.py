import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_CSV_REL_PATH = Path("outputs") / "csv" / "histogram_timing.csv"
DEFAULT_PLOT_REL_DIR = Path("outputs") / "plots"

BACKEND_STYLE = {
    "torch": {"color": "#111111", "marker": "x", "linestyle": "--"},
    "step1": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "step2": {"color": "#ff7f0e", "marker": "s", "linestyle": "-"},
}
CUSTOM_MARKERS = ["o", "s", "^", "v", "D", "P", "X", "*", "<", ">"]


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Plot benchmark figures from a benchmark CSV file."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(DEFAULT_CSV_REL_PATH),
        help=f"Input benchmark CSV path (default: {DEFAULT_CSV_REL_PATH})",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=str(DEFAULT_PLOT_REL_DIR),
        help=f"Output plot directory (default: {DEFAULT_PLOT_REL_DIR})",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=256,
        help="Number of bins to plot (default: 256)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        action="append",
        default=[],
        help="Tile sizes to plot. Can be repeated. If not provided, defaults to 4096.",
    )
    return parser.parse_args()


def _style(name: str) -> dict:
    return BACKEND_STYLE.get(
        name, {"color": "#2ca02c", "marker": "^", "linestyle": "-"}
    )


def _finalize_plot(title: str, xlabel: str, ylabel: str):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.grid(True, alpha=0.25)
    handles, _ = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(fontsize=8)
    plt.tight_layout()


def _plot_backend(df: pd.DataFrame, backend: str, metric_col: str, style: dict):
    if metric_col not in df.columns:
        return

    if backend == "torch" or "tile_size" not in df.columns:
        g = df[["N", metric_col]].dropna()
        if g.empty:
            return
        g = g.groupby("N", as_index=False)[metric_col].mean().sort_values("N")
        plt.plot(
            g["N"],
            g[metric_col],
            marker=style["marker"],
            linestyle=style["linestyle"],
            color=style["color"],
            label=backend,
        )
    else:
        ts_values = sorted(df["tile_size"].dropna().unique())
        grouped = df.dropna(subset=[metric_col]).groupby("tile_size", sort=True)
        for idx, (ts, group) in enumerate(grouped):
            g = group[["N", metric_col]].dropna()
            if g.empty:
                continue
            g = g.groupby("N", as_index=False)[metric_col].mean().sort_values("N")

            if len(ts_values) > 1:
                label = f"{backend} (ts={ts})"
                marker = CUSTOM_MARKERS[idx % len(CUSTOM_MARKERS)]
                alpha = max(0.4, 1.0 - (idx * 0.15))
            else:
                label = backend
                marker = style["marker"]
                alpha = 1.0

            plt.plot(
                g["N"],
                g[metric_col],
                marker=marker,
                linestyle=style["linestyle"],
                color=style["color"],
                alpha=alpha,
                label=label,
            )


def plot_runtime(df: pd.DataFrame, out_dir: Path, bins: int) -> Path:
    plt.figure(figsize=(10, 5))
    for backend in ["torch", "step1", "step2"]:
        _plot_backend(df, backend, f"{backend}_time_us", _style(backend))

    _finalize_plot(
        title=f"Runtime vs N (bins={bins})",
        xlabel="Number of Elements (N)",
        ylabel="Runtime (us)",
    )

    out_path = out_dir / "duration.png"
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def plot_throughput(df: pd.DataFrame, out_dir: Path, bins: int) -> Path:
    plt.figure(figsize=(10, 5))
    for backend in ["torch", "step1", "step2"]:
        _plot_backend(df, backend, f"{backend}_gmelem_s", _style(backend))

    _finalize_plot(
        title=f"Throughput vs N (bins={bins})",
        xlabel="Number of Elements (N)",
        ylabel="GElem/s",
    )

    out_path = out_dir / "throughput.png"
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def plot_error(df: pd.DataFrame, out_dir: Path, bins: int) -> Path:
    plt.figure(figsize=(10, 5))
    for backend in ["step1", "step2"]:
        _plot_backend(df, backend, f"{backend}_mean_diff", _style(backend))

    _finalize_plot(
        title=f"Error vs N (bins={bins})",
        xlabel="Number of Elements (N)",
        ylabel="Mean Abs Error",
    )

    out_path = out_dir / "error.png"
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def main():
    args = _parse_args()
    base = Path(__file__).resolve().parent

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = base / csv_path
    if not csv_path.exists():
        raise FileNotFoundError(f"Benchmark CSV not found: {csv_path}")

    plot_dir = Path(args.plot_dir)
    if not plot_dir.is_absolute():
        plot_dir = base / plot_dir
    plot_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    required_columns = {"N"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {sorted(missing)} (file: {csv_path})"
        )

    plot_df = df

    if "bins" in plot_df.columns:
        plot_df = plot_df[plot_df["bins"] == args.bins]
        if plot_df.empty:
            available_bins = sorted(df["bins"].dropna().unique())
            raise RuntimeError(
                f"No rows found for bins={args.bins} in {csv_path}. "
                f"Available bins: {available_bins}"
            )

    tile_sizes = args.tile_size if args.tile_size else [4096]
    if "tile_size" in plot_df.columns:
        plot_df = plot_df[plot_df["tile_size"].isin(tile_sizes)]
        if plot_df.empty:
            available_ts = sorted(df["tile_size"].dropna().unique())
            raise RuntimeError(
                f"No rows found for tile_sizes={tile_sizes} in {csv_path} (with bins={args.bins}). "
                f"Available tile sizes: {available_ts}"
            )

    if plot_df.empty:
        raise RuntimeError(f"No data found in {csv_path}.")

    runtime_path = plot_runtime(plot_df, plot_dir, args.bins)
    throughput_path = plot_throughput(plot_df, plot_dir, args.bins)
    error_path = plot_error(plot_df, plot_dir, args.bins)

    print(f"Loaded CSV: {csv_path}")
    print(
        f"Filters applied: bins={args.bins}, tile_sizes={tile_sizes if 'tile_size' in df.columns else 'N/A'}"
    )
    print(f"Saved runtime plot:    {runtime_path}")
    print(f"Saved throughput plot: {throughput_path}")
    print(f"Saved error plot:      {error_path}")


if __name__ == "__main__":
    main()
