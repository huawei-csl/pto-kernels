import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_CSV_REL_PATH = Path("outputs") / "csv" / "matmul_timing.csv"
DEFAULT_PLOT_REL_DIR = Path("outputs") / "plots"
DEFAULT_PLOT_N = 16384
DEFAULT_PLOT_K = 16384

BACKEND_STYLE = {
    "torch": {"color": "#111111", "marker": "x", "linestyle": "--"},
    "original": {"color": "#ff7f0e", "marker": "s", "linestyle": "-."},
    "custom": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
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
        "--n",
        type=int,
        default=DEFAULT_PLOT_N,
        help=f"N value to plot (default: {DEFAULT_PLOT_N})",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_PLOT_K,
        help=f"K value to plot (default: {DEFAULT_PLOT_K})",
    )
    parser.add_argument(
        "--swizzle",
        action="append",
        default=[],
        help=(
            "Filter plotted custom swizzles using 'direction,count' (or 'direction:count'). "
            "Repeat this argument to keep multiple swizzles."
        ),
    )
    return parser.parse_args()


def _parse_swizzle_pair(raw: str) -> tuple[int, int]:
    text = raw.strip().replace(":", ",")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(
            f"Invalid --swizzle value '{raw}'. Expected 'direction,count' (for example: 0,5)."
        )

    direction = int(parts[0])
    count = int(parts[1])
    return direction, count


def _build_swizzle_filter(raw_values: list[str]) -> list[tuple[int, int]]:
    if not raw_values:
        return []

    out = []
    seen = set()
    for raw in raw_values:
        cfg = _parse_swizzle_pair(raw)
        if cfg in seen:
            continue
        out.append(cfg)
        seen.add(cfg)
    return out


def _style(name: str) -> dict:
    return BACKEND_STYLE.get(
        name, {"color": "#2ca02c", "marker": "^", "linestyle": "-"}
    )


def _custom_style(index: int) -> dict:
    cmap = plt.get_cmap("tab20")
    return {
        "color": cmap(index % 20),
        "marker": CUSTOM_MARKERS[index % len(CUSTOM_MARKERS)],
        "linestyle": "-",
    }


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


def _plot_baseline(df: pd.DataFrame, metric_col: str, label: str, style: dict):
    if metric_col not in df.columns:
        return
    g = df[["M", metric_col]].dropna()
    if g.empty:
        return
    g = g.groupby("M", as_index=False)[metric_col].mean().sort_values("M")
    plt.plot(
        g["M"],
        g[metric_col],
        marker=style["marker"],
        linestyle=style["linestyle"],
        color=style["color"],
        label=label,
    )


def _iter_custom_metric_groups(df: pd.DataFrame, metric_col: str):
    if metric_col not in df.columns:
        return

    has_swizzle_cols = {
        "custom_swizzle_direction",
        "custom_swizzle_count",
    }.issubset(df.columns)

    if has_swizzle_cols:
        grouped = df.dropna(subset=[metric_col]).groupby(
            ["custom_swizzle_direction", "custom_swizzle_count"], sort=False
        )
        for idx, ((direction, count), group) in enumerate(grouped):
            g = group[["M", metric_col]].dropna()
            if g.empty:
                continue
            g = g.groupby("M", as_index=False)[metric_col].mean().sort_values("M")
            label = f"custom(d={int(direction)}, c={int(count)})"
            yield label, g, _custom_style(idx)
    else:
        g = df[["M", metric_col]].dropna()
        if g.empty:
            return
        g = g.groupby("M", as_index=False)[metric_col].mean().sort_values("M")
        yield "custom", g, _style("custom")


def plot_runtime(df: pd.DataFrame, out_dir: Path, n: int, k: int) -> Path:
    plt.figure(figsize=(10, 5))
    _plot_baseline(df, "torch_time_us", "torch", _style("torch"))
    _plot_baseline(df, "original_time_us", "original", _style("original"))

    for label, group, style in _iter_custom_metric_groups(df, "custom_time_us"):
        plt.plot(
            group["M"],
            group["custom_time_us"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            color=style["color"],
            label=label,
        )

    _finalize_plot(title=f"Runtime vs M: k={k}, n={n}", xlabel="M", ylabel="Runtime (us)")

    out_path = out_dir / "duration.png"
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def plot_tflops(df: pd.DataFrame, out_dir: Path, n: int, k: int) -> Path:
    plt.figure(figsize=(10, 5))
    _plot_baseline(df, "torch_tflops", "torch", _style("torch"))
    _plot_baseline(df, "original_tflops", "original", _style("original"))

    for label, group, style in _iter_custom_metric_groups(df, "custom_tflops"):
        plt.plot(
            group["M"],
            group["custom_tflops"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            color=style["color"],
            label=label,
        )

    _finalize_plot(title=f"TFLOPS vs M: k={k}, n={n}", xlabel="M", ylabel="TFLOPS")

    out_path = out_dir / "flops.png"
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def plot_error(df: pd.DataFrame, out_dir: Path, n: int, k: int) -> Path:
    plt.figure(figsize=(10, 5))
    _plot_baseline(df, "original_mean_diff", "original", _style("original"))

    for label, group, style in _iter_custom_metric_groups(df, "custom_mean_diff"):
        plt.plot(
            group["M"],
            group["custom_mean_diff"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            color=style["color"],
            label=label,
        )

    _finalize_plot(title=f"Error vs M: k={k}, n={n}", xlabel="M", ylabel="Mean Abs Error")

    out_path = out_dir / "error.png"
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def main():
    args = _parse_args()
    swizzle_filter = _build_swizzle_filter(args.swizzle)

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
    required_columns = {"M", "N", "K", "torch_time_us", "torch_tflops"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {sorted(missing)} (file: {csv_path})"
        )

    plot_df = df[(df["N"] == args.n) & (df["K"] == args.k)]
    if plot_df.empty:
        available_shapes = (
            df[["N", "K"]].drop_duplicates().sort_values(["N", "K"]).to_dict("records")
        )
        raise RuntimeError(
            f"No rows found for N={args.n}, K={args.k} in {csv_path}. "
            f"Available shapes: {available_shapes}"
        )

    if swizzle_filter:
        required_swizzle_cols = {"custom_swizzle_direction", "custom_swizzle_count"}
        if not required_swizzle_cols.issubset(plot_df.columns):
            raise RuntimeError(
                "--swizzle filter was provided, but the CSV has no swizzle columns."
            )

        mask = pd.Series(False, index=plot_df.index)
        for direction, count in swizzle_filter:
            mask |= (plot_df["custom_swizzle_direction"] == direction) & (
                plot_df["custom_swizzle_count"] == count
            )
        plot_df = plot_df[mask]

        if plot_df.empty:
            available_swizzles = (
                df[["custom_swizzle_direction", "custom_swizzle_count"]]
                .dropna()
                .drop_duplicates()
                .sort_values(["custom_swizzle_direction", "custom_swizzle_count"])
                .to_dict("records")
            )
            raise RuntimeError(
                "No rows found for requested swizzle filter. "
                f"Available swizzles in CSV: {available_swizzles}"
            )

    runtime_path = plot_runtime(plot_df, plot_dir, args.n, args.k)
    flops_path = plot_tflops(plot_df, plot_dir, args.n, args.k)
    error_path = plot_error(plot_df, plot_dir, args.n, args.k)

    print(f"Loaded CSV: {csv_path}")
    if swizzle_filter:
        print(
            "Applied swizzle filter: "
            + ", ".join(f"(direction={d}, count={c})" for d, c in swizzle_filter)
        )
    print(f"Saved runtime plot (N={args.n}, K={args.k}): {runtime_path}")
    print(f"Saved flops plot (N={args.n}, K={args.k}):   {flops_path}")
    print(f"Saved error plot (N={args.n}, K={args.k}):   {error_path}")


if __name__ == "__main__":
    main()
