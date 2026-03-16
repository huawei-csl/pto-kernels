import argparse
from pathlib import Path

from bench_common import BENCH_BATCHES
from plot_common import (
    add_common_plot_args,
    block_dim_from_path,
    collect_csv_paths,
    ensure_matplotlib,
    load_nonempty_rows,
    plt,
    resolve_dir_arg,
    save_figure,
)

DEFAULT_PLOT_NAME = "bw_vs_shape.png"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Fast Hadamard benchmark bandwidth from CSV files."
    )
    add_common_plot_args(parser)
    parser.add_argument(
        "--plot-name",
        type=str,
        default=DEFAULT_PLOT_NAME,
        help=f"Output plot filename (default: {DEFAULT_PLOT_NAME}).",
    )
    return parser.parse_args()


def plot_bandwidth(input_dir: Path, output_path: Path):
    if not ensure_matplotlib():
        return

    csv_paths = collect_csv_paths(
        input_dir,
        "fht_pto_bd*.csv",
        "fht_pto_bd",
        "no benchmark CSV files found",
    )
    if not csv_paths:
        return

    fig, axes = plt.subplots(
        1,
        len(csv_paths),
        figsize=(7 * len(csv_paths), 6),
        sharey=True,
    )
    if len(csv_paths) == 1:
        axes = [axes]

    for ax, csv_path in zip(axes, csv_paths):
        block_dim = block_dim_from_path(csv_path, "fht_pto_bd")
        data = {}
        rows = load_nonempty_rows(csv_path)
        if rows is None:
            continue
        for row in rows:
            batch = int(row["batch"])
            n = int(row["N"])
            bw = float(row["bandwidth_gbs"])
            data.setdefault(n, {})[batch] = bw

        for idx, hidden_dim in enumerate(sorted(data.keys())):
            batches = sorted(data[hidden_dim].keys())
            bws = [data[hidden_dim][b] for b in batches]

            if idx < 10:
                marker = "o"
            else:
                marker = ["s", "^", "D"][idx - 10]

            ax.plot(
                batches,
                bws,
                marker=marker,
                markersize=4,
                label=f"hidden_dim={hidden_dim}",
            )

        ax.set_xscale("log", base=2)
        ax.set_xticks(BENCH_BATCHES)
        ax.set_xticklabels([str(b) for b in BENCH_BATCHES], rotation=45, fontsize=7)
        ax.set_xlabel("batch")
        ax.set_title(f"BLOCK_DIM={block_dim}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)

    axes[0].set_ylabel("Bandwidth (GB/s)")
    fig.suptitle("Fast Hadamard PTO-ISA: Bandwidth vs Shape")
    fig.tight_layout()
    save_figure(fig, output_path, bbox_inches=None)
    print(f"\nPlot saved to {output_path}")


def main():
    args = _parse_args()
    base = Path(__file__).resolve().parent

    csv_dir = resolve_dir_arg(base, args.csv_dir)
    plot_dir = resolve_dir_arg(base, args.plot_dir)

    output_path = plot_dir / args.plot_name
    plot_bandwidth(csv_dir, output_path)


if __name__ == "__main__":
    main()
