import argparse
import csv
from pathlib import Path

from jit_util_hadamard import chmod_output_path

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

BENCH_BATCHES = [1, 5, 8, 10, 16, 20, 32, 40, 64, 128, 256, 512, 1024]

DEFAULT_CSV_DIR = Path("outputs") / "csv"
DEFAULT_PLOT_DIR = Path("outputs") / "plots"
DEFAULT_PLOT_NAME = "bw_vs_shape.png"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Fast Hadamard benchmark bandwidth from CSV files."
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
    parser.add_argument(
        "--plot-name",
        type=str,
        default=DEFAULT_PLOT_NAME,
        help=f"Output plot filename (default: {DEFAULT_PLOT_NAME}).",
    )
    return parser.parse_args()


def plot_bandwidth(input_dir: Path, output_path: Path):
    if plt is None:
        print("Warning: matplotlib is not installed; skipping plot generation.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    chmod_output_path(output_path.parent)
    csv_paths = sorted(
        input_dir.glob("fht_pto_bd*.csv"),
        key=lambda path: int(path.stem.removeprefix("fht_pto_bd")),
    )

    if not csv_paths:
        print(f"Warning: no benchmark CSV files found in {input_dir}.")
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
        block_dim = int(csv_path.stem.removeprefix("fht_pto_bd"))
        data = {}
        with csv_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
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
                last_markers = ["s", "^", "D"]
                marker = last_markers[idx - 10]

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
    fig.savefig(output_path, dpi=150)
    chmod_output_path(output_path)
    print(f"\nPlot saved to {output_path}")


def main():
    args = _parse_args()
    base = Path(__file__).resolve().parent

    csv_dir = Path(args.csv_dir)
    if not csv_dir.is_absolute():
        csv_dir = base / csv_dir

    plot_dir = Path(args.plot_dir)
    if not plot_dir.is_absolute():
        plot_dir = base / plot_dir

    output_path = plot_dir / args.plot_name
    plot_bandwidth(csv_dir, output_path)


if __name__ == "__main__":
    main()
