import argparse
from pathlib import Path

from plot_common import (
    add_common_plot_args,
    plot_comparison_csv,
    plot_csv_collection,
    resolve_dir_arg,
)

# The bench writes <CSV_PREFIX><block_dim>.csv; block_dim_from_path reads the
# integer back off that stem.
CSV_PREFIX = "fht_twosided_compare_bd"

TWOSIDED_COLOR = "#dc2626"
VECTOR_COLOR = "#2563eb"

# Both arms run at every N the bench sweeps, so these plot the whole CSV.
LINE_PLOTS = (
    {
        "filename": "hadamard_twosided_duration_bd{block_dim}.png",
        "series": (
            (
                "twosided_duration_us",
                "Two-sided PTO (Kronecker split)",
                TWOSIDED_COLOR,
                "s-",
            ),
            ("vector_duration_us", "Vector PTO (butterfly)", VECTOR_COLOR, "o-"),
        ),
        "y_label": "Duration (us)",
        "title": "Cube vs Vector Hadamard Duration",
    },
    {
        "filename": "hadamard_twosided_bandwidth_bd{block_dim}.png",
        "series": (
            (
                "twosided_bandwidth_gbs",
                "Two-sided PTO (Kronecker split)",
                TWOSIDED_COLOR,
                "s-",
            ),
            ("vector_bandwidth_gbs", "Vector PTO (butterfly)", VECTOR_COLOR, "o-"),
        ),
        "y_label": "Bandwidth (GB/s)",
        "title": "Cube vs Vector Hadamard Bandwidth",
    },
)

HEATMAPS = (
    {
        "filename": "hadamard_twosided_speedup_vs_vector_bd{block_dim}.png",
        "key": "twosided_speedup_vs_vector",
        "title": "Cube PTO Speedup over Vector PTO",
    },
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot two-sided-vs-vector Hadamard benchmark results from CSV " "files."
        )
    )
    return add_common_plot_args(parser).parse_args()


def plot_hadamard_twosided(csv_path, plot_dir):
    plot_comparison_csv(
        csv_path,
        plot_dir,
        prefix=CSV_PREFIX,
        line_plots=LINE_PLOTS,
        heatmaps=HEATMAPS,
    )


def main():
    args = _parse_args()
    base = Path(__file__).resolve().parent

    csv_dir = resolve_dir_arg(base, args.csv_dir)
    plot_dir = resolve_dir_arg(base, args.plot_dir)

    plot_csv_collection(
        csv_dir,
        plot_dir,
        pattern=f"{CSV_PREFIX}*.csv",
        prefix=CSV_PREFIX,
        warning="no two-sided Hadamard benchmark CSV files found",
        plot_csv_fn=plot_hadamard_twosided,
    )


if __name__ == "__main__":
    main()
