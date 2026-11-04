import argparse
from pathlib import Path

from plot_common import (
    add_common_plot_args,
    block_dim_from_path,
    ensure_matplotlib,
    ensure_plot_dir,
    load_nonempty_rows,
    make_batched_line_plot,
    make_speedup_heatmap,
    plot_comparison_csv,
    plot_csv_collection,
    resolve_dir_arg,
)

# The bench writes <CSV_PREFIX><block_dim>.csv; block_dim_from_path reads the
# integer back off that stem.
CSV_PREFIX = "fht_onesided_compare_bd"

# Column that is blank on rows the two-sided kernel cannot run (N < 256).
TWO_SIDED_KEY = "twosided_duration_us"

ONESIDED_COLOR = "#16a34a"
VECTOR_COLOR = "#2563eb"
TWOSIDED_COLOR = "#dc2626"

# Every row has a one-sided and a vector arm, so these plot the whole CSV.
LINE_PLOTS = (
    {
        "filename": "hadamard_onesided_duration_bd{block_dim}.png",
        "series": (
            (
                "onesided_duration_us",
                "One-sided PTO (dense matmul)",
                ONESIDED_COLOR,
                "D-",
            ),
            ("vector_duration_us", "Vector PTO (butterfly)", VECTOR_COLOR, "o-"),
        ),
        "y_label": "Duration (us)",
        "title": "One-sided vs Vector Hadamard Duration",
    },
    {
        "filename": "hadamard_onesided_bandwidth_bd{block_dim}.png",
        "series": (
            (
                "onesided_bandwidth_gbs",
                "One-sided PTO (dense matmul)",
                ONESIDED_COLOR,
                "D-",
            ),
            ("vector_bandwidth_gbs", "Vector PTO (butterfly)", VECTOR_COLOR, "o-"),
        ),
        "y_label": "Bandwidth (GB/s)",
        "title": "One-sided vs Vector Hadamard Bandwidth",
    },
)

HEATMAPS = (
    {
        "filename": "hadamard_onesided_speedup_vs_vector_bd{block_dim}.png",
        "key": "onesided_speedup_vs_vector",
        "title": "One-sided PTO Speedup over Vector PTO",
    },
)

# These need the two-sided arm, so they only cover the rows that have one.
TWO_SIDED_LINE_PLOTS = (
    {
        "filename": "hadamard_onesided_vs_twosided_duration_bd{block_dim}.png",
        "series": (
            (
                "onesided_duration_us",
                "One-sided PTO (dense matmul)",
                ONESIDED_COLOR,
                "D-",
            ),
            (
                "twosided_duration_us",
                "Two-sided PTO (Kronecker split)",
                TWOSIDED_COLOR,
                "s-",
            ),
            ("vector_duration_us", "Vector PTO (butterfly)", VECTOR_COLOR, "o-"),
        ),
        "y_label": "Duration (us)",
        "title": "One-sided vs Two-sided vs Vector Hadamard Duration",
    },
)

TWO_SIDED_HEATMAPS = (
    {
        "filename": "hadamard_onesided_speedup_vs_twosided_bd{block_dim}.png",
        "key": "onesided_speedup_vs_twosided",
        "title": "One-sided PTO Speedup over Two-sided PTO",
    },
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot one-sided Hadamard benchmark results (against the vector and "
            "two-sided kernels) from CSV files."
        )
    )
    return add_common_plot_args(parser).parse_args()


def plot_two_sided_subset(csv_path, plot_dir):
    """The two-sided kernel needs N >= 256, so its columns are blank below that.

    Plotting those cells would mean float(""); filter the rows instead, which
    leaves a complete batch x N grid over the sizes it does support.
    """
    rows = load_nonempty_rows(csv_path)
    if rows is None:
        return

    paired = [row for row in rows if row.get(TWO_SIDED_KEY)]
    if not paired:
        print(
            f"Warning: no two-sided arm in {csv_path.name}; skipping the "
            f"one-sided vs two-sided plots."
        )
        return
    if len(paired) != len(rows):
        skipped = sorted({int(row["N"]) for row in rows if not row.get(TWO_SIDED_KEY)})
        print(
            f"Note: no two-sided arm for N={skipped} in {csv_path.name}; those "
            f"rows are left out of the one-sided vs two-sided plots."
        )

    block_dim = block_dim_from_path(csv_path, CSV_PREFIX)
    ensure_plot_dir(plot_dir)

    for plot in TWO_SIDED_LINE_PLOTS:
        make_batched_line_plot(
            paired,
            block_dim,
            plot_dir / plot["filename"].format(block_dim=block_dim),
            plot["series"],
            plot["y_label"],
            plot["title"],
        )

    for heatmap in TWO_SIDED_HEATMAPS:
        make_speedup_heatmap(
            paired,
            block_dim,
            plot_dir / heatmap["filename"].format(block_dim=block_dim),
            heatmap["key"],
            heatmap["title"],
        )


def plot_hadamard_onesided(csv_path, plot_dir):
    # Checked once here so the two passes below cannot print the same warning
    # twice, and so a container without matplotlib skips rather than tracebacks.
    if not ensure_matplotlib():
        return

    plot_comparison_csv(
        csv_path,
        plot_dir,
        prefix=CSV_PREFIX,
        line_plots=LINE_PLOTS,
        heatmaps=HEATMAPS,
    )
    plot_two_sided_subset(csv_path, plot_dir)


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
        warning="no one-sided Hadamard benchmark CSV files found",
        plot_csv_fn=plot_hadamard_onesided,
    )


if __name__ == "__main__":
    main()
