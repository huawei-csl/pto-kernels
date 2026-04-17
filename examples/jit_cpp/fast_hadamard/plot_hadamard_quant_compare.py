import argparse
from pathlib import Path

from plot_common import (
    block_dim_from_path,
    collect_csv_paths,
    ensure_matplotlib,
    ensure_plot_dir,
    load_nonempty_rows,
    make_batched_line_plot,
    make_speedup_heatmap,
    pair_csv_paths_by_block_dim,
    pair_rows_by_shape,
    resolve_dir_arg,
)

DEFAULT_INT8_CSV_DIR = Path("fuse_int8_quant") / "outputs" / "csv"
DEFAULT_INT4_CSV_DIR = Path("fuse_int4_quant") / "outputs" / "csv"
DEFAULT_PLOT_DIR = Path("outputs") / "plots"

LINE_PLOTS = (
    {
        "filename": "hadamard_quant_int4_int8_duration_bd{block_dim}.png",
        "series": (
            ("int4_fused_duration_us", "Fused PTO int4", "#2563eb", "o-"),
            ("int4_separate_duration_us", "Separate PTO int4", "#0891b2", "d-"),
            ("int8_fused_duration_us", "Fused PTO int8", "#dc2626", "s-"),
            ("int8_separate_duration_us", "Separate PTO int8", "#ea580c", "^-"),
        ),
        "y_label": "Duration (us)",
        "title": "Hadamard+Quant Duration: Int4 vs Int8",
    },
    {
        "filename": "hadamard_quant_int4_int8_bandwidth_bd{block_dim}.png",
        "series": (
            ("int4_fused_bandwidth_gbs", "Fused PTO int4", "#2563eb", "o-"),
            ("int4_separate_bandwidth_gbs", "Separate PTO int4", "#0891b2", "d-"),
            ("int8_fused_bandwidth_gbs", "Fused PTO int8", "#dc2626", "s-"),
            ("int8_separate_bandwidth_gbs", "Separate PTO int8", "#ea580c", "^-"),
        ),
        "y_label": "Effective Bandwidth (GB/s)",
        "title": "Hadamard+Quant Effective Bandwidth: Int4 vs Int8",
    },
)

HEATMAPS = (
    {
        "filename": "hadamard_quant_int4_fused_speedup_vs_int8_bd{block_dim}.png",
        "key": "int4_fused_speedup_vs_int8",
        "title": "Fused PTO Int4 Speedup over Fused PTO Int8",
        "colorbar_label": "log2(int4 fused speedup)",
    },
    {
        "filename": "hadamard_quant_int4_separate_speedup_vs_int8_bd{block_dim}.png",
        "key": "int4_separate_speedup_vs_int8",
        "title": "Separate PTO Int4 Speedup over Separate PTO Int8",
        "colorbar_label": "log2(int4 separate speedup)",
    },
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Plot PTO int8-vs-int4 fused Hadamard+quant benchmark comparisons from CSV files."
    )
    parser.add_argument(
        "--csv-dir-int8",
        type=str,
        default=str(DEFAULT_INT8_CSV_DIR),
        help=f"Input int8 CSV directory (default: {DEFAULT_INT8_CSV_DIR}).",
    )
    parser.add_argument(
        "--csv-dir-int4",
        type=str,
        default=str(DEFAULT_INT4_CSV_DIR),
        help=f"Input int4 CSV directory (default: {DEFAULT_INT4_CSV_DIR}).",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=str(DEFAULT_PLOT_DIR),
        help=f"Output plot directory (default: {DEFAULT_PLOT_DIR}).",
    )
    return parser.parse_args()


def _merged_rows(int8_rows, int4_rows):
    merged = []
    for batch, n, int8_row, int4_row in pair_rows_by_shape(
        int8_rows,
        int4_rows,
        left_label="int8",
        right_label="int4",
    ):
        int8_fused_duration_us = float(int8_row["fused_duration_us"])
        int4_fused_duration_us = float(int4_row["fused_duration_us"])
        int8_separate_duration_us = float(int8_row["separate_duration_us"])
        int4_separate_duration_us = float(int4_row["separate_duration_us"])

        merged.append(
            {
                "batch": str(batch),
                "N": str(n),
                "int8_fused_duration_us": f"{int8_fused_duration_us:.4f}",
                "int4_fused_duration_us": f"{int4_fused_duration_us:.4f}",
                "int8_separate_duration_us": f"{int8_separate_duration_us:.4f}",
                "int4_separate_duration_us": f"{int4_separate_duration_us:.4f}",
                "int8_fused_bandwidth_gbs": (
                    f"{float(int8_row['fused_effective_bandwidth_gbs']):.4f}"
                ),
                "int4_fused_bandwidth_gbs": (
                    f"{float(int4_row['fused_effective_bandwidth_gbs']):.4f}"
                ),
                "int8_separate_bandwidth_gbs": (
                    f"{float(int8_row['separate_effective_bandwidth_gbs']):.4f}"
                ),
                "int4_separate_bandwidth_gbs": (
                    f"{float(int4_row['separate_effective_bandwidth_gbs']):.4f}"
                ),
                "int4_fused_speedup_vs_int8": (
                    f"{(int8_fused_duration_us / int4_fused_duration_us if int4_fused_duration_us > 0 else 0.0):.4f}"
                ),
                "int4_separate_speedup_vs_int8": (
                    f"{(int8_separate_duration_us / int4_separate_duration_us if int4_separate_duration_us > 0 else 0.0):.4f}"
                ),
            }
        )
    return merged


def plot_hadamard_quant_compare(
    int8_csv_path: Path,
    int4_csv_path: Path,
    plot_dir: Path,
):
    if not ensure_matplotlib():
        return

    int8_rows = load_nonempty_rows(int8_csv_path)
    int4_rows = load_nonempty_rows(int4_csv_path)
    if int8_rows is None or int4_rows is None:
        return

    merged_rows = _merged_rows(int8_rows, int4_rows)
    if not merged_rows:
        print(
            f"Warning: no common Hadamard+quant shapes between {int8_csv_path.name} "
            f"and {int4_csv_path.name}."
        )
        return

    block_dim = block_dim_from_path(int8_csv_path, "fht_quant_compare_bd")
    ensure_plot_dir(plot_dir)

    for plot in LINE_PLOTS:
        make_batched_line_plot(
            merged_rows,
            block_dim,
            plot_dir / plot["filename"].format(block_dim=block_dim),
            plot["series"],
            plot["y_label"],
            plot["title"],
        )

    for heatmap in HEATMAPS:
        make_speedup_heatmap(
            merged_rows,
            block_dim,
            plot_dir / heatmap["filename"].format(block_dim=block_dim),
            heatmap["key"],
            heatmap["title"],
            colorbar_label=heatmap["colorbar_label"],
        )

    print(f"Plotted {int8_csv_path.name} vs {int4_csv_path.name}")


def main():
    args = _parse_args()
    base = Path(__file__).resolve().parent

    int8_csv_dir = resolve_dir_arg(base, args.csv_dir_int8)
    int4_csv_dir = resolve_dir_arg(base, args.csv_dir_int4)
    plot_dir = resolve_dir_arg(base, args.plot_dir)

    int8_paths = collect_csv_paths(
        int8_csv_dir,
        pattern="fht_quant_compare_bd*.csv",
        prefix="fht_quant_compare_bd",
        warning="no int8 Hadamard+quant benchmark CSV files found",
    )
    int4_paths = collect_csv_paths(
        int4_csv_dir,
        pattern="fht_quant_compare_bd*.csv",
        prefix="fht_quant_compare_bd",
        warning="no int4 Hadamard+quant benchmark CSV files found",
    )

    for _block_dim, int8_csv_path, int4_csv_path in pair_csv_paths_by_block_dim(
        int8_paths,
        int4_paths,
        prefix="fht_quant_compare_bd",
        left_label="int8",
        right_label="int4",
    ):
        plot_hadamard_quant_compare(int8_csv_path, int4_csv_path, plot_dir)


if __name__ == "__main__":
    main()
