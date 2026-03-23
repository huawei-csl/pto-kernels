import argparse
from pathlib import Path

from plot_common import (
    add_common_plot_args,
    plot_comparison_csv,
    plot_csv_collection,
    resolve_dir_arg,
)

LINE_PLOTS = (
    {
        "filename": "quantize_duration_bd{block_dim}.png",
        "series": (
            ("pto_duration_us", "PTO int4", "#dc2626", "s--"),
            ("torch_ref_duration_us", "torch packed ref", "#2563eb", "o-"),
        ),
        "y_label": "Duration (us)",
        "title": "Int4 Quantize Duration: PTO vs torch packed ref",
    },
    {
        "filename": "quantize_bandwidth_bd{block_dim}.png",
        "series": (
            ("pto_bandwidth_gbs", "PTO int4", "#dc2626", "s--"),
            ("torch_ref_bandwidth_gbs", "torch packed ref", "#2563eb", "o-"),
        ),
        "y_label": "Effective Bandwidth (GB/s)",
        "title": "Int4 Quantize Effective Bandwidth: PTO vs torch packed ref",
    },
)

HEATMAPS = (
    {
        "filename": "quantize_speedup_heatmap_bd{block_dim}.png",
        "key": "pto_speedup_vs_torch_ref",
        "title": "PTO int4 Speedup over torch packed ref",
        "colorbar_label": "log2(PTO int4 speedup)",
    },
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Plot int4 quantize benchmark comparison from CSV files."
    )
    return add_common_plot_args(parser).parse_args()


def plot_quantize(csv_path: Path, plot_dir: Path):
    plot_comparison_csv(
        csv_path,
        plot_dir,
        prefix="quantize_compare_bd",
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
        pattern="quantize_compare_bd*.csv",
        prefix="quantize_compare_bd",
        warning="no int4 quantize benchmark CSV files found",
        plot_csv_fn=plot_quantize,
    )


if __name__ == "__main__":
    main()
