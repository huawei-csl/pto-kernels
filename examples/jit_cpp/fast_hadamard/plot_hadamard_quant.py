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
        "filename": "hadamard_quant_duration_bd{block_dim}.png",
        "series": (
            ("fused_duration_us", "Fused PTO", "#dc2626", "s-"),
            ("separate_duration_us", "Separate PTO", "#ea580c", "^-"),
            ("torch_unfused_duration_us", "torch + torch_npu unfused", "#2563eb", "o-"),
        ),
        "y_label": "Duration (us)",
        "title": "Hadamard+Quant Duration",
    },
    {
        "filename": "hadamard_quant_bandwidth_bd{block_dim}.png",
        "series": (
            ("fused_effective_bandwidth_gbs", "Fused PTO", "#dc2626", "s-"),
            ("separate_effective_bandwidth_gbs", "Separate PTO", "#ea580c", "^-"),
        ),
        "y_label": "Effective Bandwidth (GB/s)",
        "title": "Hadamard+Quant Effective Bandwidth",
    },
)

HEATMAPS = (
    {
        "filename": "hadamard_quant_speedup_vs_separate_bd{block_dim}.png",
        "key": "fused_speedup_vs_separate",
        "title": "Fused PTO Speedup over Separate PTO",
    },
    {
        "filename": "hadamard_quant_speedup_vs_torch_bd{block_dim}.png",
        "key": "fused_speedup_vs_torch_unfused",
        "title": "Fused PTO Speedup over torch + torch_npu unfused",
    },
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Plot fused Hadamard+quant benchmark comparison from CSV files."
    )
    return add_common_plot_args(parser).parse_args()


def plot_hadamard_quant(csv_path, plot_dir):
    plot_comparison_csv(
        csv_path,
        plot_dir,
        prefix="fht_quant_compare_bd",
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
        pattern="fht_quant_compare_bd*.csv",
        prefix="fht_quant_compare_bd",
        warning="no Hadamard+quant benchmark CSV files found",
        plot_csv_fn=plot_hadamard_quant,
    )


if __name__ == "__main__":
    main()
