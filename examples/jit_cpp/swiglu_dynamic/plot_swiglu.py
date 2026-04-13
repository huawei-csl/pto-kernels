import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
FAST_HADAMARD_DIR = THIS_DIR.parent / "fast_hadamard"
if str(FAST_HADAMARD_DIR) not in sys.path:
    sys.path.insert(0, str(FAST_HADAMARD_DIR))

from plot_common import (
    add_common_plot_args,
    plot_comparison_csv,
    plot_csv_collection,
    resolve_dir_arg,
)

CSV_PREFIX = "swiglu_compare_bd"
CSV_PATTERN = f"{CSV_PREFIX}*.csv"

LINE_PLOTS = (
    {
        "filename": "swiglu_duration_bd{block_dim}.png",
        "series": (
            ("pto_duration_us", "PTO SwiGLU", "#dc2626", "s--"),
            ("torch_npu_duration_us", "torch_npu.npu_swiglu", "#2563eb", "o-"),
        ),
        "y_label": "Duration (us)",
        "title": "SwiGLU Duration: PTO vs torch_npu",
    },
    {
        "filename": "swiglu_tflops_bd{block_dim}.png",
        "series": (
            ("pto_tflops", "PTO SwiGLU", "#dc2626", "s--"),
            ("torch_npu_tflops", "torch_npu.npu_swiglu", "#2563eb", "o-"),
        ),
        "y_label": "Effective TFLOPS",
        "title": "SwiGLU Effective TFLOPS: PTO vs torch_npu",
    },
)

HEATMAPS = (
    {
        "filename": "swiglu_speedup_heatmap_bd{block_dim}.png",
        "key": "pto_speedup_vs_torch_npu",
        "title": "SwiGLU PTO Speedup over torch_npu",
        "colorbar_label": "log2(PTO speedup)",
    },
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Plot SwiGLU benchmark comparison from CSV files."
    )
    return add_common_plot_args(parser).parse_args()


def plot_swiglu(csv_path: Path, plot_dir: Path):
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
        pattern=CSV_PATTERN,
        prefix=CSV_PREFIX,
        warning="no SwiGLU benchmark CSV files found",
        plot_csv_fn=plot_swiglu,
    )


if __name__ == "__main__":
    main()
