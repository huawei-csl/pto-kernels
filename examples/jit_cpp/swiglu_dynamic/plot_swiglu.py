# pylint: disable=wrong-import-position
import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
FAST_HADAMARD_DIR = THIS_DIR.parent / "fast_hadamard"
if str(FAST_HADAMARD_DIR) not in sys.path:
    sys.path.insert(0, str(FAST_HADAMARD_DIR))

from plot_common import (  # noqa: E402
    add_common_plot_args,
    block_dim_from_path,
    ensure_matplotlib,
    ensure_plot_dir,
    load_nonempty_rows,
    make_batched_line_plot,
    make_speedup_heatmap,
    plot_csv_collection,
    resolve_dir_arg,
)

CSV_PREFIX = "swiglu_compare_bd"
CSV_PATTERN = f"{CSV_PREFIX}*.csv"

DURATION_LINE_PLOT = {
    "filename": "swiglu_duration_bd{block_dim}.png",
    "series": (
        ("pto_duration_us", "PTO SwiGLU", "#dc2626", "s--"),
        ("torch_npu_duration_us", "torch_npu.npu_swiglu", "#2563eb", "o-"),
    ),
    "y_label": "Duration (us)",
    "title": "SwiGLU Duration: PTO vs torch_npu",
}

EFFECTIVE_TOPS_LINE_PLOT = {
    "filename": "swiglu_effective_tops_bd{block_dim}.png",
    "series": (
        ("pto_effective_tops", "PTO SwiGLU", "#dc2626", "s--"),
        (
            "torch_npu_effective_tops",
            "torch_npu.npu_swiglu",
            "#2563eb",
            "o-",
        ),
    ),
    "y_label": "Effective TOPS",
    "title": "SwiGLU Effective Elementwise TOPS: PTO vs torch_npu",
}

LEGACY_TFLOPS_LINE_PLOT = {
    "filename": "swiglu_tflops_bd{block_dim}.png",
    "series": (
        ("pto_tflops", "PTO SwiGLU", "#dc2626", "s--"),
        ("torch_npu_tflops", "torch_npu.npu_swiglu", "#2563eb", "o-"),
    ),
    "y_label": "Legacy effective TFLOPS",
    "title": "SwiGLU Legacy Effective TFLOPS: PTO vs torch_npu",
}

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
    if not ensure_matplotlib():
        return

    rows = load_nonempty_rows(csv_path)
    if rows is None:
        return

    block_dim = block_dim_from_path(csv_path, CSV_PREFIX)
    ensure_plot_dir(plot_dir)

    metric_plot = (
        EFFECTIVE_TOPS_LINE_PLOT
        if "pto_effective_tops" in rows[0]
        else LEGACY_TFLOPS_LINE_PLOT
    )
    for plot in (DURATION_LINE_PLOT, metric_plot):
        make_batched_line_plot(
            rows,
            block_dim,
            plot_dir / plot["filename"].format(block_dim=block_dim),
            plot["series"],
            plot["y_label"],
            plot["title"],
        )

    for heatmap in HEATMAPS:
        make_speedup_heatmap(
            rows,
            block_dim,
            plot_dir / heatmap["filename"].format(block_dim=block_dim),
            heatmap["key"],
            heatmap["title"],
            colorbar_label=heatmap.get("colorbar_label", "log2(speedup)"),
        )

    print(f"Plotted {csv_path.name}")


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
