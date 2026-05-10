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

CSV_PREFIX = "layernorm_compare_bd"
CSV_PATTERN = f"{CSV_PREFIX}*.csv"

# The CSV uses "rows" as the batch axis and "N" as the hidden-dim axis.
# plot_common helpers read "batch" from the CSV, so we rewrite "rows" -> "batch"
# in the loaded rows before passing to the common helpers.

DURATION_LINE_PLOT = {
    "filename": "layernorm_duration_bd{block_dim}.png",
    "series": (
        ("pto_duration_us", "PTO LayerNorm", "#dc2626", "s--"),
        ("torch_duration_us", "F.layer_norm (fp16)", "#2563eb", "o-"),
    ),
    "y_label": "Duration (us)",
    "title": "LayerNorm Duration: PTO (fp16) vs F.layer_norm (fp16)",
}

BANDWIDTH_LINE_PLOT = {
    "filename": "layernorm_bandwidth_bd{block_dim}.png",
    "series": (
        ("pto_bandwidth_gbs", "PTO LayerNorm", "#dc2626", "s--"),
        ("torch_bandwidth_gbs", "F.layer_norm (fp16)", "#2563eb", "o-"),
    ),
    "y_label": "Effective Bandwidth (GB/s)",
    "title": "LayerNorm Effective Bandwidth: PTO (fp16) vs F.layer_norm (fp16)",
}

HEATMAPS = (
    {
        "filename": "layernorm_speedup_heatmap_bd{block_dim}.png",
        "key": "pto_speedup_vs_torch",
        "title": "LayerNorm PTO Speedup over F.layer_norm (fp16)",
        "colorbar_label": "log2(PTO speedup)",
    },
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Plot LayerNorm benchmark comparison from CSV files."
    )
    return add_common_plot_args(parser).parse_args()


def _rows_with_batch_alias(rows):
    """plot_common expects a 'batch' column; our CSV uses 'rows'."""
    return [{**row, "batch": row["rows"]} for row in rows]


def plot_layernorm(csv_path: Path, plot_dir: Path):
    if not ensure_matplotlib():
        return

    rows = load_nonempty_rows(csv_path)
    if rows is None:
        return

    rows = _rows_with_batch_alias(rows)
    block_dim = block_dim_from_path(csv_path, CSV_PREFIX)
    ensure_plot_dir(plot_dir)

    for plot in (DURATION_LINE_PLOT, BANDWIDTH_LINE_PLOT):
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
        warning="no LayerNorm benchmark CSV files found",
        plot_csv_fn=plot_layernorm,
    )


if __name__ == "__main__":
    main()
