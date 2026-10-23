"""Plot a GDN speedup heatmap from gdn_mega_sweep.py JSON; no NPU needed."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
import numpy as np  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, help="Filename stem; writes PNG and SVG beside input by default")
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    heads, lengths = sorted(data["heads"]), sorted(data["lengths"])
    cells = {(r["heads"], r["length"]): r for r in data["records"]}
    if len(cells) != len(data["records"]) or set(cells) != {(h, n) for h in heads for n in lengths}:
        raise ValueError("Heatmap requires one checked result for every requested cell")
    for row in cells.values():
        if row.get("status") == "failed":
            if row["speedup"] is not None or not any(row[v].get("status") == "failed" for v in ["baseline", "current"]):
                raise ValueError("Failed comparisons must identify a failed implementation and have no speedup")
        elif row["speedup"] is None or not np.isfinite(row["speedup"]) or row["speedup"] <= 0:
            raise ValueError("Passing speedups must be finite and positive")
    values = np.array([[np.nan if cells[h, n].get("status") == "failed" else cells[h, n]["speedup"]
                        for n in lengths] for h in heads], dtype=float)
    valid = values[~np.isnan(values)]
    if not valid.size:
        raise ValueError("No passing comparisons to plot")
    # A fixed 1x neutral point makes regressions visibly different from wins.
    norm = TwoSlopeNorm(vmin=min(0.5, float(valid.min())), vcenter=1.0,
                        vmax=max(1.5, float(valid.max())))
    fig, ax = plt.subplots(figsize=(11, 7), layout="constrained")
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#d9dde3")
    mesh = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(lengths)), [f"{n // 1024}K" if n >= 1024 and n % 1024 == 0 else str(n) for n in lengths])
    ax.set_yticks(range(len(heads)), [f"{h} / {h // data['group_size']}" for h in heads])
    ax.set_xlabel("Sequence length (tokens)")
    ax.set_ylabel("Value heads H / key heads Hg")
    ax.set_title("GDN mega kernel: original / optimized latency", fontsize=17, pad=16)
    for i, h in enumerate(heads):
        for j, n in enumerate(lengths):
            value = values[i, j]
            if np.isnan(value):
                failed = [name for name in ["baseline", "current"] if cells[h, n][name].get("status") == "failed"]
                ax.text(j, i, "/".join(failed) + "\nFAIL", ha="center", va="center", fontsize=10, color="#374151")
                continue
            color = "white" if norm(value) > 0.8 or norm(value) < 0.15 else "#17202a"
            ax.text(j, i, f"{value:.2f}×", ha="center", va="center", fontsize=12, weight="bold", color=color)
    ax.set_xticks(np.arange(-0.5, len(lengths)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(heads)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(mesh, ax=ax, label="Speedup (higher is faster)", shrink=0.8)
    bar.set_ticks([norm.vmin, 1.0, *range(2, int(np.ceil(norm.vmax)))])
    state = "nonzero" if data["initial_state"] else "zero"
    hardware = data.get("environment", {}).get("device_name", data["device"])
    fig.supxlabel(
        f"{hardware} · D = chunk size = 128 · GQA group {data['group_size']} · {state} initial state · gate scale {data['gate_scale']:g}\n"
        f"5 checked warmups · median of 15 × 20 graph calls · allocation excluded · gray = correctness failure",
        fontsize=10,
    )
    output = args.output or args.input.with_suffix("")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180)
    svg = output.with_suffix(".svg")
    fig.savefig(svg)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    print(f"Saved {output.with_suffix('.png')} and {output.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
