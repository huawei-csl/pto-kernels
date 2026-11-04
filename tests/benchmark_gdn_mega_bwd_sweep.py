"""Extensive per-device TP sweep and heatmaps for benchmark_gdn_mega_bwd.py."""

import json
from pathlib import Path
import subprocess
import sys

HEADS = (64, 48, 32, 16)
TPS = (1, 2, 4, 8)
LENGTHS = (256, 1024, 4096, 16384, 32768)
FORWARD_HEADS = (8, 12, 16, 24, 32, 48, 64)


def plot(rows, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import numpy as np

    ratios = [r["bwd_fwd_ratio"] for r in rows if r.get("bwd_fwd_ratio") is not None]
    norm = Normalize(vmin=0, vmax=max(ratios, default=1))
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#eeeeee")
    with plt.rc_context({"font.size": 11, "savefig.dpi": 180}):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        for heads, ax in zip(HEADS, axes.flat):
            cells = {
                (r["tp"], r["tokens"]): r
                for r in rows
                if r["global_value_heads"] == heads
            }
            values = np.array(
                [
                    [cells[tp, length].get("bwd_fwd_ratio") for length in LENGTHS]
                    for tp in TPS
                ],
                dtype=float,
            )
            im = ax.imshow(
                np.ma.masked_invalid(values), cmap=cmap, norm=norm, aspect="auto"
            )
            for i, tp in enumerate(TPS):
                for j, length in enumerate(LENGTHS):
                    row = cells[tp, length]
                    bwd, fwd = row["median_ms"], row.get("forward_median_ms")
                    label = (
                        f"{bwd:.2f} / {fwd:.2f}\n{bwd / fwd:.1f}×"
                        if fwd is not None
                        else f"{bwd:.2f} / N/A"
                    )
                    color = (
                        "white"
                        if np.isfinite(values[i, j]) and norm(values[i, j]) > 0.6
                        else "#222222"
                    )
                    ax.text(
                        j, i, label, ha="center", va="center", color=color, fontsize=10
                    )
            ax.set_title(
                f"{heads} value heads · 16 Q/K heads", loc="left", weight="bold", pad=12
            )
            ax.set_xticks(range(5), ("256", "1K", "4K", "16K", "32K"))
            ax.set_yticks(range(4), [f"TP={tp}" for tp in TPS])
            ax.set_xlabel("Sequence length (tokens)")
            ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=2)
            ax.tick_params(which="both", length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
        fig.subplots_adjust(
            left=0.06, right=0.88, top=0.86, bottom=0.16, hspace=0.38, wspace=0.18
        )
        fig.colorbar(
            im,
            cax=fig.add_axes((0.91, 0.23, 0.018, 0.55)),
            label="Backward / forward latency (×)",
        )
        fig.suptitle(
            "GDN latency · tensor-parallel head scaling",
            fontsize=19,
            weight="bold",
            y=0.96,
        )
        fig.text(
            0.47,
            0.91,
            "Each cell: backward / forward milliseconds, then ratio",
            ha="center",
            fontsize=12,
        )
        fig.text(
            0.5,
            0.045,
            "Ascend910B2 · FP16 · D=C=128 · backward recomputes; forward saves no caches\n"
            "Median synchronized wall time: backward includes allocation; forward uses preallocated buffers.\n"
            "Heads per device = global heads / TP; no communication. N/A: forward does not support local Hv=2, 4, 6.",
            ha="center",
            fontsize=10,
            color="#555555",
        )
        for extension in ("png", "svg"):
            fig.savefig(destination / f"heatmap.{extension}", facecolor="white")
        plt.close(fig)


def measure(heads, tp, repeats, destination, forward=False):
    # Separate processes release allocations between head/TP pairs and directions.
    command = [
        sys.executable,
        str(Path(__file__).with_name("benchmark_gdn_mega_bwd.py")),
        "--heads",
        str(heads // tp),
        "--key-heads",
        str(16 // tp),
        "--tokens",
        *map(str, LENGTHS),
        "--repeats",
        str(repeats),
    ]
    if forward:
        command.append("--forward-only")
    log = destination / f"h{heads}_tp{tp}{'_fwd' if forward else ''}.log"
    result = subprocess.run(command, text=True, capture_output=True)
    log.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"Sweep failed; see {log}")
    rows = [
        json.loads(line)
        for line in result.stdout.splitlines()
        if line.startswith('{"tokens":')
    ]
    if [r["tokens"] for r in rows] != list(LENGTHS):
        raise RuntimeError(f"Incomplete benchmark output; see {log}")
    return rows


def save(rows, destination):
    temporary = destination / "results.jsonl.tmp"
    temporary.write_text("".join(json.dumps(row) + "\n" for row in rows))
    temporary.replace(destination / "results.jsonl")


def add_forward(rows, destination, repeats):
    for heads in HEADS:
        for tp in TPS:
            selected = [
                r for r in rows if r["global_value_heads"] == heads and r["tp"] == tp
            ]
            supported = heads // tp in FORWARD_HEADS
            print(
                f"Forward Hv={heads}, TP={tp}: {'measuring' if supported else 'unsupported'}",
                flush=True,
            )
            measurements = (
                measure(heads, tp, repeats, destination, forward=True)
                if supported
                else []
            )
            forward = {r["tokens"]: r for r in measurements}
            for row in selected:
                result = forward.get(row["tokens"])
                row["forward_status"] = (
                    "ok" if result else "unsupported local value heads"
                )
                row["forward_median_ms"] = (
                    result["forward_medians"]["forward_no_cache_ms"] if result else None
                )
                row["forward_samples"] = result["forward_samples"] if result else {}
                row["forward_device"] = result["forward_device"] if result else {}
                row["bwd_fwd_ratio"] = (
                    row["median_ms"] / row["forward_median_ms"] if result else None
                )
                row["forward_timing"] = (
                    "synchronized wall, preallocated buffers, no saved caches"
                )
            save(rows, destination)


def run(args):
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for heads in HEADS:
        for tp in TPS:
            print(
                f"Backward Hv={heads}, Q/K=16, TP={tp} (local {heads // tp}/{16 // tp})",
                flush=True,
            )
            for row in measure(heads, tp, args.repeats, destination):
                row.update(
                    global_value_heads=heads,
                    global_qk_heads=16,
                    tp=tp,
                    timing="synchronized wall including allocation; no TP communication",
                    dtype="float16",
                    head_dim=128,
                    chunk_size=128,
                    seed=42,
                )
                rows.append(row)
            save(rows, destination)
    add_forward(rows, destination, args.repeats)
    plot(rows, destination)
    print(f"Saved {len(rows)} configurations and heatmap.png/.svg to {destination}")
