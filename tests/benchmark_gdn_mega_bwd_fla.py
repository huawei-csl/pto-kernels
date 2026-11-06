"""Shape-matched PTO/FLA latency heatmaps; keep benchmark differences explicit."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


def shape_key(shape):
    return tuple(
        shape[k]
        for k in (
            "batch_size",
            "sequence_length_tokens",
            "query_key_heads",
            "value_heads",
            "query_key_head_dimension",
            "value_head_dimension",
        )
    )


def compare(fla, pto):
    measured = {(1, r["tokens"], r["key_heads"], r["heads"], 128, 128): r for r in pto}
    if len(measured) != len(pto):
        raise ValueError("Duplicate PTO shapes")
    rows, seen = [], set()
    for item in fla["results"]:
        key = shape_key(item["shape"])
        if key in seen:
            raise ValueError(f"Duplicate FLA shape: {key}")
        seen.add(key)
        ours = measured.get(key)
        row = {
            "shape": item["shape"],
            "status": "matched" if ours else "missing PTO shape",
        }
        for mode, field in (("forward", "forward_cache_ms"), ("backward", "median_ms")):
            theirs = item["latency"][mode]["median_ms"]
            own = (
                None
                if ours is None
                else (
                    ours["forward_medians"][field] if mode == "forward" else ours[field]
                )
            )
            if (
                not math.isfinite(theirs)
                or theirs <= 0
                or (own is not None and (not math.isfinite(own) or own <= 0))
            ):
                raise ValueError(f"Nonpositive {mode} timing: {key}")
            row[mode] = {
                "fla_ms": theirs,
                "pto_ms": own,
                "speedup": theirs / own if own is not None else None,
            }
        if ours and not (ours["inverse_cache"] and ours["gate_cache"]):
            raise ValueError(f"PTO backward must consume both forward caches: {key}")
        rows.append(row)
    return rows


def plot(rows, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    import numpy as np

    pairs = sorted(
        {(r["shape"]["value_heads"], r["shape"]["query_key_heads"]) for r in rows},
        reverse=True,
    )
    lengths = sorted({r["shape"]["sequence_length_tokens"] for r in rows})
    cells = {
        (
            r["shape"]["value_heads"],
            r["shape"]["query_key_heads"],
            r["shape"]["sequence_length_tokens"],
        ): r
        for r in rows
    }
    ratios = [
        r[m]["speedup"]
        for r in rows
        for m in ("forward", "backward")
        if r[m]["speedup"] is not None
    ]
    if not ratios:
        raise ValueError("No matching shapes to plot")
    bound = max(1, int(np.ceil(max(abs(np.log2(ratios))))))
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#eeeeee")
    with plt.rc_context({"font.size": 11, "savefig.dpi": 180}):
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        for ax, mode in zip(axes, ("forward", "backward")):
            values = np.full((len(pairs), len(lengths)), np.nan)
            for i, (hv, hk) in enumerate(pairs):
                for j, length in enumerate(lengths):
                    item = cells.get((hv, hk, length), {}).get(mode, {})
                    speedup = item.get("speedup")
                    if speedup is None:
                        label, color = "N/A", "#555555"
                    else:
                        values[i, j] = np.log2(speedup)
                        label = f"{speedup:.2f}×\n{item['fla_ms']:.2f} / {item['pto_ms']:.2f}"
                        rgb = cmap(norm(values[i, j]))[:3]
                        color = (
                            "white"
                            if np.dot(rgb, (0.2126, 0.7152, 0.0722)) < 0.48
                            else "#222222"
                        )
                    ax.text(
                        j, i, label, ha="center", va="center", color=color, fontsize=11
                    )
            im = ax.imshow(
                np.ma.masked_invalid(values), cmap=cmap, norm=norm, aspect="auto"
            )
            ax.set_title(
                mode.capitalize(), loc="left", weight="bold", fontsize=15, pad=12
            )
            ax.set_xticks(
                range(len(lengths)),
                [f"{n // 1024}K" if n % 1024 == 0 else str(n) for n in lengths],
            )
            ax.set_yticks(range(len(pairs)), [f"{hv} / {hk}" for hv, hk in pairs])
            ax.set_xlabel("Sequence length (tokens)")
            ax.set_ylabel("Value / QK heads per device")
            ax.set_xticks(np.arange(-0.5, len(lengths), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(pairs), 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=2)
            ax.tick_params(which="both", length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
        fig.subplots_adjust(left=0.07, right=0.89, top=0.76, bottom=0.25, wspace=0.24)
        ticks = range(-bound, bound + 1)
        bar = fig.colorbar(
            im, cax=fig.add_axes((0.92, 0.30, 0.015, 0.41)), ticks=list(ticks)
        )
        bar.ax.set_yticklabels([f"{2.0**n:g}×" for n in ticks])
        bar.set_label("FLA / PTO latency · logarithmic colors")
        fig.suptitle("PTO vs FLA · Ascend910B2", fontsize=20, weight="bold", y=0.96)
        fig.text(
            0.5,
            0.90,
            "Reported speedup = FLA time / PTO time   ·   >1× means PTO is faster",
            ha="center",
            fontsize=13,
        )
        fig.text(
            0.5,
            0.85,
            "Matching shapes, different dtypes: PTO FP16 / C=128   vs   FLA BF16 / C=64",
            ha="center",
            fontsize=12,
            color="#8a3b12",
        )
        fig.text(
            0.5,
            0.14,
            "Each cell: speedup, then FLA / PTO milliseconds. Separate eager runs; median synchronized wall time.\n"
            "PTO: cache-saving forward with preallocated buffers; backward consumes saved inverse + gates/beta.\n"
            "FLA: no-grad forward; autograd backward reuses saved tensors. Different input distributions and state outputs.\n"
            "B=1, D=128; per-device head counts, no TP communication. See comparison.json for timing/allocation details.",
            ha="center",
            va="center",
            fontsize=10,
            color="#555555",
        )
        fig.savefig(
            destination / "speedup_heatmap.png", facecolor="white", bbox_inches="tight"
        )
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fla", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_data/gdn_mega_bwd/fla_comparison"),
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--plot-only", action="store_true", help="Reuse pto_results.jsonl; no NPU work"
    )
    parser.add_argument(
        "--cache-dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Only Hv=Hqk=16 at the two shortest lengths",
    )
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    fla = json.loads(args.fla.read_text())
    if args.quick:
        shortest = sorted(
            {r["shape"]["sequence_length_tokens"] for r in fla["results"]}
        )[:2]
        fla["results"] = [
            r
            for r in fla["results"]
            if r["shape"]["value_heads"] == r["shape"]["query_key_heads"] == 16
            and r["shape"]["sequence_length_tokens"] in shortest
        ]
    groups = {}
    for item in fla["results"]:
        b, length, hk, hv, k, v = shape_key(item["shape"])
        if (b, k, v) != (1, 128, 128):
            raise ValueError(f"Unsupported PTO shape: {item['shape']}")
        groups.setdefault((hv, hk), set()).add(length)
    args.output.mkdir(parents=True, exist_ok=True)
    raw = args.output / "pto_results.jsonl"
    if not args.plot_only:
        pto = []
        for (hv, hk), lengths in sorted(groups.items(), reverse=True):
            print(f"PTO cached comparison: Hv={hv}, Hqk={hk}", flush=True)
            command = [
                sys.executable,
                str(Path(__file__).with_name("benchmark_gdn_mega_bwd.py")),
                "--heads",
                str(hv),
                "--key-heads",
                str(hk),
                "--tokens",
                *map(str, sorted(lengths)),
                "--inverse-cache",
                "--gate-cache",
                "--cache-dtype",
                args.cache_dtype,
                "--wall-only",
                "--repeats",
                str(args.repeats),
            ]
            result = subprocess.run(command, text=True, capture_output=True)
            log = args.output / f"pto_hv{hv}_hk{hk}.log"
            log.write_text(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError(f"PTO benchmark failed; see {log}")
            batch = [
                json.loads(line)
                for line in result.stdout.splitlines()
                if line.startswith('{"tokens":')
            ]
            if [r["tokens"] for r in batch] != sorted(lengths):
                raise ValueError(f"Incomplete PTO benchmark: {log}")
            pto.extend(batch)
            raw.write_text("".join(json.dumps(r) + "\n" for r in pto))
    else:
        pto = [json.loads(line) for line in raw.read_text().splitlines()]
    rows = compare(fla, pto)
    root = Path(__file__).resolve().parents[1]
    sources = [
        *sorted((root / "csrc/kernel/gdn_mega_bwd").glob("*.h")),
        root / "csrc/kernel/gdn_mega_bwd/kernel.cpp",
        Path(__file__),
        Path(__file__).with_name("benchmark_gdn_mega_bwd.py"),
    ]
    output = {
        "speedup_definition": "FLA median ms / PTO median ms; >1 means PTO faster",
        "fla_source": str(args.fla.resolve()),
        "fla_sha256": hashlib.sha256(args.fla.read_bytes()).hexdigest(),
        "fla_metadata": {k: v for k, v in fla.items() if k != "results"},
        "pto_measurement": {
            "device": "Ascend910B2, NPU 0",
            "dtype": "float16",
            "gate_dtype": "float32",
            "inverse_cache_dtypes": sorted({r["cache_dtype"] for r in pto}),
            "chunk_size": 128,
            "samples_per_direction": sorted({len(r["samples_ms"]) for r in pto}),
            "forward": "Cache-saving forward; preallocated buffers; two warmup calls per mode; synchronized wall time including dispatch.",
            "backward": "Both forward caches; one warmup; synchronized wall including six output allocations, workspace allocation and native dispatch; no autograd wrapper.",
            "inputs": "q=0.1*randn, k=L2-normalized randn, v=randn, g=-0.1*uniform, beta=uniform; absent h0/dht.",
            "state_outputs": "Forward also returns final state; backward also returns dh0. FLA requests neither.",
            "caveat": "Shape-matched reported timings, not a controlled same-dtype, same-input or same-allocation benchmark.",
        },
        "source_sha256": {
            str(p.relative_to(root) if p.is_absolute() else p): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in sources
        },
        "comparisons": rows,
    }
    report = args.output / "comparison.json"
    if args.plot_only:
        previous = json.loads(report.read_text())
        for key in ("source_sha256", "pto_measurement"):
            output[key] = previous[
                key
            ]  # Preserve measurement provenance when replotting.
    report.write_text(json.dumps(output, indent=2) + "\n")
    plot(rows, args.output)
    print(
        f"Saved {sum(r['status'] == 'matched' for r in rows)} shape comparisons and speedup_heatmap.png to {args.output}"
    )


if __name__ == "__main__":
    main()
