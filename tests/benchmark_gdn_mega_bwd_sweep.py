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

    modes = {(r.get("inverse_cache", False), r.get("gate_cache", False)) for r in rows}
    if len(modes) != 1:
        raise ValueError("A heatmap must not mix cache modes")
    cached = modes == {(True, True)}
    dtypes = {r.get("cache_dtype", "float32") for r in rows}
    if cached and len(dtypes) != 1:
        raise ValueError("A heatmap must not mix inverse cache dtypes")
    caption = (
        f"backward uses saved {next(iter(dtypes))} inverse + gates/beta; forward saves these buffers"
        if cached
        else "backward recomputes; forward saves no caches"
    )
    heads_grid = sorted({r["global_value_heads"] for r in rows}, reverse=True)
    lengths = sorted({r["tokens"] for r in rows})
    tps = sorted({r["tp"] for r in rows})
    ratios = [r["bwd_fwd_ratio"] for r in rows if r.get("bwd_fwd_ratio") is not None]
    norm = Normalize(vmin=0, vmax=max(ratios, default=1))
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#eeeeee")
    with plt.rc_context({"font.size": 11, "savefig.dpi": 180}):
        nrows = (len(heads_grid) + 1) // 2
        fig, axes = plt.subplots(nrows, 2, figsize=(15, 4 + 3 * nrows), squeeze=False)
        for ax in list(axes.flat)[len(heads_grid) :]:
            ax.set_visible(False)
        for heads, ax in zip(heads_grid, axes.flat):
            cells = {
                (r["tp"], r["tokens"]): r
                for r in rows
                if r["global_value_heads"] == heads
            }
            values = np.array(
                [
                    [cells[tp, length].get("bwd_fwd_ratio") for length in lengths]
                    for tp in tps
                ],
                dtype=float,
            )
            im = ax.imshow(
                np.ma.masked_invalid(values), cmap=cmap, norm=norm, aspect="auto"
            )
            for i, tp in enumerate(tps):
                for j, length in enumerate(lengths):
                    row = cells[tp, length]
                    bwd, fwd = row["median_ms"], row.get("forward_median_ms")
                    label = (
                        f"{bwd:.2f} / {fwd:.2f}\n{bwd / fwd:.1f}×"
                        if bwd is not None and fwd is not None
                        else (f"{bwd:.2f} / N/A" if bwd is not None else "N/A")
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
            ax.set_xticks(
                range(len(lengths)),
                [f"{n // 1024}K" if n % 1024 == 0 else str(n) for n in lengths],
            )
            ax.set_yticks(range(len(tps)), [f"TP={tp}" for tp in tps])
            ax.set_xlabel("Sequence length (tokens)")
            ax.set_xticks(np.arange(-0.5, len(lengths), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(tps), 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=2)
            ax.tick_params(which="both", length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
        fig.subplots_adjust(
            left=0.06, right=0.88, top=0.82, bottom=0.22, hspace=0.38, wspace=0.18
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
            (
                "Cached backward / cache-saving forward (ms), then ratio"
                if cached
                else "Each cell: backward / forward milliseconds, then ratio"
            ),
            ha="center",
            fontsize=12,
        )
        fig.text(
            0.5,
            0.045,
            f"Ascend910B2 · FP16 · D=C=128 · {caption}\n"
            "Median synchronized wall time: backward includes allocation; forward uses preallocated buffers.\n"
            "Heads per device = global heads / TP; no communication. N/A: forward does not support local Hv=2, 4, 6.",
            ha="center",
            fontsize=10,
            color="#555555",
        )
        fig.savefig(destination / "heatmap.png", facecolor="white", bbox_inches="tight")
        plt.close(fig)


def measure(
    heads,
    tp,
    repeats,
    destination,
    forward=False,
    cached=False,
    cache_dtype="float32",
    lengths=LENGTHS,
    wall_only=False,
):
    # Separate processes release allocations between head/TP pairs and directions.
    command = [
        sys.executable,
        str(Path(__file__).with_name("benchmark_gdn_mega_bwd.py")),
        "--heads",
        str(heads // tp),
        "--key-heads",
        str(16 // tp),
        "--tokens",
        *map(str, lengths),
        "--repeats",
        str(repeats),
    ]
    if cached:
        command.extend(
            ("--inverse-cache", "--gate-cache", "--cache-dtype", cache_dtype)
        )
    if wall_only:
        command.append("--wall-only")
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
    if [r["tokens"] for r in rows] != list(lengths):
        raise RuntimeError(f"Incomplete benchmark output; see {log}")
    return rows


def save(rows, destination):
    temporary = destination / "results.jsonl.tmp"
    temporary.write_text("".join(json.dumps(row) + "\n" for row in rows))
    temporary.replace(destination / "results.jsonl")


def run(args):
    """Measure forward producing caches and backward consuming those exact buffers."""
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    lengths = getattr(args, "lengths", LENGTHS)
    for heads in getattr(args, "global_heads", HEADS):
        for tp in getattr(args, "tps", TPS):
            supported = heads // tp in FORWARD_HEADS
            print(
                f"Cached backward Hv={heads}, TP={tp}: "
                f"{'measuring' if supported else 'unsupported forward'}",
                flush=True,
            )
            measurements = (
                measure(
                    heads,
                    tp,
                    args.repeats,
                    destination,
                    cached=True,
                    cache_dtype=args.cache_dtype,
                    lengths=lengths,
                    wall_only=getattr(args, "wall_only", False),
                )
                if supported
                else [
                    {
                        "tokens": length,
                        "median_ms": None,
                        "samples_ms": [],
                        "forward_medians": {},
                        "forward_samples": {},
                        "forward_device": {},
                    }
                    for length in lengths
                ]
            )
            for row in measurements:
                if supported:
                    assert row["inverse_cache"] and row["gate_cache"]
                fwd = row["forward_medians"].get("forward_cache_ms")
                row.update(
                    global_value_heads=heads,
                    global_qk_heads=16,
                    tp=tp,
                    heads=heads // tp,
                    key_heads=16 // tp,
                    inverse_cache=True,
                    cache_dtype=args.cache_dtype,
                    gate_cache=True,
                    saved_values=[
                        "inverse_factors",
                        "gate_prefix_hi",
                        "gate_prefix_lo",
                        "beta",
                    ],
                    backward_status=(
                        "ok" if supported else "unsupported forward cache producer"
                    ),
                    forward_status=(
                        "ok" if supported else "unsupported local value heads"
                    ),
                    forward_median_ms=fwd,
                    bwd_fwd_ratio=row["median_ms"] / fwd if fwd is not None else None,
                    timing="synchronized wall including backward allocation; forward excluded; no TP communication",
                    forward_timing="synchronized wall, preallocated buffers, inverse cache written and gate/beta buffers retained",
                    dtype="float16",
                    head_dim=128,
                    chunk_size=128,
                    seed=42,
                )
                rows.append(row)
            save(rows, destination)
    plot(rows, destination)
    print(
        f"Saved {sum(r['backward_status'] == 'ok' for r in rows)} cached configurations "
        f"and heatmap.png to {destination}"
    )
