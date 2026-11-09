"""The three GDN evaluation figures; consumes saved data, never imports Torch."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
import numpy as np


def panel(ax, values, labels, columns, rows, title, norm, cmap="YlOrRd"):
    values = np.asarray(values, dtype=float)
    colors = plt.get_cmap(cmap).copy()
    colors.set_bad("#eeeeee")
    im = ax.imshow(np.ma.masked_invalid(values), norm=norm, cmap=colors, aspect="auto")
    for i, row in enumerate(labels):
        for j, label in enumerate(row):
            rgb = colors(norm(values[i, j]))[:3]
            color = (
                "white"
                if np.isfinite(values[i, j])
                and np.dot(rgb, (0.2126, 0.7152, 0.0722)) < 0.45
                else "#222222"
            )
            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=9)
    ax.set(
        xticks=range(len(columns)),
        xticklabels=columns,
        yticks=range(len(rows)),
        yticklabels=rows,
        title=title,
    )
    ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def save(fig, path, title, caption):
    fig.suptitle(title, fontsize=16, weight="bold")
    fig.supxlabel(caption, fontsize=9)
    fig.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot(report, destination):
    from benchmarking.gdn_mega_bwd import fla_shape, GRADIENTS

    measured = {tuple(r["shape"]): r["median_ms"] for r in report["measurements"]}
    cache = report["cache_dtype"]
    cache_label = "No cache" if cache == "none" else f"{cache.upper()} cache"
    cache_detail = (
        "No saved buffers passed to backward; inverse and gates/beta recomputed."
        if cache == "none"
        else "Forward saves inverse + gates/beta; backward consumes them."
    )
    lengths = sorted({r["shape"][0] for r in report["grid"]})
    ticks = [f"{n // 1024}K" if n % 1024 == 0 else str(n) for n in lengths]
    heads = sorted({r["heads"] for r in report["grid"]}, reverse=True)
    tps = sorted({r["tp"] for r in report["grid"]})
    ratios = [m["backward"] / m["forward"] for m in measured.values()]
    norm = Normalize(0, max(ratios, default=1))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    for ax, h in zip(axes.flat, heads):
        cells = {
            (r["tp"], r["shape"][0]): measured.get(tuple(r["shape"]))
            for r in report["grid"]
            if r["heads"] == h
        }
        values, labels = [], []
        for tp in tps:
            row = [cells.get((tp, length)) for length in lengths]
            values.append([m["backward"] / m["forward"] if m else np.nan for m in row])
            labels.append(
                [
                    (
                        f"{m['backward']:.2f} / {m['forward']:.2f} ms\n{m['backward']/m['forward']:.2f}×"
                        if m
                        else "N/A"
                    )
                    for m in row
                ]
            )
        im = panel(
            ax,
            values,
            labels,
            ticks,
            [f"TP={tp}" for tp in tps],
            f"Global Hv={h}, Hqk=16",
            norm,
        )
        ax.set_xlabel("Sequence length")
    for ax in list(axes.flat)[len(heads) :]:
        ax.set_visible(False)
    fig.colorbar(
        im, ax=axes.ravel().tolist(), label="Backward / forward latency", shrink=0.7
    )
    save(
        fig,
        destination / "heatmap.png",
        f"Backward / forward · {cache_label}",
        "Each cell: backward / forward ms, then ratio. Median synchronized wall time; backward includes allocation.\n"
        f"{cache_detail} FP16 inputs, D=C=128. No TP communication.\n"
        "N/A: forward does not support local Hv=2/4/6. Repeated local shapes reuse the same timing samples.",
    )

    cases = report["precision"]
    values, labels = [], []
    for case in cases:
        row, text = [], []
        for name in GRADIENTS:
            item = case["gradients"].get(name)
            row.append(item["error"] if item else np.nan)
            if item:
                failed = item["error"] > item["budget"] or not item.get(
                    "flashqla", {}
                ).get("passed", True)
                label = f"{item['error']:.2e}" + (" *" if failed else "")
                if "flashqla" in item:
                    relative = item["flashqla"]["relative_l2"]
                    label += (
                        f"\n{100 * relative:.3g}% L2"
                        if relative is not None
                        else "\nL2 N/A"
                    )
                text.append(label)
            else:
                text.append("N/A")
        values.append(row)
        labels.append(text)
    values = np.asarray(values)
    positive = values[np.isfinite(values) & (values > 0)]
    lo, hi = (
        (min(positive.min(), 1e-5), max(positive.max(), 1e-4))
        if positive.size
        else (1e-8, 1)
    )
    fig, ax = plt.subplots(figsize=(11, 3 + 0.55 * len(cases)), layout="constrained")
    im = panel(
        ax,
        np.maximum(values, lo),
        labels,
        GRADIENTS,
        [r["label"] for r in cases],
        f"FP16 inputs · {cache_label}",
        LogNorm(lo, hi),
    )
    fig.colorbar(im, ax=ax, label="Max absolute error", shrink=0.7)
    passed = sum(r["status"] == "passed" for r in cases)
    save(
        fig,
        destination / "precision.png",
        "Backward error vs recursive FP32 Torch autograd",
        f"{passed}/{len(cases)} cases pass. Per gradient: max error ≤ 2 × reordered FP16 Torch max error + 1e-5.\n"
        "* = over budget; N/A = incomplete. Full budgets and failures in results.json."
        + (
            "\nL2 rows also check error/reference L2 ≤ 2% against FP64 (FlashQLA bound)."
            if report["flashqla_bound"]
            else ""
        ),
    )

    fla = {fla_shape(item): item for item in report["fla"]["results"]}
    pairs = sorted({s[1:] for s in fla}, reverse=True)
    lengths = sorted({s[0] for s in fla})
    ticks = [f"{n // 1024}K" if n % 1024 == 0 else str(n) for n in lengths]
    logs = [
        np.log2(item["latency"][mode]["median_ms"] / measured[s][mode])
        for s, item in fla.items()
        if s in measured
        for mode in ("forward", "backward")
    ]
    bound = max(1, np.ceil(max(map(abs, logs), default=1)))
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
    fig, axes = plt.subplots(
        1, 2, figsize=(15, 3 + 0.7 * len(pairs)), layout="constrained"
    )
    for ax, mode in zip(axes, ("forward", "backward")):
        values, labels = [], []
        for hv, hk in pairs:
            row, text = [], []
            for length in lengths:
                s = (length, hv, hk)
                if s in measured and s in fla:
                    theirs, ours = (
                        fla[s]["latency"][mode]["median_ms"],
                        measured[s][mode],
                    )
                    row.append(np.log2(theirs / ours))
                    text.append(f"{theirs/ours:.2f}×\n{theirs:.2f} / {ours:.2f}")
                else:
                    row.append(np.nan)
                    text.append("N/A")
            values.append(row)
            labels.append(text)
        im = panel(
            ax,
            values,
            labels,
            ticks,
            [f"{hv}/{hk}" for hv, hk in pairs],
            mode.capitalize(),
            norm,
            "RdYlGn",
        )
        ax.set(xlabel="Sequence length", ylabel="Local value/QK heads")
    bar = fig.colorbar(im, ax=axes.tolist(), label="FLA / PTO latency", shrink=0.7)
    bar.set_ticks(
        np.arange(-bound, bound + 1),
        labels=[f"{2.**v:g}×" for v in np.arange(-bound, bound + 1)],
    )
    config = report["fla"].get("input_configuration", {})
    fla_dtype = config.get("dtypes", {}).get("q", "unspecified")
    fla_chunk = config.get("chunk_size_tokens", "unspecified")
    save(
        fig,
        destination / "speedup.png",
        f"FLA / PTO speedup · PTO {cache_label}",
        f"Each cell: speedup (>1× means PTO faster), then FLA / PTO ms. Exact shapes; PTO FP16/C=128 vs FLA {fla_dtype}/C={fla_chunk}.\n"
        f"PTO forward uses preallocated buffers; backward includes allocation. {cache_detail}\n"
        "Input distributions, state outputs and allocation/autograd overhead differ; full FLA metadata is saved in results.json.",
    )
