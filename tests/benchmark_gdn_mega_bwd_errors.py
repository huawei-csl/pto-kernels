"""Small saved-forward backward audit using the exact pytest oracle/assertions."""

import json

CASES = (
    ("Random · 128", 128, "both"),
    ("Tail · 129", 129, "both"),
    ("Zero beta · 129", 129, "zero_beta"),
    ("Identical keys · 257", 257, "replace"),
    ("Weak decay · 513", 513, "weak"),
    ("Near-dependent + weak · 257", 257, "correlated_weak"),
    ("Strong decay · 256", 256, "strong_decay"),
)
GRADIENTS = ("dq", "dk", "dv", "dg", "dbeta", "dh0")
FLASHQLA_SOURCE = "https://github.com/QwenLM/FlashQLA/blob/a14dbf3474514e1674526431e960325f72081e9f/tests/gdr_common.py#L189-L200"


def plot(report, destination, relative=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    import numpy as np

    rows = report["cases"]

    def value(row, gradient):
        item = row["gradients"].get(gradient, {})
        if relative:
            result = item.get("flashqla", {}).get("relative_l2")
            return np.nan if result is None else 100 * result
        return item.get("error", np.nan)

    errors = np.array([[value(r, g) for g in GRADIENTS] for r in rows])
    positive = errors[np.isfinite(errors) & (errors > 0)]
    lo, hi = (
        (min(positive.min(), 1e-5), max(positive.max(), 1e-4))
        if positive.size
        else (1e-8, 1)
    )
    fig, ax = plt.subplots(figsize=(10, 2.8 + 0.5 * len(rows)), layout="constrained")
    im = ax.imshow(
        np.ma.masked_invalid(errors if relative else np.maximum(errors, lo)),
        norm=Normalize(0, max(2, hi)) if relative else LogNorm(lo, hi),
        cmap="YlOrRd",
        aspect="auto",
    )
    for i, row in enumerate(rows):
        for j, g in enumerate(GRADIENTS):
            item = row["gradients"].get(g)
            label = (
                "N/A"
                if item is None
                else (
                    f"{errors[i, j]:.3f}%"
                    + (" *" if not item["flashqla"]["passed"] else "")
                    if relative
                    else f"{item['error']:.2e}"
                    + (" *" if item["error"] > item["budget"] else "")
                )
            )
            color = (
                "white"
                if np.isfinite(errors[i, j]) and im.norm(max(errors[i, j], lo)) > 0.65
                else "#222222"
            )
            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=10)
    ax.set_xticks(range(len(GRADIENTS)), GRADIENTS)
    ax.set_yticks(range(len(rows)), [r["label"] for r in rows])
    ax.tick_params(length=0)
    ax.set_title(
        (
            "Backward L2 relative error vs FP64 autograd"
            if relative
            else "Backward max absolute error vs FP32 autograd"
        )
        + f"\nFP16 inputs · {report['cache_dtype']} inverse + gates/beta · Hv/Hqk=8/2",
        fontsize=13,
        pad=16,
    )
    fig.colorbar(
        im,
        ax=ax,
        label=(
            "L2 relative error (%)" if relative else "Max absolute error (log colors)"
        ),
        shrink=0.8,
    )
    failed = sum(r["status"] != "passed" for r in rows)
    fig.supxlabel(
        f"{len(rows) - failed}/{len(rows)} cases pass · "
        + (
            "FlashQLA bound: ‖kernel − FP64‖₂ ≤ 0.02 × ‖FP64‖₂\n"
            if relative
            else "per gradient: error ≤ 2 × reordered FP16 Torch error + 1e-5\n"
        )
        + "* = over budget; N/A = incomplete. Actual errors/budgets and failures are saved in errors.json.",
        fontsize=10,
    )
    name = "flashqla_error_heatmap" if relative else "error_heatmap"
    fig.savefig(destination / f"{name}.png", dpi=180, facecolor="white")
    plt.close(fig)


def run(args):
    destination = args.output
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "errors.json"
    if args.plot_only:
        report = json.loads(path.read_text())
    else:
        import torch
        import torch_npu  # noqa: F401
        from test_gdn_mega_bwd import test_backward

        torch.npu.set_device(0)
        torch.npu.config.allow_internal_format = False
        report = dict(
            cache_dtype=args.cache_dtype,
            dtype="float16",
            gate_dtype="float32",
            seed=42,
            golden="Independent recursive FP32 PyTorch forward + autograd.grad; FP32 gradients retained",
            budget="Per gradient tensor: 2 * max(abs(reordered FP16 Torch - golden)) + 1e-5",
            cases=[],
        )
        if args.flashqla_bound:
            report["flashqla_bound"] = dict(
                source=FLASHQLA_SOURCE,
                threshold=0.02,
                reference="Independent recursive FP64 PyTorch autograd; identical rounded inputs and adjoints to the FP32 audit",
                scope="Applies their metric/threshold to our seven cases; not their CUDA kernel, BF16 input distribution or complete test suite",
            )
        for label, tokens, mode in CASES[:2] if args.quick else CASES:
            print(f"Numerics: {label}", flush=True)
            measured = []
            row = dict(
                label=label,
                tokens=tokens,
                mode=mode,
                heads=8,
                key_heads=2,
                status="passed",
            )
            try:
                test_backward(
                    [tokens],
                    8,
                    2,
                    mode,
                    torch.float16,
                    "npu:0",
                    "both",
                    getattr(torch, args.cache_dtype),
                    measured,
                    flashqla_bound=args.flashqla_bound,
                )
            except AssertionError as error:
                row.update(status="failed", failure=str(error))
            row["gradients"] = {
                r["gradient"]: {k: v for k, v in r.items() if k != "gradient"}
                for r in measured
            }
            report["cases"].append(row)
            path.write_text(json.dumps(report, indent=2) + "\n")
    plot(report, destination)
    if report.get("flashqla_bound"):
        plot(report, destination, relative=True)
    if any(r["status"] != "passed" for r in report["cases"]):
        raise RuntimeError(f"Numerical audit failed; see {path}")
