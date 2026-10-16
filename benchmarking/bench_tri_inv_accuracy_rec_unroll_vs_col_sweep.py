"""
Numerical accuracy benchmark comparing the triangular inverse recursive unroll
kernel against the vector-only column sweep kernel, measured in bits of
precision against a float64 reference.

For each matrix, bits = -log2(relative error), where the relative error is
either the normwise Frobenius error ||X - X*||_F / ||X*||_F or the max
elementwise (componentwise) error max_ij |X_ij - X*_ij| / |X*_ij| over the
upper triangle (the structural nonzeros). The max absolute elementwise error
max_ij |X_ij - X*_ij| is reported as well. As a ceiling, the same metrics are
reported for the exact inverse rounded to the output dtype ("fp16_rounding"):
no kernel that writes fp16 can do better than that.
"""

import csv
import functools
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from pto_kernels import pto_tri_inv_col_sweep, pto_tri_inv_rec_unroll

torch.manual_seed(42)

# Both kernels take an implicit unit diagonal, but read opposite triangles:
# rec_unroll (is_lower=False) reads the strictly upper triangle and returns
# inv(I + U), while col_sweep reads the strictly lower triangle and returns
# inv(I + L). Each algorithm gets the layout it needs and is checked against
# the inverse of exactly the tensor it was handed.
ALGOS = {
    "rec_unroll": (
        lambda U: U,
        lambda inp: pto_tri_inv_rec_unroll(inp, is_bsnd_format=False),
    ),
    "col_sweep": (
        lambda U: U.transpose(-1, -2).contiguous(),
        pto_tri_inv_col_sweep,
    ),
}
BASELINE = "fp16_rounding"


@functools.lru_cache(maxsize=None)
def scale_for_cond(n: int, cond_target: float) -> float:
    """Scale that puts cond(I + scale * A) at `cond_target`, for a strictly
    upper triangular A of size n with entries uniform in (-1, 1). Bisected on a
    fixed reference draw, as in tests/test_tri_inv_rec_unroll.py."""
    g = torch.Generator().manual_seed(0)
    a = torch.triu(torch.rand((n, n), generator=g, dtype=torch.float64) * 2 - 1, 1)
    eye = torch.eye(n, dtype=torch.float64)
    lo, hi = 0.0, 50.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if float(torch.linalg.cond(eye + mid * a)) < cond_target:
            lo = mid
        else:
            hi = mid
    return lo


def conditioned_triu_matrix(n: int, batch: int, cond_target: float):
    base = torch.triu(torch.rand((batch, 1, n, n)) * 2 - 1, diagonal=1)
    return scale_for_cond(n, cond_target) * base


def exact_inverse(A: torch.Tensor) -> torch.Tensor:
    """float64 inverse of I + A over the last two dimensions."""
    n = A.shape[-1]
    return torch.linalg.inv(torch.eye(n, dtype=torch.float64) + A.double())


def accuracy(actual: torch.Tensor, golden: torch.Tensor):
    """Per-matrix normwise bits, max elementwise relative bits and max
    elementwise absolute error."""
    n = golden.shape[-1]
    diff = (actual.double() - golden).abs()
    tiny = torch.finfo(torch.float64).tiny
    frob = diff.flatten(-2).norm(dim=-1) / golden.flatten(-2).norm(dim=-1)
    # Only the upper triangle of inv(I + U) is structurally nonzero; entries
    # that vanish there by cancellation would make the ratio meaningless.
    mask = torch.ones(n, n, dtype=torch.bool).triu() & (golden.abs() > tiny)
    rel = torch.where(mask, diff / golden.abs().clamp_min(tiny), 0.0)
    elem = rel.flatten(-2).amax(dim=-1)
    abs_err = diff.flatten(-2).amax(dim=-1)
    return (
        -torch.log2(frob.clamp_min(tiny)),
        -torch.log2(elem.clamp_min(tiny)),
        abs_err,
    )


def plot_csv(path, metric: str, ylabel: str, suffix: str):
    series = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["algo"], float(row["cond_target"]))
            series.setdefault(key, [[], []])
            series[key][0].append(int(row["matrix_size"]))
            series[key][1].append(float(row[metric]))

    conds = sorted({cond for _, cond in series})
    ncols = min(4, len(conds))
    nrows = -(-len(conds) // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=True, squeeze=False
    )
    axes = axes.flatten()
    for ax in axes[len(conds) :]:
        ax.set_visible(False)
    for ax, cond in zip(axes, conds):
        for (algo, c), (xs, ys) in sorted(series.items()):
            if c != cond:
                continue
            order = sorted(range(len(xs)), key=xs.__getitem__)
            ax.plot(
                [xs[i] for i in order],
                [ys[i] for i in order],
                marker="o",
                linestyle="--" if algo == BASELINE else "-",
                label=algo,
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Matrix size")
        ax.set_title(f"cond(I + A) = {cond:g}")
        ax.grid(True)
    for ax in axes[::ncols]:
        ax.set_ylabel(ylabel)
    axes[0].legend()
    fig.suptitle("tri_inv accuracy: rec_unroll vs col_sweep (fp16)")
    fig.tight_layout()
    fig.savefig(str(path).replace(".csv", f"_{suffix}.png"))
    plt.close(fig)


def bench(
    path: str = "benchmark_data/tri_inv_accuracy_rec_unroll_vs_col_sweep.csv",
    batch: int = 64,
):
    rows = [
        [
            "algo",
            "dtype",
            "matrix_size",
            "cond_target",
            "mean_cond",
            "mean_bits_frob",
            "min_bits_frob",
            "mean_bits_elem",
            "min_bits_elem",
            "max_abs_err",
            "bits_max_abs_err",
        ]
    ]
    dtype, dtype_name = torch.float16, "fp16"
    for n in (16, 32, 64, 128):
        for cond_target in (1.5, 4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0):
            # One draw shared by all algorithms, so they see identical matrices.
            U = conditioned_triu_matrix(n, batch, cond_target).to(dtype)
            golden = exact_inverse(U)
            mean_cond = float(
                torch.linalg.cond(torch.eye(n, dtype=torch.float64) + U.double()).mean()
            )
            if golden.abs().max() > torch.finfo(dtype).max:
                # Ill-conditioned enough that inv(I + A) itself overflows the
                # output dtype; no kernel can be scored meaningfully here.
                print(
                    f"skip: N={n}, cond={cond_target:g} inverse overflows {dtype_name}"
                )
                continue
            results = {BASELINE: golden.to(dtype)}
            for algo_name, (make_input, algo) in ALGOS.items():
                inp = make_input(U)
                out = algo(inp.npu()).cpu()
                torch.npu.synchronize()
                # col_sweep returns inv(I + U^T) = inv(I + U)^T; undo the
                # transpose so every result is compared with the same golden.
                results[algo_name] = out if inp is U else out.transpose(-1, -2)

            for algo_name, actual in results.items():
                bits_frob, bits_elem, abs_err = accuracy(actual, golden)
                stats = [
                    float(bits_frob.mean()),
                    float(bits_frob.min()),
                    float(bits_elem.mean()),
                    float(bits_elem.min()),
                    float(abs_err.max()),
                    float(-torch.log2(abs_err.max())),
                ]
                print(
                    f"{algo_name:>13}, {dtype_name}, N={n:>3}, "
                    f"cond={cond_target:>4g} (mean {mean_cond:.2f}), "
                    f"bits frob mean/min {stats[0]:5.2f}/{stats[1]:5.2f}, "
                    f"elem rel mean/min {stats[2]:5.2f}/{stats[3]:5.2f}, "
                    f"max abs err {stats[4]:.2e} ({stats[5]:5.2f} bits)"
                )
                rows.append([algo_name, dtype_name, n, cond_target, mean_cond, *stats])

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)
    print(output)
    plot_csv(output, "mean_bits_frob", "Bits of precision (normwise, mean)", "frob")
    plot_csv(
        output,
        "bits_max_abs_err",
        "Bits of precision: -log2(max abs elementwise error)",
        "elem",
    )


if __name__ == "__main__":
    bench()
