"""
Sweep of the ``cube_fraction`` parameter of ``tri_inv_mix``.

``tri_inv_mix`` gives the leading ``cube_fraction`` of the tiles to the cube
units, which run the recursive unroll algorithm, and the rest to the vector
units, which run the column sweep. The two parts are disjoint and run
concurrently, so the kernel ends with whichever engine is slower on its part
and the fraction that minimizes the runtime is the one that makes them finish
together. This benchmark measures the whole curve rather than that one point:
``cube_fraction=0`` and ``cube_fraction=1`` are the degenerate ends, where the
mix launch runs one algorithm over all the tiles while the other engine idles.

The two single-engine kernels are measured over the same tiles as flat
references, so every mix point can be read as a speedup over the better of
them -- which is what the mix kernel has to beat to be worth launching. From
the two reference times the sweep also prints the balance prediction
t_vec / (t_cube + t_vec), the fraction at which both parts would take equally
long if each engine scaled perfectly with its tile count, and compares it to
the fraction that actually won.

The defaults reproduce the measurement behind ``DefaultTriInvMixCubeFraction``
in ``csrc/host/torch_tri_inv_mix.h``: steps of 0.05 over 1536 to 24576 tiles,
dense and BSND. The recommendation printed at the end is the fraction with the
lowest geometric-mean runtime across the configurations of a matrix size.
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from bench_tri_inv_mix import random_triu_matrix, to_bsnd
from pto_kernels import pto_tri_inv_col_sweep, pto_tri_inv_mix, pto_tri_inv_rec_unroll
from utils import do_bench

matplotlib.use("Agg")

# Each layout maps to (make_input, mix, rec_unroll, col_sweep), where
# make_input turns a strictly upper triangular (bdx, bdy, n, n) tensor into the
# kernel's input and the three callables invert it. As in bench_tri_inv_mix.py,
# every kernel is run with is_lower=False on one and the same tensor: the mixed
# kernel hands that single input to both algorithms, so a comparison it takes
# part in has to feed all three the same way.
LAYOUTS = {
    "dense": (
        lambda U: U,
        lambda inp, f: pto_tri_inv_mix(
            inp, is_bsnd_format=False, is_lower=False, cube_fraction=f
        ),
        lambda inp: pto_tri_inv_rec_unroll(inp, is_bsnd_format=False, is_lower=False),
        lambda inp: pto_tri_inv_col_sweep(inp, is_lower=False),
    ),
    "bsnd": (
        to_bsnd,
        lambda inp, f: pto_tri_inv_mix(
            inp, is_bsnd_format=True, is_lower=False, cube_fraction=f
        ),
        lambda inp: pto_tri_inv_rec_unroll(inp, is_bsnd_format=True, is_lower=False),
        lambda inp: pto_tri_inv_col_sweep(inp, is_lower=False, is_bsnd_format=True),
    ),
}

MATRIX_SIZES = (16, 32, 64, 128)

# (block_dim_x, block_dim_y) pairs, i.e. 1536, 6144 and 24576 tiles: the range
# over which the kernel is engine-bound, which is the only range where the
# split point is what decides the runtime. Below it the launch dominates and
# the curve is flat, so sweeping there says nothing.
BLOCK_DIMS = ((24, 64), (24, 256), (24, 1024))

FRACTION_STEP = 0.05

# The mix curve is shallow near its minimum -- a few percent over several
# neighbouring fractions -- so it needs more iterations than do_bench's
# defaults to place the argmin rather than the noise.
WARMUP_ITERS, BENCHMARK_ITERS = 20, 100


def measure(fn, U) -> float:
    return do_bench(
        lambda: fn(U),
        warmup_iters=WARMUP_ITERS,
        benchmark_iters=BENCHMARK_ITERS,
        unit="ms",
    )


def bandwidth_gbps(U: torch.Tensor, ms: float) -> float:
    # I/O: Input and output have 16 bits, so (2 + 2) bytes per element
    return (2 + 2) * U.numel() / (ms / 1e3) / 1e9


def sweep(layouts, block_dims, fractions, path: Path):
    rows = [
        [
            "algo",
            "cube_fraction",
            "layout",
            "dtype",
            "matrix_size",
            "block_dim_x",
            "block_dim_y",
            "tiles",
            "shape",
            "time_ms",
            "bandwidth_gbps",
        ]
    ]
    # (layout, n, bdx, bdy) -> {"rec_unroll": ms, "col_sweep": ms,
    #                           "mix": {fraction: ms}}
    results: dict = defaultdict(lambda: {"mix": {}})
    dtype_name = "fp16"

    for layout in layouts:
        make_input, mix, rec_unroll, col_sweep = LAYOUTS[layout]
        for n in MATRIX_SIZES:
            for bdx, bdy in block_dims:
                U = random_triu_matrix(n, bdx, bdy).to(torch.float16)
                U = make_input(U).npu()
                shape = "x".join(str(d) for d in U.shape)
                key = (layout, n, bdx, bdy)

                def record(algo, fraction, ms, U=U, shape=shape, n=n, bdx=bdx, bdy=bdy):
                    rows.append(
                        [
                            algo,
                            "" if fraction is None else f"{fraction:.2f}",
                            layout,
                            dtype_name,
                            n,
                            bdx,
                            bdy,
                            bdx * bdy,
                            shape,
                            ms,
                            bandwidth_gbps(U, ms),
                        ]
                    )

                for algo, fn in (("rec_unroll", rec_unroll), ("col_sweep", col_sweep)):
                    ms = measure(fn, U)
                    results[key][algo] = ms
                    record(algo, None, ms)
                    print(
                        f"{algo:>10}, {layout:>5}, N={n:>3}, tiles={bdx * bdy:>6}, "
                        f"shape={shape}, {ms:.3f} ms"
                    )

                for fraction in fractions:
                    ms = measure(lambda inp, f=fraction: mix(inp, f), U)
                    results[key]["mix"][fraction] = ms
                    record("mix", fraction, ms)
                    print(
                        f"{'mix':>10}, {layout:>5}, N={n:>3}, tiles={bdx * bdy:>6}, "
                        f"shape={shape}, cube_fraction={fraction:.2f}, {ms:.3f} ms"
                    )

                del U
                torch.npu.empty_cache()

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)
    print(path)
    return results


def print_per_config(results: dict):
    """One line per configuration: where the sweep's minimum is, how it
    compares to the better single-engine kernel, and how far it sits from the
    fraction that the two reference times predict would balance the engines."""
    print()
    print(
        f"{'layout':>6} {'N':>4} {'tiles':>6} {'cube ms':>8} {'vec ms':>8} "
        f"{'predict':>8} {'best f':>7} {'best ms':>8} {'vs single':>10} "
        f"{'f=0.50':>8}"
    )
    for (layout, n, bdx, bdy), res in sorted(results.items()):
        t_cube, t_vec = res["rec_unroll"], res["col_sweep"]
        by_fraction = res["mix"]
        best_f = min(by_fraction, key=by_fraction.get)
        best_ms = by_fraction[best_f]
        # If each engine's time scaled linearly with its share of the tiles,
        # the two parts would finish together at this fraction.
        predicted = t_vec / (t_cube + t_vec)
        half = min(by_fraction, key=lambda f: abs(f - 0.5))
        print(
            f"{layout:>6} {n:>4} {bdx * bdy:>6} {t_cube:>8.3f} {t_vec:>8.3f} "
            f"{predicted:>8.2f} {best_f:>7.2f} {best_ms:>8.3f} "
            f"{min(t_cube, t_vec) / best_ms:>9.2f}x "
            f"{min(t_cube, t_vec) / by_fraction[half]:>7.2f}x"
        )


def print_recommendation(results: dict, fractions):
    """The fraction to default to per matrix size: the one with the lowest
    geometric mean, over that size's configurations, of the runtime relative to
    the best fraction for each configuration. Taking it relative first keeps a
    single slow configuration -- the largest tile count is an order of
    magnitude slower than the smallest -- from deciding the mean on its own."""
    per_size: dict = defaultdict(list)
    for (_, n, _, _), res in results.items():
        by_fraction = res["mix"]
        best = min(by_fraction.values())
        per_size[n].append({f: ms / best for f, ms in by_fraction.items()})

    print()
    print("Relative runtime (geometric mean over configurations, 1.00 = best):")
    print()
    header = "    N " + " ".join(f"{f:>6.2f}" for f in fractions)
    print(header)
    best_per_size = {}
    for n in sorted(per_size):
        geo = {
            f: math.exp(sum(math.log(r[f]) for r in per_size[n]) / len(per_size[n]))
            for f in fractions
        }
        best_per_size[n] = min(geo, key=geo.get)
        print(f"{n:>5} " + " ".join(f"{geo[f]:>6.3f}" for f in fractions))
    print()
    for n, f in best_per_size.items():
        print(f"  matrix_size {n:>3}: cube_fraction {f:.2f}")


def plot(results: dict, fractions, path: Path):
    """Speedup over the better single-engine kernel against cube_fraction, one
    panel per matrix size. The dashed line at 1.0 is that kernel: above it the
    mix kernel is worth launching, below it one engine alone is faster."""
    sizes = sorted({n for _, n, _, _ in results})
    tile_counts = sorted({bdx * bdy for _, _, bdx, bdy in results})
    layouts = sorted({layout for layout, _, _, _ in results})
    colors = {t: f"C{k}" for k, t in enumerate(tile_counts)}
    styles = dict(zip(layouts, ("-", "--", ":")))

    ncols = 2
    nrows = (len(sizes) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(11, 4 * nrows), squeeze=False, sharex=True
    )
    for ax, n in zip(axes.flat, sizes):
        for (layout, size, bdx, bdy), res in sorted(results.items()):
            if size != n:
                continue
            single = min(res["rec_unroll"], res["col_sweep"])
            ax.plot(
                fractions,
                [single / res["mix"][f] for f in fractions],
                styles[layout],
                color=colors[bdx * bdy],
                marker=".",
                label=f"{layout}, {bdx * bdy} tiles",
            )
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(f"matrix size {n}")
        ax.set_xlabel("cube_fraction")
        ax.set_ylabel("speedup over best single-engine kernel")
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.legend(fontsize="small")
    for ax in axes.flat[len(sizes) :]:
        ax.set_visible(False)
    fig.suptitle("tri_inv_mix: cube_fraction sweep vs rec_unroll and col_sweep")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layout",
        nargs="+",
        choices=sorted(LAYOUTS),
        default=sorted(LAYOUTS),
        help="input layout(s) to benchmark",
    )
    parser.add_argument(
        "--output",
        default="benchmark_data/tri_inv_mix_cube_fraction.csv",
        help="output CSV path; the plot is written next to it as .png",
    )
    parser.add_argument(
        "--block-dims",
        nargs="+",
        metavar="BDX,BDY",
        default=None,
        help="tile counts to benchmark, as bdx,bdy pairs (default: "
        + " ".join(f"{x},{y}" for x, y in BLOCK_DIMS)
        + "); each pair benchmarks bdx * bdy tiles",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=FRACTION_STEP,
        help=f"cube_fraction step (default: {FRACTION_STEP})",
    )
    args = parser.parse_args()

    block_dims = BLOCK_DIMS
    if args.block_dims is not None:
        try:
            block_dims = tuple(
                tuple(int(v) for v in pair.split(",")) for pair in args.block_dims
            )
        except ValueError:
            parser.error("--block-dims takes integer bdx,bdy pairs, e.g. 24,256")
        if any(len(pair) != 2 or min(pair) < 1 for pair in block_dims):
            parser.error("--block-dims takes positive bdx,bdy pairs, e.g. 24,256")
    if not 0.0 < args.step <= 1.0:
        parser.error("--step must be in (0, 1]")
    fractions = [
        min(1.0, k * args.step) for k in range(int(round(1.0 / args.step)) + 1)
    ]

    output = Path(args.output)
    results = sweep(args.layout, block_dims, fractions, output)
    print_per_config(results)
    print_recommendation(results, fractions)
    plot(results, fractions, output.with_suffix(".png"))
