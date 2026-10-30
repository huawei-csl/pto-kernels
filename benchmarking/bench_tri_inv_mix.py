"""
Performance benchmark comparing the mixed cube/vector triangular inverse
kernel against the two single-engine kernels it is built from.

``tri_inv_mix`` gives the leading ``cube_fraction`` of the tiles (a measured
per-matrix-size default) to the cube units, which run the recursive unroll algorithm, and the
rest to the vector units, which run the column sweep. The question this
benchmark answers is whether that beats running either algorithm alone over all
the tiles, and from which problem size onwards. Pass ``--cube-fraction`` to
override the split, e.g. to find the fraction at which both engines finish
together: it is balanced when it equals t_vec / (t_cube + t_vec), with t_cube
and t_vec the rec_unroll and col_sweep times over the same tiles.

Run with ``--layout dense`` (default) for contiguous row-major tiles of shape
(block_dim_x, block_dim_y, n, n), or ``--layout bsnd`` for the same tiles laid
out as a [B, S, N, D] tensor (B=1, S=block_dim_x * n, N=block_dim_y, D=n),
where consecutive tile rows are N * D elements apart in memory.

The tile counts come from ``--block-dims``, given as ``bdx,bdy`` pairs; the
default sweep runs 96 up to 24576 tiles per matrix size.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from pto_kernels import pto_tri_inv_col_sweep, pto_tri_inv_mix, pto_tri_inv_rec_unroll
from utils import do_bench

matplotlib.use("Agg")


def to_bsnd(U: torch.Tensor) -> torch.Tensor:
    """(num_chunks, N, D, D) tiles -> [1, num_chunks * D, N, D] BSND tensor."""
    num_chunks, num_heads, n, _ = U.shape
    return U.transpose(1, 2).contiguous().reshape(1, num_chunks * n, num_heads, n)


# Extra keyword arguments of pto_tri_inv_mix, set from --cube-fraction.
MIX_KWARGS: dict = {}

# Each entry maps an algorithm to (make_input, run), where make_input turns a
# strictly upper triangular (block_dim_x, block_dim_y, n, n) tensor into the
# kernel's input. All three kernels take an implicit unit diagonal, so the same
# strictly triangular tensor means the same matrices to all of them.
#
# Unlike bench_tri_inv_rec_unroll_vs_col_sweep.py, which lets the column sweep
# take its cheaper default (is_lower=True) on a transposed input, every kernel
# here is run with is_lower=False on one and the same tensor: the mixed kernel
# hands that single input to both algorithms, so a comparison it takes part in
# has to feed all three the same way. The column sweep therefore pays its upper
# triangular transpose here and is a little slower than in that benchmark.
ALGOS = {
    "dense": {
        "mix": (
            lambda U: U,
            lambda inp: pto_tri_inv_mix(
                inp, is_bsnd_format=False, is_lower=False, **MIX_KWARGS
            ),
        ),
        "rec_unroll": (
            lambda U: U,
            lambda inp: pto_tri_inv_rec_unroll(
                inp, is_bsnd_format=False, is_lower=False
            ),
        ),
        "col_sweep": (
            lambda U: U,
            lambda inp: pto_tri_inv_col_sweep(inp, is_lower=False),
        ),
    },
    "bsnd": {
        "mix": (
            to_bsnd,
            lambda inp: pto_tri_inv_mix(
                inp, is_bsnd_format=True, is_lower=False, **MIX_KWARGS
            ),
        ),
        "rec_unroll": (
            to_bsnd,
            lambda inp: pto_tri_inv_rec_unroll(
                inp, is_bsnd_format=True, is_lower=False
            ),
        ),
        "col_sweep": (
            to_bsnd,
            lambda inp: pto_tri_inv_col_sweep(inp, is_lower=False, is_bsnd_format=True),
        ),
    },
}

# (block_dim_x, block_dim_y) pairs, i.e. 96, 384, 1536, 6144 and 24576 tiles:
# enough of a span to cross over from launch-bound to engine-bound and then to
# stay engine-bound for two more quadruplings, which is where the mixed kernel
# is meant to pay off. The largest entry allocates 2 * 24576 * 128 * 128 * 2
# bytes = 1.6 GB at N=128, so it fits an HBM device but not a small one; pass
# --block-dims to cut it down or to push it further.
BLOCK_DIMS = ((24, 4), (24, 16), (24, 64), (24, 256), (24, 1024))

# More iterations than do_bench's defaults: this is a three-way comparison and
# the differences at the small end are a few tens of microseconds, which the
# default 15 iterations do not resolve -- they come out non-monotone in the
# tile count, which the kernels are not.
WARMUP_ITERS, BENCHMARK_ITERS = 20, 100


def random_triu_matrix(n: int, block_dim_x: int, block_dim_y: int, scale: float = 0.1):
    return scale * torch.triu(torch.rand((block_dim_x, block_dim_y, n, n)), diagonal=1)


def plot_csv(path, layout: str = "dense"):
    series = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["algo"], row["block_dim_x"], row["block_dim_y"])
            series.setdefault(key, {})[int(row["matrix_size"])] = float(
                row["bandwidth_gbps"]
            )

    plt.figure()
    sizes = sorted({n for vals in series.values() for n in vals})
    width = 0.8 / len(series)
    algos = sorted({algo for algo, _, _ in series})
    bdys = sorted({bdy for _, _, bdy in series}, key=int)
    colors = {algo: f"C{k}" for k, algo in enumerate(algos)}
    patterns = ("", "//", "..", "xx", "\\\\", "++")
    hatches = {bdy: patterns[k % len(patterns)] for k, bdy in enumerate(bdys)}
    for i, ((algo, bdx, bdy), vals) in enumerate(
        sorted(series.items(), key=lambda kv: (kv[0][0], int(kv[0][1]), int(kv[0][2])))
    ):
        offset = (i - (len(series) - 1) / 2) * width
        plt.bar(
            [j + offset for j in range(len(sizes))],
            [vals.get(n, 0.0) for n in sizes],
            width=width,
            color=colors[algo],
            hatch=hatches[bdy],
            edgecolor="white",
            label=f"{algo}, bdx={bdx}, bdy={bdy}",
        )
    plt.xticks(range(len(sizes)), [str(n) for n in sizes])
    plt.xlabel("Matrix size")
    plt.ylabel("Bandwidth (GB/s)")
    plt.title(f"tri_inv: mix vs rec_unroll vs col_sweep ({layout})")
    plt.legend(fontsize="small")
    plt.grid(True, axis="y")
    plt.gca().set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(str(path).replace(".csv", ".png"))
    plt.close()


def print_speedups(times: dict):
    """Speedup of the mixed kernel over the better of the two single-engine
    kernels, which is what it has to beat to be worth launching."""
    print()
    print(
        f"{'N':>5} {'bdx':>5} {'bdy':>5} {'tiles':>7} "
        f"{'mix ms':>9} {'best ms':>9} {'best':>11} {'speedup':>8}"
    )
    for (n, bdx, bdy), per_algo in sorted(times.items()):
        if "mix" not in per_algo:
            continue
        others = {a: t for a, t in per_algo.items() if a != "mix"}
        if not others:
            continue
        best_algo = min(others, key=others.get)
        best = others[best_algo]
        print(
            f"{n:>5} {bdx:>5} {bdy:>5} {bdx * bdy:>7} "
            f"{per_algo['mix']:>9.3f} {best:>9.3f} {best_algo:>11} "
            f"{best / per_algo['mix']:>7.2f}x"
        )


def bench(path: str | None = None, layout: str = "dense", block_dims=BLOCK_DIMS):
    if path is None:
        suffix = "" if layout == "dense" else f"_{layout}"
        path = f"benchmark_data/tri_inv_mix{suffix}.csv"
    rows = [
        [
            "algo",
            "layout",
            "dtype",
            "matrix_size",
            "block_dim_x",
            "block_dim_y",
            "shape",
            "time_ms",
            "bandwidth_gbps",
        ]
    ]
    times: dict = defaultdict(dict)
    dtype, dtype_name = torch.float16, "fp16"
    for algo_name, (make_input, algo) in ALGOS[layout].items():
        for n in (16, 32, 64, 128):
            for block_dim_x, block_dim_y in block_dims:
                U = random_triu_matrix(n, block_dim_x, block_dim_y).to(dtype)
                U = make_input(U).npu()
                ms = do_bench(
                    lambda inp=U, fn=algo: fn(inp),
                    warmup_iters=WARMUP_ITERS,
                    benchmark_iters=BENCHMARK_ITERS,
                    unit="ms",
                )
                n_el = U.numel()
                # I/O: Input and output have 16 bits, so (2 + 2) bytes per element
                gbps = (2 + 2) * n_el / (ms / 1e3) / 1e9
                shape = "x".join(str(d) for d in U.shape)
                times[(n, block_dim_x, block_dim_y)][algo_name] = ms
                print(
                    f"{algo_name}, {layout}, {dtype_name}, N={n}, bdx={block_dim_x}, "
                    f"bdy={block_dim_y}, shape={shape}, {ms:.3f} ms, {gbps:.3f} GB/s"
                )
                rows.append(
                    [
                        algo_name,
                        layout,
                        dtype_name,
                        n,
                        block_dim_x,
                        block_dim_y,
                        shape,
                        ms,
                        gbps,
                    ]
                )

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)
    print(output)
    plot_csv(output, layout)
    print_speedups(times)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layout",
        nargs="+",
        choices=sorted(ALGOS),
        default=["dense"],
        help="input layout(s) to benchmark",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="output CSV path (only valid with a single --layout)",
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
        "--cube-fraction",
        type=float,
        default=None,
        help="fraction of the tiles the mix kernel runs on the cube units "
        "(default: the kernel's per-matrix-size value)",
    )
    args = parser.parse_args()
    if args.cube_fraction is not None:
        if not 0.0 <= args.cube_fraction <= 1.0:
            parser.error("--cube-fraction must be in [0, 1]")
        MIX_KWARGS["cube_fraction"] = args.cube_fraction
    if args.output is not None and len(args.layout) > 1:
        parser.error("--output requires a single --layout")
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
    for lay in args.layout:
        bench(args.output, lay, block_dims)
