"""
Performance benchmark comparing the triangular inverse recursive unroll kernel
against the vector-only column sweep kernel.

Run with ``--layout dense`` (default) for contiguous row-major tiles of shape
(block_dim_x, block_dim_y, n, n), or ``--layout bsnd`` for the same tiles laid
out as a [B, S, N, D] tensor (B=1, S=block_dim_x * n, N=block_dim_y, D=n),
where consecutive tile rows are N * D elements apart in memory.
"""

import argparse
import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from pto_kernels import pto_tri_inv_col_sweep, pto_tri_inv_rec_unroll
from utils import do_bench

matplotlib.use("Agg")


def to_bsnd(U: torch.Tensor) -> torch.Tensor:
    """(num_chunks, N, D, D) tiles -> [1, num_chunks * D, N, D] BSND tensor."""
    num_chunks, num_heads, n, _ = U.shape
    return U.transpose(1, 2).contiguous().reshape(1, num_chunks * n, num_heads, n)


# Each entry maps an algorithm to (make_input, run), where make_input turns a
# strictly upper triangular (block_dim_x, block_dim_y, n, n) tensor into the
# kernel's input. Both kernels take an implicit unit diagonal.
#
# Dense: rec_unroll (is_lower=False) reads the strictly upper triangle, while
# col_sweep (default is_lower=True) reads the strictly lower one, so col_sweep
# gets the transpose -- same shape, same element count, same amount of work.
#
# BSND: both kernels are fed the same upper triangular BSND tensor with
# is_lower=False.
ALGOS = {
    "dense": {
        "rec_unroll": (
            lambda U: U,
            lambda inp: pto_tri_inv_rec_unroll(inp, is_bsnd_format=False),
        ),
        "col_sweep": (
            lambda U: U.transpose(-1, -2).contiguous(),
            pto_tri_inv_col_sweep,
        ),
    },
    "bsnd": {
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


def random_triu_matrix(n: int, block_dim_x: int, block_dim_y: int, scale: float = 0.1):
    U = scale * torch.triu(torch.rand((block_dim_x, block_dim_y, n, n)), diagonal=1)
    return U


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
    hatches = dict(zip(bdys, ("", "//", "..", "xx")))
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
    plt.title(f"tri_inv: rec_unroll vs col_sweep ({layout})")
    plt.legend()
    plt.grid(True, axis="y")
    plt.gca().set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(str(path).replace(".csv", ".png"))
    plt.close()


def bench(path: str | None = None, layout: str = "dense"):
    if path is None:
        suffix = "" if layout == "dense" else f"_{layout}"
        path = f"benchmark_data/tri_inv_rec_unroll_vs_col_sweep{suffix}.csv"
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
    dtype, dtype_name = torch.float16, "fp16"
    for algo_name, (make_input, algo) in ALGOS[layout].items():
        for n in (16, 32, 64, 128):
            for block_dim_x, block_dim_y in ((24, 4), (24, 8), (24, 16)):
                U = random_triu_matrix(n, block_dim_x, block_dim_y).to(dtype)
                U = make_input(U).npu()
                ms = do_bench(lambda inp=U, fn=algo: fn(inp), unit="ms")
                n_el = U.numel()
                # I/O: Input and output have 16 bits, so (2 + 2) bytes per element
                gbps = (2 + 2) * n_el / (ms / 1e3) / 1e9
                shape = "x".join(str(d) for d in U.shape)
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
    args = parser.parse_args()
    if args.output is not None and len(args.layout) > 1:
        parser.error("--output requires a single --layout")
    for lay in args.layout:
        bench(args.output, lay)
