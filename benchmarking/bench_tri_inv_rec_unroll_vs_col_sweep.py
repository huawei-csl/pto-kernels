"""
Performance benchmark comparing the triangular inverse recursive unroll kernel
against the vector-only column sweep kernel.
"""

import matplotlib.pyplot as plt
import torch
import csv
from pathlib import Path
from pto_kernels import pto_tri_inv_col_sweep, pto_tri_inv_rec_unroll
from utils import do_bench

# Both kernels take an implicit unit diagonal, but they read opposite
# triangles of the buffer they are handed: rec_unroll (is_lower=False) reads
# the strictly upper triangle, while col_sweep reads the strictly lower one
# (its tests pass an upper-triangular tensor transposed, whose storage is
# lower-triangular row-major). Feeding both the same strictly upper tensor
# makes col_sweep invert the identity, so each algorithm gets the transpose it
# needs -- same shape, same element count, same amount of work.
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


def random_triu_matrix(n: int, block_dim_x: int, block_dim_y: int, scale: float = 0.1):
    U = scale * torch.triu(torch.rand((block_dim_x, block_dim_y, n, n)), diagonal=1)
    return U


def plot_csv(path):
    series = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["algo"], row["block_dim_x"], row["block_dim_y"])
            series.setdefault(key, [[], []])
            series[key][0].append(int(row["matrix_size"]))
            series[key][1].append(float(row["bandwidth_gbps"]))

    for (algo, bdx, bdy), (xs, ys) in sorted(series.items()):
        order = sorted(range(len(xs)), key=xs.__getitem__)
        plt.plot(
            [xs[i] for i in order],
            [ys[i] for i in order],
            marker="o",
            label=f"{algo}, bdx={bdx}, bdy={bdy}",
        )
    plt.xlabel("Matrix size")
    plt.ylabel("Bandwidth (GB/s)")
    plt.title("tri_inv: rec_unroll vs col_sweep")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(str(path).replace(".csv", ".png"))


def bench(path: str = "benchmark_data/tri_inv_rec_unroll_vs_col_sweep.csv"):
    rows = [
        [
            "algo",
            "dtype",
            "matrix_size",
            "block_dim_x",
            "block_dim_y",
            "time_ms",
            "bandwidth_gbps",
        ]
    ]
    dtype, dtype_name = torch.float16, "fp16"
    for algo_name, (make_input, algo) in ALGOS.items():
        for n in (16, 32, 64, 128):
            for block_dim_x, block_dim_y in ((20, 4), (20, 16)):
                U = random_triu_matrix(n, block_dim_x, block_dim_y).to(dtype)
                U = make_input(U).npu()
                ms = do_bench(lambda inp=U, fn=algo: fn(inp), unit="ms")
                n_el = U.numel()
                # I/O: Input and output have 16 bits, so (2 + 2) bytes per element
                gbps = (2 + 2) * n_el / (ms / 1e3) / 1e9
                print(
                    f"{algo_name}, {dtype_name}, N={n}, bdx={block_dim_x}, "
                    f"bdy={block_dim_y}, {ms:.3f} ms, {gbps:.3f} GB/s"
                )
                rows.append(
                    [algo_name, dtype_name, n, block_dim_x, block_dim_y, ms, gbps]
                )

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)
    print(output)
    plot_csv(output)


if __name__ == "__main__":
    bench()
