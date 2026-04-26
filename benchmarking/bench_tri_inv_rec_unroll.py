"""
Performance benchmark for triangular inverse recursive unroll kernel.
"""

import matplotlib.pyplot as plt
import torch
import csv
from pathlib import Path
from pto_kernels import pto_tri_inv_rec_unroll
from utils import do_bench


def random_triu_matrix(n: int, block_dim_x: int, block_dim_y: int, scale: float = 0.1):
    U = scale * torch.triu(torch.rand((block_dim_x, block_dim_y, n, n)), diagonal=1)
    return U


def plot_csv(path):
    series = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["dtype"], row["block_dim_x"], row["block_dim_y"])
            series.setdefault(key, [[], []])
            series[key][0].append(int(row["matrix_size"]))
            series[key][1].append(float(row["bandwidth_gbps"]))

    for (dt, bdx, bdy), (xs, ys) in sorted(series.items()):
        order = sorted(range(len(xs)), key=xs.__getitem__)
        plt.plot(
            [xs[i] for i in order],
            [ys[i] for i in order],
            marker="o",
            label=f"{dt}, bdx={bdx}, bdy={bdy}",
        )
    plt.xlabel("Matrix size")
    plt.ylabel("Bandwidth (GB/s)")
    plt.title("tri_inv_rec_unroll: bf16 vs fp16")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(str(path).replace(".csv", ".png"))


def bench(path: str = "benchmark_data/tri_inv_rec_unroll.csv"):
    rows = [
        [
            "dtype",
            "matrix_size",
            "block_dim_x",
            "block_dim_y",
            "time_ms",
            "bandwidth_gbps",
        ]
    ]
    for dtype in (torch.float16,):
        for n in (16, 32, 64, 128):
            for block_dim_x, block_dim_y in (
                (20, 4),
                (20, 8),
                (20, 16),
                (32, 16),
                (32, 32),
            ):
                U = random_triu_matrix(n, block_dim_x, block_dim_y).to(dtype).npu()
                ms = do_bench(
                    lambda inp=U: pto_tri_inv_rec_unroll(inp, is_bsnd_format=False),
                    unit="ms",
                )
                n_el = U.numel()
                # I/O: Input has 16 bits and output is fp32, so (2 + 4) bytes per element
                gbps = (2 + 4) * n_el / (ms / 1e3) / 1e9
                dtype_name = "fp16" if dtype == torch.float16 else "bf16"
                print(
                    f"{dtype_name}, N={n}, bdx={block_dim_x}, bdy={block_dim_y}, "
                    f"{ms:.3f} ms, {gbps:.3f} GB/s"
                )
                rows.append([dtype_name, n, block_dim_x, block_dim_y, ms, gbps])

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)
    print(output)
    plot_csv(output)


if __name__ == "__main__":
    bench()
