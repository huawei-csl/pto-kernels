import matplotlib.pyplot as plt
import numpy as np
import torch
import csv
from pathlib import Path
from pto_kernels import pto_tri_inv
from utils import do_bench


def rand_np_tril(batch_size: int, n: int, dtype: np.dtype):
    "Returns a random unit lower triangular matrix of size n."
    A = np.random.rand(batch_size, n, n).astype(dtype)
    A = np.tril(A)
    for k in range(batch_size):
        np.fill_diagonal(A[k, :, :], 1.0)
    return A.astype(dtype)


def plot_csv(path):
    series = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["dtype"], row["batch_size"])
            series.setdefault(key, [[], []])
            series[key][0].append(int(row["matrix_size"]))
            series[key][1].append(float(row["bandwidth_gbps"]))

    for (dt, bs), (xs, ys) in sorted(series.items()):
        order = sorted(range(len(xs)), key=xs.__getitem__)
        plt.plot(
            [xs[i] for i in order],
            [ys[i] for i in order],
            marker="o",
            label=f"{dt}, bs={bs}",
        )
    plt.xlabel("Matrix size")
    plt.ylabel("Bandwidth (GB/s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(path).replace(".csv", ".png"))


def bench(path: str = "benchmark_data/tri_inv_col_sweep.csv"):
    rows = [["dtype", "matrix_size", "batch_size", "time_ms", "bandwidth_gbps"]]
    for dt in (np.float16, np.float32):
        for S in (16, 32, 64, 128):
            for bs in (256,):
                x = torch.from_numpy(rand_np_tril(bs, S, dt).transpose(0, 2, 1)).npu()
                ms = do_bench(lambda inp=x: pto_tri_inv(inp), unit="ms")
                n_el, size = x.numel(), x.element_size()
                gbps = 2 * n_el * size / (ms / 1e3) / 1e9
                print(f"{dt.__name__}, N={S}, bs={bs}, {ms:.3f} ms, {gbps:.3f} GB/s")
                rows.append([dt.__name__, S, bs, ms, gbps])

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)
    print(output)
    plot_csv(output)


if __name__ == "__main__":
    bench()
