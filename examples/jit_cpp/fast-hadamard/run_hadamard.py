import math
import os
import csv
import matplotlib.pyplot as plt

import torch
import torch_npu  # noqa

from jit_util_hadamard import jit_compile

# Test configs matching test_hadamard_pto.py
TEST_BATCHES = [1, 7, 22, 65]
TEST_NS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
TEST_SEEDS = [0, 1]

BENCH_BATCHES = [1, 5, 8, 10, 16, 20, 32, 40, 64, 128, 256, 512, 1024]
BENCH_BLOCK_DIMS = [20, 24]

DEVICE = "npu"
DTYPE = torch.float16


def hadamard_ref_inplace(x):
    """Reference FHT matching the kernel's TGATHER(P0101/P1010) + TADD/TSUB layout.

    Keeps the same dtype (half) and device as input so that rounding
    behavior matches the kernel exactly.
    """
    x = x.clone()
    n = x.shape[-1]
    n_half = n // 2
    log2_n = int(math.log2(n))
    for _ in range(log2_n):
        even = x[..., 0::2].clone()
        odd = x[..., 1::2].clone()
        x[..., :n_half] = even + odd
        x[..., n_half:] = even - odd
    return x


def test_correctness(hadamard_func):
    """Run correctness tests across (batch, N, seed) configs."""
    print("=" * 60)
    print("CORRECTNESS TESTS")
    print("=" * 60)

    passed = 0
    total = 0
    for seed in TEST_SEEDS:
        for batch in TEST_BATCHES:
            for n in TEST_NS:
                total += 1
                torch.manual_seed(seed)
                log2_n = int(math.log2(n))
                x = torch.randn(batch, n, device=DEVICE, dtype=DTYPE)

                y_ref = hadamard_ref_inplace(x)

                hadamard_func(x, batch, n, log2_n)
                torch.npu.synchronize()

                if torch.allclose(x, y_ref):
                    passed += 1
                    print(f"  PASS  seed={seed} batch={batch:>4d}, N={n:>5d}")
                else:
                    maxdiff = (x - y_ref).abs().max().item()
                    print(
                        f"  FAIL  seed={seed} batch={batch:>4d}, N={n:>5d}"
                        f"  max_diff={maxdiff:.6f}"
                    )

    print(f"\n{passed}/{total} tests passed.\n")


def benchmark(hadamard_func, warmup=2, repeats=20, output_dir="./perf_data/"):
    """Benchmark across (batch, N, block_dim) configs.

    Uses separate input tensors per run to avoid L2 cache reuse,
    and a single timing-event pair averaged over all runs.
    """
    os.makedirs(output_dir, exist_ok=True)

    for block_dim in BENCH_BLOCK_DIMS:
        print(f"\n{'=' * 60}")
        print(f"BENCHMARK (BLOCK_DIM={block_dim})")
        print(f"{'=' * 60}")
        header = (
            f"{'batch':>6s}  {'N':>6s}"
            f"  {'duration_us':>12s}  {'bandwidth_gbs':>14s}"
        )
        print(header)
        print("-" * len(header))

        records = []

        for batch in BENCH_BATCHES:
            for n in TEST_NS:
                log2_n = int(math.log2(n))
                allocated = warmup + repeats

                # Separate GM tensors to avoid L2 cache reuse
                x_list = [
                    torch.randn(batch, n, device=DEVICE, dtype=DTYPE)
                    for _ in range(allocated)
                ]

                # Warmup
                for i in range(warmup):
                    hadamard_func(x_list[i], batch, n, log2_n, block_dim=block_dim)
                torch.npu.synchronize()

                # Timed runs — single event pair, average over repeats
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)

                start.record()
                for i in range(repeats):
                    hadamard_func(
                        x_list[warmup + i],
                        batch,
                        n,
                        log2_n,
                        block_dim=block_dim,
                    )
                end.record()
                torch.npu.synchronize()

                duration_ms = start.elapsed_time(end) / repeats
                dur_us = duration_ms * 1e3

                # Bandwidth: read + write = 2 * batch * n * sizeof(half)
                data_bytes = 2 * batch * n * 2
                bw_gbs = (data_bytes / 1e9) / (dur_us / 1e6) if dur_us > 0 else 0.0

                print(f"{batch:>6d}  {n:>6d}" f"  {dur_us:>12.2f}  {bw_gbs:>14.2f}")
                records.append(f"{batch},{n},{dur_us:.4f},{bw_gbs:.4f}")

        csv_path = os.path.join(output_dir, f"fht_pto_bd{block_dim}.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("batch,N,duration_us,bandwidth_gbs\n")
            f.write("\n".join(records) + "\n")
        print(f"\nSaved to {csv_path}")


def plot_bandwidth(input_dir="./perf_data/", output_path="bw_vs_shape.png"):
    """Generate bandwidth plot from benchmark CSVs."""

    fig, axes = plt.subplots(1, len(BENCH_BLOCK_DIMS), figsize=(14, 6), sharey=True)
    if len(BENCH_BLOCK_DIMS) == 1:
        axes = [axes]

    for ax, block_dim in zip(axes, BENCH_BLOCK_DIMS):
        csv_path = os.path.join(input_dir, f"fht_pto_bd{block_dim}.csv")
        if not os.path.exists(csv_path):
            ax.set_title(f"BLOCK_DIM={block_dim} (no data)")
            continue

        # Parse CSV: hidden_dim -> {batch: bw}
        data = {}
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                batch = int(row["batch"])
                n = int(row["N"])
                bw = float(row["bandwidth_gbs"])
                data.setdefault(n, {})[batch] = bw

        for idx, hidden_dim in enumerate(sorted(data.keys())):
            batches = sorted(data[hidden_dim].keys())
            bws = [data[hidden_dim][b] for b in batches]

            if idx < 10:
                marker = "o"
            else:
                last_markers = ["s", "^", "D"]
                marker = last_markers[idx - 10]

            ax.plot(
                batches,
                bws,
                marker=marker,
                markersize=4,
                label=f"hidden_dim={hidden_dim}",
            )

        ax.set_xscale("log", base=2)
        ax.set_xticks(BENCH_BATCHES)
        ax.set_xticklabels([str(b) for b in BENCH_BATCHES], rotation=45, fontsize=7)
        ax.set_xlabel("batch")
        ax.set_title(f"BLOCK_DIM={block_dim}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)

    axes[0].set_ylabel("Bandwidth (GB/s)")
    fig.suptitle("Fast Hadamard PTO-ISA: Bandwidth vs Shape")
    fig.tight_layout()
    fig.savefig(input_dir + output_path, dpi=150)
    print(f"\nPlot saved to {input_dir+output_path}")


if __name__ == "__main__":
    torch.npu.set_device("npu")

    print("Compiling fast_hadamard_pto-isa.cpp ...")
    hadamard_func = jit_compile("fast_hadamard_pto-isa.cpp")
    print()

    test_correctness(hadamard_func)
    benchmark(hadamard_func)
    plot_bandwidth()
