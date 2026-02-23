import os
import csv

import matplotlib.pyplot as plt
import torch
import torch_npu  # noqa

from jit_util_block_rotate import jit_compile

M_TILE = 128
N = 128
K = 128

DEVICE = "npu:7"
DTYPE = torch.float16

# Correctness tests: match ascendc-to-pto-isa repo coverage
# 1280 M values = 128, 256, ..., 128*1280
TEST_M_STEP = 128
TEST_M_COUNT = 320 * 4  # 1280 tests
TEST_SEEDS = [0]
TEST_ATOL = 1e-3

# Benchmark: match ascendc-to-pto-isa bench parameters
BENCH_M_STEP = 128 * 8  # 1024
BENCH_M_MAX = 150000
BENCH_MS = list(range(BENCH_M_STEP, BENCH_M_MAX + 1, BENCH_M_STEP))


def test_correctness(block_rotate_func):
    """Run correctness tests: C_kernel vs C_ref = A @ B, using mean_diff < atol."""
    print("=" * 60)
    print("CORRECTNESS TESTS")
    print("=" * 60)

    M_list = list(range(TEST_M_STEP, TEST_M_STEP * TEST_M_COUNT + 1, TEST_M_STEP))

    passed = 0
    failed = 0
    total = 0
    for seed in TEST_SEEDS:
        for m in M_list:
            total += 1
            torch.manual_seed(seed)
            a = torch.randn(m, K, device=DEVICE, dtype=DTYPE)
            b = torch.randn(K, N, device=DEVICE, dtype=DTYPE)
            c = torch.zeros(m, N, device=DEVICE, dtype=DTYPE)

            block_rotate_func(a, b, c, m)
            torch.npu.synchronize()

            c_ref = torch.matmul(a, b)

            if torch.equal(c, c_ref):
                passed += 1
            else:
                failed += 1
                if failed <= 20:  # limit output for large test suites
                    print(f"  FAIL  seed={seed} M={m:>6d}  c != c_ref")

            if total % 200 == 0:
                print(
                    f"  ... {total}/{len(M_list) * len(TEST_SEEDS)} tests done ({passed} passed)"
                )

    print(f"\n{passed}/{total} tests passed", end="")
    if failed > 0:
        print(f" ({failed} FAILED)", end="")
    print(".\n")


def _median(vals):
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2


def benchmark(
    block_rotate_func,
    warmup=5,
    repeats=20,
    n_loops=5,
    output_dir="./perf_data_double_buffer/",
):
    """Benchmark block rotation across M values.

    Runs n_loops independent timing loops (each with fresh matrices) and
    takes the median duration to eliminate outlier spikes.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"BENCHMARK  (n_loops={n_loops}, repeats={repeats})")
    print(f"{'=' * 60}")
    header = f"{'M':>8s}  {'dur_us':>10s}  {'bw_gbs':>10s}  {'add_us':>10s}  {'mm_us':>10s}  {'add_bw':>10s}  {'mm_bw':>10s}"
    print(header)
    print("-" * len(header))

    records = []
    allocated = warmup + repeats

    for m in BENCH_MS:
        kernel_samples = []
        add_samples = []
        mm_samples = []

        for _loop in range(n_loops):
            a_list = [
                torch.randn(m, K, device=DEVICE, dtype=DTYPE) for _ in range(allocated)
            ]
            b = torch.randn(K, N, device=DEVICE, dtype=DTYPE)
            c_list = [
                torch.zeros(m, N, device=DEVICE, dtype=DTYPE) for _ in range(allocated)
            ]

            # Warmup
            for i in range(warmup):
                block_rotate_func(a_list[i], b, c_list[i], m)
            torch.npu.synchronize()

            # --- Kernel benchmark ---
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            for i in range(repeats):
                block_rotate_func(a_list[warmup + i], b, c_list[warmup + i], m)
            end.record()
            torch.npu.synchronize()
            kernel_samples.append(start.elapsed_time(end) / repeats * 1e3)

            # --- Add baseline ---
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            for i in range(repeats):
                a_list[warmup + i].add_(1.0)
            end.record()
            torch.npu.synchronize()
            add_samples.append(start.elapsed_time(end) / repeats * 1e3)

            # --- torch.matmul baseline ---
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            for i in range(repeats):
                torch.matmul(a_list[warmup + i], b)
            end.record()
            torch.npu.synchronize()
            mm_samples.append(start.elapsed_time(end) / repeats * 1e3)

        dur_us = _median(kernel_samples)
        add_us = _median(add_samples)
        mm_us = _median(mm_samples)

        # Bandwidth: read A(m*K*2) + read B(K*N*2) + write C(m*N*2)
        data_bytes = (m * K + K * N + m * N) * 2
        bw_gbs = (data_bytes / dur_us) * 1e-3 if dur_us > 0 else 0.0
        add_bw = (data_bytes / add_us) * 1e-3 if add_us > 0 else 0.0
        mm_bw = (data_bytes / mm_us) * 1e-3 if mm_us > 0 else 0.0

        print(
            f"{m:>8d}  {dur_us:>10.2f}  {bw_gbs:>10.2f}  {add_us:>10.2f}  {mm_us:>10.2f}  {add_bw:>10.2f}  {mm_bw:>10.2f}"
        )
        records.append(
            {
                "M": m,
                "duration_us": dur_us,
                "bandwidth_gbs": bw_gbs,
                "add_duration_us": add_us,
                "torchmm_duration_us": mm_us,
                "add_bandwidth_gbs": add_bw,
                "torchmm_bandwidth_gbs": mm_bw,
            }
        )

    csv_path = os.path.join(output_dir, "block_rotate_fp16_double_buffer.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "M",
            "duration_us",
            "bandwidth_gbs",
            "add_duration_us",
            "torchmm_duration_us",
            "add_bandwidth_gbs",
            "torchmm_bandwidth_gbs",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"\nSaved to {csv_path}")

    return records


def plot_results(records, output_dir="./perf_data_double_buffer/"):
    """Generate bandwidth and duration plots from benchmark data."""
    os.makedirs(output_dir, exist_ok=True)

    ms = [r["M"] for r in records]
    dur = [r["duration_us"] for r in records]
    bw = [r["bandwidth_gbs"] for r in records]
    add_dur = [r["add_duration_us"] for r in records]
    mm_dur = [r["torchmm_duration_us"] for r in records]
    add_bw = [r["add_bandwidth_gbs"] for r in records]
    mm_bw = [r["torchmm_bandwidth_gbs"] for r in records]

    COLOR_KERNEL = "#dc2626"
    COLOR_ADD = "#16a34a"
    COLOR_MM = "#9333ea"

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Left: Duration vs M ---
    ax = axes[0]
    ax.plot(
        ms,
        add_dur,
        "-",
        color=COLOR_ADD,
        label="Add (baseline)",
        linewidth=1.5,
        alpha=0.7,
    )
    ax.plot(
        ms, mm_dur, "-", color=COLOR_MM, label="torch.matmul", linewidth=1.5, alpha=0.7
    )
    ax.plot(
        ms,
        dur,
        "o-",
        color=COLOR_KERNEL,
        label="PTO-ISA block_rotate (double buffer)",
        linewidth=2,
        markersize=2,
    )
    ax.set_xlabel("M (rows)", fontsize=12)
    ax.set_ylabel("Duration (us)", fontsize=12)
    ax.set_title("Kernel Duration vs M", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Right: Bandwidth vs M ---
    ax = axes[1]
    ax.plot(
        ms,
        add_bw,
        "-",
        color=COLOR_ADD,
        label="Add (baseline)",
        linewidth=1.5,
        alpha=0.7,
    )
    ax.plot(
        ms, mm_bw, "-", color=COLOR_MM, label="torch.matmul", linewidth=1.5, alpha=0.7
    )
    ax.plot(
        ms,
        bw,
        "o-",
        color=COLOR_KERNEL,
        label="PTO-ISA block_rotate (double buffer)",
        linewidth=2,
        markersize=2,
    )
    ax.set_xlabel("M (rows)", fontsize=12)
    ax.set_ylabel("Bandwidth (GB/s)", fontsize=12)
    ax.set_title("Effective Bandwidth vs M", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Block Rotate FP16 Double Buffer (PTO-ISA): Performance",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    plot_path = os.path.join(output_dir, "block_rotate_fp16_double_buffer_perf.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {plot_path}")

    # --- Summary stats ---
    peak_bw = max(bw)
    peak_m = ms[bw.index(peak_bw)]
    print(f"\nPeak bandwidth: {peak_bw:.1f} GB/s at M={peak_m}")
    print(f"Duration range: {min(dur):.1f} - {max(dur):.1f} us")


if __name__ == "__main__":
    torch.npu.set_device(DEVICE)

    print("Compiling block_rotate_fp16_double_buffer.cpp ...")
    block_rotate_func = jit_compile("block_rotate_fp16_double_buffer.cpp")
    print()

    test_correctness(block_rotate_func)
    records = benchmark(block_rotate_func)
    plot_results(records)
