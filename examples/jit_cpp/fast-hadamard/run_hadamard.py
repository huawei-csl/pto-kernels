import math

import torch
import torch_npu  # noqa

from jit_util_hadamard import jit_compile


def hadamard_ref_inplace(x):
    """Reference FHT matching the kernel's TGATHER(P0101/P1010) + TADD/TSUB layout.

    Keeps the same dtype (half) and device as input so that rounding
    behavior matches the kernel exactly — the original test_hadamard_pto.py
    does the same by comparing two half-precision NPU results.
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


# Test configs matching test_hadamard_pto.py
TEST_BATCHES = [1, 7, 22, 65]
TEST_NS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
TEST_SEEDS = [0, 1]


def test_correctness(hadamard_func):
    """Run correctness tests across (batch, N, seed) configs."""
    device = "npu"
    dtype = torch.float16

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
                x = torch.randn(batch, n, device=device, dtype=dtype)

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


def benchmark(hadamard_func, warmup=20, repeats=100):
    """Benchmark across (batch, N) configs, print duration and bandwidth."""
    device = "npu"
    dtype = torch.float16

    batches = [1, 5, 8, 10, 16, 20, 32, 40, 64, 128, 256, 512, 1024]
    ns = TEST_NS

    print("=" * 60)
    print("BENCHMARK")
    print("=" * 60)
    header = f"{'batch':>6s}  {'N':>6s}  {'duration_us':>12s}  {'bandwidth_gbs':>14s}"
    print(header)
    print("-" * len(header))

    csv_lines = ["batch,N,duration_us,bandwidth_gbs"]

    for batch in batches:
        for n in ns:
            log2_n = int(math.log2(n))
            x = torch.randn(batch, n, device=device, dtype=dtype)

            # Warmup
            for _ in range(warmup):
                hadamard_func(x, batch, n, log2_n)
            torch.npu.synchronize()

            # Timed runs
            start_events = [
                torch.npu.Event(enable_timing=True) for _ in range(repeats)
            ]
            end_events = [
                torch.npu.Event(enable_timing=True) for _ in range(repeats)
            ]

            for i in range(repeats):
                start_events[i].record()
                hadamard_func(x, batch, n, log2_n)
                end_events[i].record()

            torch.npu.synchronize()

            times_ms = [
                s.elapsed_time(e) for s, e in zip(start_events, end_events)
            ]
            dur_us = sum(times_ms) / len(times_ms) * 1000.0  # ms -> us

            # Bandwidth: read + write = 2 * batch * n * sizeof(half)
            data_bytes = 2 * batch * n * 2  # 2 bytes per half
            bw_gbs = (data_bytes / 1e9) / (dur_us / 1e6) if dur_us > 0 else 0.0

            print(f"{batch:>6d}  {n:>6d}  {dur_us:>12.3f}  {bw_gbs:>14.3f}")
            csv_lines.append(f"{batch},{n},{dur_us},{bw_gbs}")

    # Write CSV
    csv_path = "fht_pto.csv"
    with open(csv_path, "w") as f:
        f.write("\n".join(csv_lines) + "\n")
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    torch.npu.set_device("npu")

    print("Compiling fast_hadamard_pto-isa.cpp ...")
    hadamard_func = jit_compile("fast_hadamard_pto-isa.cpp")
    print()

    test_correctness(hadamard_func)
    benchmark(hadamard_func)
