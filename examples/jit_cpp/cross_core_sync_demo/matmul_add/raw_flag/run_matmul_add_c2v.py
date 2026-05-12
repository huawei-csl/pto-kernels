#!/usr/bin/env python3
"""
Correctness test and bandwidth benchmark for matmul_add_c2v kernel.

Algorithm: C = A @ B + D
  A : [batch, T]     fp16
  B : [T, T]         fp16   (constant weight)
  C : [batch, T]     fp16   (output)
  D : [batch, T]     fp16
  workspace: [num_cores * T, T]  fp16  (C2V communication buffer)
  where T = TILE_SIZE = 128

batch must be a multiple of (num_cores * T).

Usage:
    python run_matmul_add_c2v.py
"""
from __future__ import annotations

import os
import sys

import torch
import torch_npu  # noqa: F401

# ── Device setup ──────────────────────────────────────────────────────────────
_DEVICE = os.environ.get("NPU_DEVICE", "npu:7")
torch.npu.set_device(_DEVICE)
print(f"Using device: {_DEVICE}")

# Inject the device for jit_util's BLOCK_DIM query before importing it
os.environ["NPU_DEVICE"] = _DEVICE

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jit_util_matmul_add_c2v import compile_and_load, BLOCK_DIM  # noqa: E402

TILE_SIZE = 128
DTYPE = torch.float16
COMMON_KWARGS = dict(dtype=DTYPE, device=_DEVICE)

RTOL = 1e-3
ATOL = 1e-5


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_workspace() -> torch.Tensor:
    return torch.empty(BLOCK_DIM * TILE_SIZE, TILE_SIZE, **COMMON_KWARGS)


def run_kernel(kernel, A, B, D):
    """Allocate output C and workspace, launch kernel, return C."""
    C = torch.zeros_like(A)
    ws = make_workspace()
    kernel(A, B, C, D, ws)
    torch.npu.synchronize()
    return C


def ref(A, B, D):
    return (A @ B + D).to(DTYPE)


# ── Correctness tests ─────────────────────────────────────────────────────────

def test_correctness(kernel) -> None:
    print("=" * 60)
    print("CORRECTNESS TESTS")
    print("=" * 60)

    passed = failed = 0
    for seed in range(3):          # run each shape 3 times (indeterminate sync bugs show up)
        for num_rounds in range(1, 11):
            batch = num_rounds * BLOCK_DIM * TILE_SIZE
            torch.manual_seed(seed)
            A = torch.randn(batch, TILE_SIZE, **COMMON_KWARGS)
            B = torch.randn(TILE_SIZE, TILE_SIZE, **COMMON_KWARGS)
            D = torch.randn(batch, TILE_SIZE, **COMMON_KWARGS)

            C_kernel = run_kernel(kernel, A, B, D)
            C_ref    = ref(A, B, D)

            try:
                torch.testing.assert_close(C_kernel, C_ref, rtol=RTOL, atol=ATOL)
                passed += 1
            except AssertionError as e:
                failed += 1
                if failed <= 5:
                    print(f"  FAIL  seed={seed} num_rounds={num_rounds} batch={batch}: {e}")

    total = passed + failed
    status = "OK" if failed == 0 else f"FAILED ({failed}/{total})"
    print(f"\nCorrectness: {passed}/{total} passed — {status}\n")
    if failed:
        sys.exit(1)


# ── Benchmark ─────────────────────────────────────────────────────────────────

def benchmark(kernel, warmup: int = 10, repeats: int = 30) -> None:
    print("=" * 60)
    print(f"BENCHMARK  (warmup={warmup}, repeats={repeats})")
    print("=" * 60)
    header = f"{'batch':>10}  {'rounds':>6}  {'dur_us':>10}  {'bw_GB/s':>10}"
    print(header)
    print("-" * len(header))

    records = []
    for num_rounds in [1, 2, 4, 8, 16, 32, 64]:
        batch = num_rounds * BLOCK_DIM * TILE_SIZE
        torch.manual_seed(0)
        A = torch.randn(batch, TILE_SIZE, **COMMON_KWARGS)
        B = torch.randn(TILE_SIZE, TILE_SIZE, **COMMON_KWARGS)
        D = torch.randn(batch, TILE_SIZE, **COMMON_KWARGS)
        C = torch.zeros_like(A)
        ws = make_workspace()

        # Warm-up
        for _ in range(warmup):
            kernel(A, B, C, D, ws)
        torch.npu.synchronize()

        # Timed loop
        start = torch.npu.Event(enable_timing=True)
        end   = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            kernel(A, B, C, D, ws)
        end.record()
        end.synchronize()

        dur_us = start.elapsed_time(end) / repeats * 1e3  # ms → µs

        # Bytes accessed: read A + read B + read D + write C
        # (workspace is on-chip between Cube/Vec, so not counted as GM traffic)
        bytes_total = (batch * TILE_SIZE * 3 + TILE_SIZE * TILE_SIZE) * 2  # fp16 = 2 B
        bw_gbs = bytes_total / dur_us * 1e-3  # bytes/µs = GB/s

        print(f"{batch:>10d}  {num_rounds:>6d}  {dur_us:>10.2f}  {bw_gbs:>10.2f}")
        records.append(dict(batch=batch, num_rounds=num_rounds,
                            dur_us=dur_us, bw_gbs=bw_gbs))

    peak_bw = max(r["bw_gbs"] for r in records)
    print(f"\nPeak bandwidth: {peak_bw:.1f} GB/s  "
          f"(910B2 HBM roofline ≈ 1500 GB/s)")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"BLOCK_DIM (num Cube cores): {BLOCK_DIM}\n")

    kernel = compile_and_load(verbose=True)
    print()

    test_correctness(kernel)
    benchmark(kernel)
