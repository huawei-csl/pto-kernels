#!/usr/bin/env python3
"""
Correctness tests and bandwidth benchmark for naive_separate kernels.

Two kernels:
  matmul_add_c2v : C = A @ B + D
  add_matmul_v2c : C = (A + B) @ D

Both are "two-stage, no pipeline" baselines:
  • Stage 1 (GEMM or Vec-add) completes ALL rounds before stage 2 starts.
  • One kernel launch covers both stages — faster than two separate launches
    (no second kernel launch overhead), but slower than pipelined variants
    (no round-level Cube↔Vec overlap).

Workspace sizing:
  workspace[batch, TILE_SIZE] fp16  — one slot per (core, round) pair.
  This is larger than the pipelined FIFO buffers (which use ≤ FIFO_DEPTH slots
  per core regardless of num_rounds).

Benchmark sections:
  1. Correctness test: compare against torch.matmul / torch.add reference.
  2. Torch baseline: measure time for torch.matmul then torch.add (two launches).
  3. Naive kernel benchmark: measure the naive single-launch kernel.
  4. Comparison table with pipelined variants (from prior benchmarks, for context).

Usage:
    python run.py
    NPU_DEVICE=npu:5 python run.py
"""
from __future__ import annotations

import os
import sys

import torch
import torch_npu  # noqa: F401

_DEVICE = os.environ.get("NPU_DEVICE", "npu:7")
torch.npu.set_device(_DEVICE)
print(f"Using device: {_DEVICE}")
os.environ["NPU_DEVICE"] = _DEVICE

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jit_util import (  # noqa: E402
    load_matmul_add_c2v,
    load_add_matmul_v2c,
    BLOCK_DIM,
    TILE_SIZE,
)

DTYPE = torch.float16
RTOL = 1e-3
ATOL = 1e-5
_KW = dict(dtype=DTYPE, device=_DEVICE)
_WARMUP = 10
_REPEATS = 30


# ── Workspace allocation ───────────────────────────────────────────────────────
def make_workspace(batch: int) -> torch.Tensor:
    """workspace[batch, TILE_SIZE] fp16 — one slot per (core, round)."""
    return torch.zeros(batch, TILE_SIZE, **_KW)


# ── Correctness ────────────────────────────────────────────────────────────────


def _run(kernel, A, B, D) -> torch.Tensor:
    C = torch.zeros_like(A)
    ws = make_workspace(A.shape[0])
    kernel(A, B, C, D, ws)
    torch.npu.synchronize()
    return C


def test_correctness(c2v_kernel, v2c_kernel) -> None:
    print("=" * 62)
    print("CORRECTNESS TESTS")
    print("=" * 62)
    wave_rows = BLOCK_DIM * TILE_SIZE

    for name, kernel, make_tensors, ref_fn in [
        (
            "matmul_add_c2v  (C = A @ B + D)",
            c2v_kernel,
            lambda batch: (
                torch.randn(batch, TILE_SIZE, **_KW),
                torch.randn(TILE_SIZE, TILE_SIZE, **_KW),
                torch.randn(batch, TILE_SIZE, **_KW),
            ),
            lambda A, B, D: (A @ B + D).to(DTYPE),
        ),
        (
            "add_matmul_v2c  (C = (A + B) @ D)",
            v2c_kernel,
            lambda batch: (
                torch.randn(batch, TILE_SIZE, **_KW),
                torch.randn(batch, TILE_SIZE, **_KW),
                torch.randn(TILE_SIZE, TILE_SIZE, **_KW),
            ),
            lambda A, B, D: ((A + B) @ D).to(DTYPE),
        ),
    ]:
        passed = failed = 0
        for seed in range(3):
            for num_rounds in range(1, 11):
                batch = num_rounds * wave_rows
                torch.manual_seed(seed)
                A, B, D = make_tensors(batch)
                C_kernel = _run(kernel, A, B, D)
                C_ref = ref_fn(A, B, D)
                try:
                    torch.testing.assert_close(C_kernel, C_ref, rtol=RTOL, atol=ATOL)
                    passed += 1
                except AssertionError as e:
                    failed += 1
                    if failed <= 5:
                        print(f"  FAIL  seed={seed} rounds={num_rounds}: {e}")
        total = passed + failed
        status = "OK" if failed == 0 else f"FAILED ({failed}/{total})"
        print(f"  {name}: {passed}/{total} passed — {status}")

    print()
    if failed:
        sys.exit(1)


# ── Benchmark helpers ──────────────────────────────────────────────────────────


def _time_kernel(fn, warmup: int = _WARMUP, repeats: int = _REPEATS) -> float:
    """Return average kernel duration in microseconds."""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats * 1e3  # ms → µs


def _bytes_c2v(batch: int) -> int:
    """GM bytes for matmul_add_c2v: read A + read B + read D + write C  (fp16)."""
    return (batch * TILE_SIZE * 3 + TILE_SIZE * TILE_SIZE) * 2


def _bytes_v2c(batch: int) -> int:
    """GM bytes for add_matmul_v2c: read A + read B + read D + write C  (fp16)."""
    return (batch * TILE_SIZE * 3 + TILE_SIZE * TILE_SIZE) * 2


# ── Benchmark: matmul_add_c2v ──────────────────────────────────────────────────


def benchmark_c2v(c2v_kernel) -> list[dict]:
    print("=" * 62)
    print("BENCHMARK  matmul_add_c2v  (C = A @ B + D)")
    print(f"  warmup={_WARMUP}  repeats={_REPEATS}")
    print("=" * 62)
    wave_rows = BLOCK_DIM * TILE_SIZE

    hdr = (
        f"{'batch':>10}  {'rounds':>6}  "
        f"{'naive_us':>10}  {'naive_GB/s':>12}  "
        f"{'torch_us':>10}  {'torch_GB/s':>12}  "
        f"{'speedup':>8}"
    )
    print(hdr)
    print("-" * len(hdr))

    records = []
    for num_rounds in [1, 2, 4, 8, 16, 32, 64]:
        batch = num_rounds * wave_rows
        torch.manual_seed(0)
        A = torch.randn(batch, TILE_SIZE, **_KW)
        B = torch.randn(TILE_SIZE, TILE_SIZE, **_KW)
        D = torch.randn(batch, TILE_SIZE, **_KW)
        C = torch.zeros_like(A)
        ws = make_workspace(batch)

        naive_us = _time_kernel(lambda: c2v_kernel(A, B, C, D, ws))
        torch_us = _time_kernel(lambda: (torch.mm(A, B) + D))  # two-launch baseline

        nbytes = _bytes_c2v(batch)
        naive_bw = nbytes / naive_us * 1e-3
        torch_bw = nbytes / torch_us * 1e-3
        speedup = torch_us / naive_us

        print(
            f"{batch:>10d}  {num_rounds:>6d}  "
            f"{naive_us:>10.2f}  {naive_bw:>12.2f}  "
            f"{torch_us:>10.2f}  {torch_bw:>12.2f}  "
            f"{speedup:>8.2f}x"
        )
        records.append(
            dict(
                batch=batch,
                num_rounds=num_rounds,
                naive_us=naive_us,
                naive_bw=naive_bw,
                torch_us=torch_us,
                torch_bw=torch_bw,
            )
        )

    peak = max(r["naive_bw"] for r in records)
    print(
        f"\nPeak naive bandwidth: {peak:.1f} GB/s  (910B2 HBM roofline ≈ 1500 GB/s)\n"
    )
    return records


# ── Benchmark: add_matmul_v2c ──────────────────────────────────────────────────


def benchmark_v2c(v2c_kernel) -> list[dict]:
    print("=" * 62)
    print("BENCHMARK  add_matmul_v2c  (C = (A + B) @ D)")
    print(f"  warmup={_WARMUP}  repeats={_REPEATS}")
    print("=" * 62)
    wave_rows = BLOCK_DIM * TILE_SIZE

    hdr = (
        f"{'batch':>10}  {'rounds':>6}  "
        f"{'naive_us':>10}  {'naive_GB/s':>12}  "
        f"{'torch_us':>10}  {'torch_GB/s':>12}  "
        f"{'speedup':>8}"
    )
    print(hdr)
    print("-" * len(hdr))

    records = []
    for num_rounds in [1, 2, 4, 8, 16, 32, 64]:
        batch = num_rounds * wave_rows
        torch.manual_seed(0)
        A = torch.randn(batch, TILE_SIZE, **_KW)
        B = torch.randn(batch, TILE_SIZE, **_KW)
        D = torch.randn(TILE_SIZE, TILE_SIZE, **_KW)
        C = torch.zeros_like(A)
        ws = make_workspace(batch)

        naive_us = _time_kernel(lambda: v2c_kernel(A, B, C, D, ws))
        torch_us = _time_kernel(lambda: ((A + B) @ D))  # two-op baseline

        nbytes = _bytes_v2c(batch)
        naive_bw = nbytes / naive_us * 1e-3
        torch_bw = nbytes / torch_us * 1e-3
        speedup = torch_us / naive_us

        print(
            f"{batch:>10d}  {num_rounds:>6d}  "
            f"{naive_us:>10.2f}  {naive_bw:>12.2f}  "
            f"{torch_us:>10.2f}  {torch_bw:>12.2f}  "
            f"{speedup:>8.2f}x"
        )
        records.append(
            dict(
                batch=batch,
                num_rounds=num_rounds,
                naive_us=naive_us,
                naive_bw=naive_bw,
                torch_us=torch_us,
                torch_bw=torch_bw,
            )
        )

    peak = max(r["naive_bw"] for r in records)
    print(
        f"\nPeak naive bandwidth: {peak:.1f} GB/s  (910B2 HBM roofline ≈ 1500 GB/s)\n"
    )
    return records


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"BLOCK_DIM (num Cube cores): {BLOCK_DIM}\n")

    print("Compiling naive_separate kernels ...")
    c2v = load_matmul_add_c2v(verbose=True)
    v2c = load_add_matmul_v2c(verbose=True)
    print()

    test_correctness(c2v, v2c)
    benchmark_c2v(c2v)
    benchmark_v2c(v2c)
