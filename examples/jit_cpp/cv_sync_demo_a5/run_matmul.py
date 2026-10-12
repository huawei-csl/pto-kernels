#!/usr/bin/env python3
"""Correctness and bandwidth tests for direct A5 matmul/add kernels."""

from __future__ import annotations

import os
import sys

import torch
import torch_npu  # noqa: F401

os.environ.setdefault("NPU_DEVICE", "npu:0")
DEVICE = os.environ["NPU_DEVICE"]
torch.npu.set_device(DEVICE)

from jit_util import BLOCK_DIM, TILE_SIZE, CvSyncKernels  # noqa: E402

DTYPE = torch.float16
KW = dict(dtype=DTYPE, device=DEVICE)
RTOL = 1e-3
ATOL = 1e-2
WARMUP = int(os.environ.get("WARMUP", "10"))
REPEATS = int(os.environ.get("REPEATS", "30"))


def make_batch(num_rounds: int) -> int:
    return num_rounds * BLOCK_DIM * TILE_SIZE


def benchmark_bytes(batch: int) -> int:
    return (batch * TILE_SIZE * 3 + TILE_SIZE * TILE_SIZE) * 2


def benchmark_bytes_c2v(batch: int) -> int:
    return batch * TILE_SIZE * (2 + 4 + 4) + TILE_SIZE * TILE_SIZE * 2


def assert_close(name: str, got: torch.Tensor, expected: torch.Tensor) -> None:
    try:
        torch.testing.assert_close(got, expected, rtol=RTOL, atol=ATOL)
    except AssertionError as exc:
        print(f"{name} FAILED: {exc}")
        sys.exit(1)


def test_matmul_add_c2v(kernels: CvSyncKernels) -> None:
    passed = 0
    for seed in range(3):
        for rounds in range(1, 11):
            batch = make_batch(rounds)
            torch.manual_seed(seed)
            A = torch.randn(batch, TILE_SIZE, **KW)
            B = torch.randn(TILE_SIZE, TILE_SIZE, **KW)
            D = torch.randn(batch, TILE_SIZE, dtype=torch.float32, device=DEVICE)
            C = torch.empty(batch, TILE_SIZE, dtype=torch.float32, device=DEVICE)
            kernels.matmul_add_c2v(A, B, C, D)
            torch.npu.synchronize()
            expected = A @ B + D
            assert_close(f"matmul_add_c2v seed={seed} rounds={rounds}", C, expected)
            passed += 1
    print(f"matmul_add_c2v correctness: {passed}/30 passed")


def test_add_matmul_v2c(kernels: CvSyncKernels) -> None:
    passed = 0
    for seed in range(3):
        for rounds in range(1, 11):
            batch = make_batch(rounds)
            torch.manual_seed(seed)
            A = torch.randn(batch, TILE_SIZE, **KW)
            B = torch.randn(batch, TILE_SIZE, **KW)
            D = torch.randn(TILE_SIZE, TILE_SIZE, **KW)
            C = torch.empty_like(A)
            kernels.add_matmul_v2c(A, B, C, D)
            torch.npu.synchronize()
            expected = ((A + B) @ D).to(DTYPE)
            assert_close(f"add_matmul_v2c seed={seed} rounds={rounds}", C, expected)
            passed += 1
    print(f"add_matmul_v2c correctness: {passed}/30 passed")


def time_repeated(fn, *args) -> float:
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(REPEATS):
        fn(*args)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / REPEATS * 1e3


def bench_matmul_add_c2v(kernels: CvSyncKernels) -> list[tuple[int, int, float, float]]:
    print("=" * 72)
    print("matmul_add_c2v direct A5: C = A @ B + D")
    print("=" * 72)
    print(f"{'batch':>10}  {'rounds':>6}  {'dur_us':>10}  {'bw_GB/s':>10}")
    records = []
    for rounds in [1, 2, 4, 8, 16, 32, 64]:
        batch = make_batch(rounds)
        torch.manual_seed(0)
        A = torch.randn(batch, TILE_SIZE, **KW)
        B = torch.randn(TILE_SIZE, TILE_SIZE, **KW)
        D = torch.randn(batch, TILE_SIZE, dtype=torch.float32, device=DEVICE)
        C = torch.empty(batch, TILE_SIZE, dtype=torch.float32, device=DEVICE)
        for _ in range(WARMUP):
            kernels.matmul_add_c2v(A, B, C, D)
        torch.npu.synchronize()
        dur_us = time_repeated(kernels.matmul_add_c2v, A, B, C, D)
        bw = benchmark_bytes_c2v(batch) / dur_us * 1e-3
        print(f"{batch:>10d}  {rounds:>6d}  {dur_us:>10.2f}  {bw:>10.1f}")
        records.append((batch, rounds, dur_us, bw))
    print()
    return records


def bench_add_matmul_v2c(kernels: CvSyncKernels) -> list[tuple[int, int, float, float]]:
    print("=" * 72)
    print("add_matmul_v2c direct A5: C = (A + B) @ D")
    print("=" * 72)
    print(f"{'batch':>10}  {'rounds':>6}  {'dur_us':>10}  {'bw_GB/s':>10}")
    records = []
    for rounds in [1, 2, 4, 8, 16, 32, 64]:
        batch = make_batch(rounds)
        torch.manual_seed(0)
        A = torch.randn(batch, TILE_SIZE, **KW)
        B = torch.randn(batch, TILE_SIZE, **KW)
        D = torch.randn(TILE_SIZE, TILE_SIZE, **KW)
        C = torch.empty_like(A)
        for _ in range(WARMUP):
            kernels.add_matmul_v2c(A, B, C, D)
        torch.npu.synchronize()
        dur_us = time_repeated(kernels.add_matmul_v2c, A, B, C, D)
        bw = benchmark_bytes(batch) / dur_us * 1e-3
        print(f"{batch:>10d}  {rounds:>6d}  {dur_us:>10.2f}  {bw:>10.1f}")
        records.append((batch, rounds, dur_us, bw))
    print()
    return records


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print(f"BLOCK_DIM (Cube cores): {BLOCK_DIM}")
    kernels = CvSyncKernels(verbose=True)
    test_matmul_add_c2v(kernels)
    test_add_matmul_v2c(kernels)
    c2v = bench_matmul_add_c2v(kernels)
    v2c = bench_add_matmul_v2c(kernels)
    print(f"Peak matmul_add_c2v bandwidth: {max(r[3] for r in c2v):.1f} GB/s")
    print(f"Peak add_matmul_v2c bandwidth: {max(r[3] for r in v2c):.1f} GB/s")

