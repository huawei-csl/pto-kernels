#!/usr/bin/env python3
"""Smoke and bandwidth sweep for direct A5 Cube/Vector stream kernels."""

from __future__ import annotations

import os

import torch
import torch_npu  # noqa: F401

os.environ.setdefault("NPU_DEVICE", "npu:0")
DEVICE = os.environ["NPU_DEVICE"]
torch.npu.set_device(DEVICE)

from jit_util import BLOCK_DIM, TILE_SIZE, CvSyncKernels  # noqa: E402

DTYPE = torch.float16
KW = dict(dtype=DTYPE, device=DEVICE)
WARMUP = int(os.environ.get("WARMUP", "5"))
REPEATS = int(os.environ.get("REPEATS", "20"))
ITERS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def direct_bytes(num_iters: int, element_size: int) -> int:
    return BLOCK_DIM * TILE_SIZE * TILE_SIZE * element_size * num_iters


def old_equiv_bytes(num_iters: int) -> int:
    return 2 * BLOCK_DIM * TILE_SIZE * TILE_SIZE * 2 * num_iters


def time_kernel(fn, *args, num_iters: int) -> float:
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(REPEATS):
        fn(*args, num_iters)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / REPEATS * 1e3


def run_c2v(kernels: CvSyncKernels) -> list[tuple[int, float, float, float]]:
    print("=" * 72)
    print("stream_c2v direct A5 (Cube L0C -> Vec UB)")
    print("=" * 72)
    print(f"{'num_iters':>10}  {'dur_us':>10}  {'direct_GB/s':>12}  {'old_equiv_GB/s':>15}")

    wave_rows = BLOCK_DIM * TILE_SIZE
    A = torch.randn(wave_rows, TILE_SIZE, **KW)
    B = torch.randn(TILE_SIZE, TILE_SIZE, **KW)

    kernels.stream_c2v(A, B, 4)
    torch.npu.synchronize()
    print("  smoke (num_iters=4): OK")

    records = []
    for num_iters in ITERS:
        for _ in range(WARMUP):
            kernels.stream_c2v(A, B, num_iters)
        torch.npu.synchronize()
        dur_us = time_kernel(kernels.stream_c2v, A, B, num_iters=num_iters)
        direct = direct_bytes(num_iters, 4) / dur_us * 1e-3
        equiv = old_equiv_bytes(num_iters) / dur_us * 1e-3
        print(f"{num_iters:>10d}  {dur_us:>10.2f}  {direct:>12.1f}  {equiv:>15.1f}")
        records.append((num_iters, dur_us, direct, equiv))
    print()
    return records


def run_v2c(kernels: CvSyncKernels) -> list[tuple[int, float, float, float]]:
    print("=" * 72)
    print("stream_v2c direct A5 (Vec UB -> Cube L1)")
    print("=" * 72)
    print(f"{'num_iters':>10}  {'dur_us':>10}  {'direct_GB/s':>12}  {'old_equiv_GB/s':>15}")

    wave_rows = BLOCK_DIM * TILE_SIZE
    records = []
    A_smoke = torch.randn(4 * wave_rows, TILE_SIZE, **KW)
    D_smoke = torch.randn(4 * wave_rows, TILE_SIZE, **KW)
    kernels.stream_v2c(A_smoke, D_smoke, 4)
    torch.npu.synchronize()
    print("  smoke (num_iters=4): OK")

    for num_iters in ITERS:
        total_rows = num_iters * wave_rows
        A = torch.randn(total_rows, TILE_SIZE, **KW)
        D = torch.randn(total_rows, TILE_SIZE, **KW)
        for _ in range(WARMUP):
            kernels.stream_v2c(A, D, num_iters)
        torch.npu.synchronize()
        dur_us = time_kernel(kernels.stream_v2c, A, D, num_iters=num_iters)
        direct = direct_bytes(num_iters, 2) / dur_us * 1e-3
        equiv = old_equiv_bytes(num_iters) / dur_us * 1e-3
        print(f"{num_iters:>10d}  {dur_us:>10.2f}  {direct:>12.1f}  {equiv:>15.1f}")
        records.append((num_iters, dur_us, direct, equiv))
    print()
    return records


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print(f"BLOCK_DIM (Cube cores): {BLOCK_DIM}")
    kernels = CvSyncKernels(verbose=True)
    c2v = run_c2v(kernels)
    v2c = run_v2c(kernels)
    print(f"Peak stream_c2v direct: {max(r[2] for r in c2v):.1f} GB/s")
    print(f"Peak stream_c2v old-equivalent: {max(r[3] for r in c2v):.1f} GB/s")
    print(f"Peak stream_v2c direct: {max(r[2] for r in v2c):.1f} GB/s")
    print(f"Peak stream_v2c old-equivalent: {max(r[3] for r in v2c):.1f} GB/s")

