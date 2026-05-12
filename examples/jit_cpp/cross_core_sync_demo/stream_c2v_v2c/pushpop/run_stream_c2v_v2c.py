#!/usr/bin/env python3
"""
Bandwidth benchmark for stream_c2v_v2c/pushpop kernels.

C2V variant: TPUSH AccTile<float> → float slot (64 KB/core/slot)
V2C variant: TPUSH VecTile<half>  → half  slot (32 KB/core/slot)

Effective bandwidth formula (matches raw_flag for comparison):
  bw = 2 × num_cores × T² × SlotElementSize × num_iters / time
  C2V: SlotElementSize = sizeof(float) = 4  (slot is 2× larger than raw_flag)
  V2C: SlotElementSize = sizeof(half)  = 2  (same as raw_flag)

Usage:
    python run_stream_c2v_v2c.py
    NPU_DEVICE=npu:5 python run_stream_c2v_v2c.py
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
from jit_util_stream import (  # noqa: E402
    load_stream_c2v, load_stream_v2c,
    BLOCK_DIM, TILE_SIZE, FIFO_DEPTH,
    C2V_FIFO_ELEMS_PER_CORE, V2C_FIFO_ELEMS_PER_CORE,
)

WARMUP  = 5
REPEATS = 20


def _time_kernel(fn, *args, num_iters: int) -> float:
    start = torch.npu.Event(enable_timing=True)
    end   = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(REPEATS):
        fn(*args, num_iters)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / REPEATS * 1e3  # ms → µs


def run_c2v(kernel) -> None:
    print("=" * 62)
    print("stream_c2v  pushpop  (Cube TPUSH AccTile<float> → Vec TPOP)")
    print("Slot: float32, 64 KB/core/slot  (2× raw_flag half slot)")
    print("=" * 62)
    header = f"{'num_iters':>10}  {'dur_us':>10}  {'bw_GB/s':>10}"
    print(header)
    print("-" * len(header))

    wave_rows = BLOCK_DIM * TILE_SIZE
    A  = torch.randn(wave_rows, TILE_SIZE, dtype=torch.float16, device=_DEVICE)
    B  = torch.randn(TILE_SIZE, TILE_SIZE, dtype=torch.float16, device=_DEVICE)
    # fifo_mem: float32, one entry per slot element
    fifo_mem = torch.zeros(BLOCK_DIM * C2V_FIFO_ELEMS_PER_CORE,
                           dtype=torch.float32, device=_DEVICE)

    kernel(A, B, fifo_mem, 4)
    torch.npu.synchronize()
    print("  smoke (num_iters=4): OK\n")

    for num_iters in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        for _ in range(WARMUP):
            kernel(A, B, fifo_mem, num_iters)
        torch.npu.synchronize()

        dur_us = _time_kernel(kernel, A, B, fifo_mem, num_iters=num_iters)
        # float slot: 4 bytes per element; ×2 for write+read round-trip
        bw_gbs = 2 * BLOCK_DIM * TILE_SIZE * TILE_SIZE * 4 * num_iters / dur_us * 1e-3
        print(f"{num_iters:>10d}  {dur_us:>10.2f}  {bw_gbs:>10.1f}")

    print()


def run_v2c(kernel) -> None:
    print("=" * 62)
    print("stream_v2c  pushpop  (Vec TPUSH VecTile<half> → Cube TPOP)")
    print("Slot: float16, 32 KB/core/slot  (same as raw_flag)")
    print("=" * 62)
    header = f"{'num_iters':>10}  {'dur_us':>10}  {'bw_GB/s':>10}"
    print(header)
    print("-" * len(header))

    wave_rows = BLOCK_DIM * TILE_SIZE
    fifo_mem  = torch.zeros(BLOCK_DIM * V2C_FIFO_ELEMS_PER_CORE,
                            dtype=torch.float16, device=_DEVICE)

    A_smoke = torch.randn(4 * wave_rows, TILE_SIZE, dtype=torch.float16, device=_DEVICE)
    D_smoke = torch.randn(4 * wave_rows, TILE_SIZE, dtype=torch.float16, device=_DEVICE)
    kernel(A_smoke, D_smoke, fifo_mem, 4)
    torch.npu.synchronize()
    print("  smoke (num_iters=4): OK\n")

    for num_iters in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        total_rows = num_iters * wave_rows
        A = torch.randn(total_rows, TILE_SIZE, dtype=torch.float16, device=_DEVICE)
        D = torch.randn(total_rows, TILE_SIZE, dtype=torch.float16, device=_DEVICE)

        for _ in range(WARMUP):
            kernel(A, D, fifo_mem, num_iters)
        torch.npu.synchronize()

        dur_us = _time_kernel(kernel, A, D, fifo_mem, num_iters=num_iters)
        bw_gbs = 2 * BLOCK_DIM * TILE_SIZE * TILE_SIZE * 2 * num_iters / dur_us * 1e-3
        print(f"{num_iters:>10d}  {dur_us:>10.2f}  {bw_gbs:>10.1f}")

    print()


if __name__ == "__main__":
    print(f"BLOCK_DIM={BLOCK_DIM}  FIFO_DEPTH={FIFO_DEPTH}\n")
    print("Compiling stream_c2v ...")
    c2v = load_stream_c2v(verbose=True)
    print()
    print("Compiling stream_v2c ...")
    v2c = load_stream_v2c(verbose=True)
    print()
    run_c2v(c2v)
    run_v2c(v2c)
