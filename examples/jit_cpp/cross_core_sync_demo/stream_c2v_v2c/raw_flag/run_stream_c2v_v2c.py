#!/usr/bin/env python3
"""
Bandwidth benchmark for stream_c2v and stream_v2c kernels.

Both kernels measure the sustained throughput of the Cube↔Vector workspace
handshake path using `ffts_cross_core_sync` / `wait_flag_dev`.

Effective bandwidth definition (same for both directions):
  bw_eff = 2 × num_cores × T² × sizeof(fp16) × num_iters / time
           ↑ workspace write + workspace read (round-trip)

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
from jit_util_stream import load_stream_c2v, load_stream_v2c, BLOCK_DIM  # noqa: E402

TILE_SIZE  = 128
DTYPE      = torch.float16
KW         = dict(dtype=DTYPE, device=_DEVICE)

WARMUP  = 5
REPEATS = 20


def workspace_roundtrip_bytes(num_iters: int) -> int:
    """GM bytes transferred through workspace per kernel launch."""
    return 2 * BLOCK_DIM * TILE_SIZE * TILE_SIZE * 2 * num_iters  # ×2: write + read


def _time_kernel(fn, *args, num_iters: int) -> float:
    """Return median duration in µs for one call of fn(*args)."""
    start = torch.npu.Event(enable_timing=True)
    end   = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(REPEATS):
        fn(*args, num_iters)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / REPEATS * 1e3  # ms → µs


# ── stream_c2v ────────────────────────────────────────────────────────────────

def run_c2v(kernel) -> None:
    print("=" * 60)
    print("stream_c2v  (Cube L0C → workspace → Vec UB)")
    print("=" * 60)
    header = f"{'num_iters':>10}  {'dur_us':>10}  {'bw_GB/s':>10}"
    print(header)
    print("-" * len(header))

    wave_rows = BLOCK_DIM * TILE_SIZE
    A  = torch.randn(wave_rows, TILE_SIZE, **KW)
    B  = torch.randn(TILE_SIZE, TILE_SIZE, **KW)
    ws = torch.empty(wave_rows, TILE_SIZE, **KW)

    # Smoke check: run once with a few iterations, no crash = pass
    kernel(A, B, ws, 4)
    torch.npu.synchronize()
    print(f"  smoke (num_iters=4): OK")
    print()

    records = []
    for num_iters in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        for _ in range(WARMUP):
            kernel(A, B, ws, num_iters)
        torch.npu.synchronize()

        dur_us = _time_kernel(kernel, A, B, ws, num_iters=num_iters)
        bw_gbs = workspace_roundtrip_bytes(num_iters) / dur_us * 1e-3

        print(f"{num_iters:>10d}  {dur_us:>10.2f}  {bw_gbs:>10.1f}")
        records.append((num_iters, dur_us, bw_gbs))

    peak_bw = max(r[2] for r in records)
    peak_ni = max(records, key=lambda r: r[2])[0]
    print(f"\nPeak: {peak_bw:.1f} GB/s at num_iters={peak_ni}  "
          f"(910B2 HBM roofline ≈ 1500 GB/s)\n")


# ── stream_v2c ────────────────────────────────────────────────────────────────

def run_v2c(kernel) -> None:
    print("=" * 60)
    print("stream_v2c  (Vec UB → workspace → Cube L1)")
    print("=" * 60)
    header = f"{'num_iters':>10}  {'dur_us':>10}  {'bw_GB/s':>10}"
    print(header)
    print("-" * len(header))

    wave_rows = BLOCK_DIM * TILE_SIZE
    ws = torch.empty(wave_rows, TILE_SIZE, **KW)

    # Smoke check
    A_smoke = torch.randn(4 * wave_rows, TILE_SIZE, **KW)
    D_smoke = torch.randn(4 * wave_rows, TILE_SIZE, **KW)
    kernel(A_smoke, D_smoke, ws, 4)
    torch.npu.synchronize()
    print(f"  smoke (num_iters=4): OK")
    print()

    records = []
    for num_iters in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        total_rows = num_iters * wave_rows
        A = torch.randn(total_rows, TILE_SIZE, **KW)
        D = torch.randn(total_rows, TILE_SIZE, **KW)

        for _ in range(WARMUP):
            kernel(A, D, ws, num_iters)
        torch.npu.synchronize()

        dur_us = _time_kernel(kernel, A, D, ws, num_iters=num_iters)
        bw_gbs = workspace_roundtrip_bytes(num_iters) / dur_us * 1e-3

        print(f"{num_iters:>10d}  {dur_us:>10.2f}  {bw_gbs:>10.1f}")
        records.append((num_iters, dur_us, bw_gbs))

    peak_bw = max(r[2] for r in records)
    peak_ni = max(records, key=lambda r: r[2])[0]
    print(f"\nPeak: {peak_bw:.1f} GB/s at num_iters={peak_ni}  "
          f"(910B2 HBM roofline ≈ 1500 GB/s)\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"BLOCK_DIM (num Cube cores): {BLOCK_DIM}\n")

    print("Compiling stream_c2v ...")
    c2v = load_stream_c2v(verbose=True)
    print()

    print("Compiling stream_v2c ...")
    v2c = load_stream_v2c(verbose=True)
    print()

    run_c2v(c2v)
    run_v2c(v2c)
