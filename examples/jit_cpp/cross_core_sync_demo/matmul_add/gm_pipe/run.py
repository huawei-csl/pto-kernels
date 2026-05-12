#!/usr/bin/env python3
"""
Correctness tests and bandwidth benchmark for matmul_add/gm_pipe kernels.

matmul_add_c2v  (gm_pipe):  C = A @ B + D   — all float16, half FIFO slot
add_matmul_v2c  (gm_pipe):  C = (A + B) @ D — all float16, half FIFO slot

gm_pipe uses explicit TSTORE/TLOAD on GlobalTensor slot views, plus
raw ffts_cross_core_sync/wait_flag_dev for signaling — same as raw_flag
semantics but with FIFO_DEPTH=2 double-buffer slot cycling.

NOTE on FFTS signal accumulation:
  FIFO protocols leave FIFO_DEPTH "pipeline fill" signals in FFTS counters
  after each run.  Use a fresh fifo buffer PER CALL to avoid accumulation.

Usage:
    python run.py            # correctness + bandwidth
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
    load_matmul_add_c2v, load_add_matmul_v2c,
    BLOCK_DIM, TILE_SIZE, FIFO_ELEMS_PER_CORE,
)

RTOL = 1e-3
ATOL = 1e-5


def _test(kernel, name: str, ref_fn) -> None:
    print("=" * 60)
    print(f"{name}  gm_pipe")
    print("=" * 60)

    wave_rows = BLOCK_DIM * TILE_SIZE
    passed = failed = 0

    for num_rounds in [1, 4, 8]:
        batch = num_rounds * wave_rows
        torch.manual_seed(0)
        tensors = ref_fn(batch, TILE_SIZE, _DEVICE)
        A, B, D = tensors['A'], tensors['B'], tensors['D']
        C = torch.zeros(batch, TILE_SIZE, dtype=torch.float16, device=_DEVICE)

        # Fresh fifo per call: avoids FFTS counter accumulation across calls
        fifo = torch.zeros(BLOCK_DIM * FIFO_ELEMS_PER_CORE,
                           dtype=torch.float16, device=_DEVICE)
        kernel(A, B, C, D, fifo)
        torch.npu.synchronize()

        ref = tensors['ref']
        try:
            torch.testing.assert_close(C, ref, rtol=RTOL, atol=ATOL)
            passed += 1
        except AssertionError as e:
            failed += 1
            if failed <= 3:
                print(f"  FAIL rounds={num_rounds}: {e}")

    status = "OK" if failed == 0 else f"FAILED ({failed}/{passed+failed})"
    print(f"Correctness: {passed}/{passed+failed} passed — {status}\n")
    if failed:
        sys.exit(1)


def _c2v_tensors(batch, tile, device):
    kw = dict(dtype=torch.float16, device=device)
    A = torch.randn(batch, tile, **kw)
    B = torch.randn(tile, tile, **kw)
    D = torch.randn(batch, tile, **kw)
    return dict(A=A, B=B, D=D, ref=(A @ B + D).half())


def _v2c_tensors(batch, tile, device):
    kw = dict(dtype=torch.float16, device=device)
    A = torch.randn(batch, tile, **kw)
    B = torch.randn(batch, tile, **kw)
    D = torch.randn(tile, tile, **kw)
    return dict(A=A, B=B, D=D, ref=((A + B) @ D).half())


def _benchmark(kernel, name: str, make_tensors, warmup: int = 10,
               repeats: int = 30) -> None:
    print("=" * 60)
    print(f"BENCHMARK  {name}  gm_pipe")
    print(f"  warmup={warmup}  repeats={repeats}")
    print("=" * 60)

    wave_rows = BLOCK_DIM * TILE_SIZE
    hdr = f"{'batch':>10}  {'rounds':>6}  {'dur_us':>10}  {'bw_GB/s':>10}"
    print(hdr)
    print("-" * len(hdr))

    records = []
    for num_rounds in [1, 2, 4, 8, 16, 32, 64]:
        batch = num_rounds * wave_rows
        tensors = make_tensors(batch, TILE_SIZE, _DEVICE)
        A, B, D = tensors['A'], tensors['B'], tensors['D']
        C = torch.zeros(batch, TILE_SIZE, dtype=torch.float16, device=_DEVICE)

        # Pre-allocate a fresh fifo for every call so TPipe FIFO head/tail
        # pointers stored inside fifo_mem never accumulate across calls.
        # Allocation happens before the timing window — no overhead inside timer.
        n_calls = warmup + repeats
        fifos = [torch.zeros(BLOCK_DIM * FIFO_ELEMS_PER_CORE,
                             dtype=torch.float16, device=_DEVICE)
                 for _ in range(n_calls)]

        for i in range(warmup):
            kernel(A, B, C, D, fifos[i])
        torch.npu.synchronize()

        start = torch.npu.Event(enable_timing=True)
        end   = torch.npu.Event(enable_timing=True)
        start.record()
        for i in range(repeats):
            kernel(A, B, C, D, fifos[warmup + i])
        end.record()
        end.synchronize()

        dur_us = start.elapsed_time(end) / repeats * 1e3
        # Bytes: read A + read B/D + read D/B + write C  (all fp16 = 2 bytes)
        bytes_total = (batch * TILE_SIZE * 3 + TILE_SIZE * TILE_SIZE) * 2
        bw_gbs = bytes_total / dur_us * 1e-3

        print(f"{batch:>10d}  {num_rounds:>6d}  {dur_us:>10.2f}  {bw_gbs:>10.2f}")
        records.append(dict(batch=batch, num_rounds=num_rounds,
                            dur_us=dur_us, bw_gbs=bw_gbs))

    peak_bw = max(r["bw_gbs"] for r in records)
    print(f"\nPeak bandwidth: {peak_bw:.1f} GB/s  "
          f"(910B2 HBM roofline ≈ 1500 GB/s)\n")


if __name__ == "__main__":
    print(f"BLOCK_DIM={BLOCK_DIM}\n")

    print("Compiling matmul_add_c2v ...")
    c2v = load_matmul_add_c2v(verbose=True)
    print()
    print("Compiling add_matmul_v2c ...")
    v2c = load_add_matmul_v2c(verbose=True)
    print()

    _test(c2v, "matmul_add_c2v  (C = A @ B + D)",    _c2v_tensors)
    _test(v2c, "add_matmul_v2c  (C = (A + B) @ D)", _v2c_tensors)

    _benchmark(c2v, "matmul_add_c2v  (C = A @ B + D)",    _c2v_tensors)
    _benchmark(v2c, "add_matmul_v2c  (C = (A + B) @ D)", _v2c_tensors)
