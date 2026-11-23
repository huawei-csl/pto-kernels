#!/usr/bin/env python3
"""
Correctness tests and bandwidth benchmark for matmul_add/pushpop kernels.

matmul_add_c2v  (pushpop):  C = A @ B + D
  TPUSH<C2VPipe, TileL0C, TILE_UP_DOWN>(pipe, c_l0)  on Cube
  TPOP<C2VPipe, VecTile<float>, TILE_UP_DOWN>(pipe, c_ub_float)  on Vec
  Float32 slot (AccTile::DType=float); D:f32, C:f32.

add_matmul_v2c  (pushpop):  C = (A + B) @ D
  TPUSH<V2CPipe, TileVecUB, TILE_UP_DOWN>(pipe, a_ub)  on Vec
  TPOP<V2CPipe, TileL1, TILE_UP_DOWN>(pipe, ab_l1)  on Cube
  Half slot; all float16.

FIFO_DEPTH=1 workaround (see PTO_API_BUGS.md Bug 1):
  TPipe TileData TPUSH/TPOP with FIFO_DEPTH=2 and TILE_UP_DOWN is broken for
  num_rounds > 1: both Vec sub-blocks share a single tileIndex counter, so the
  FIFO slot selection drifts by 2× per logical round.
  Setting FIFO_DEPTH=1 forces SyncPeriod=1 (strict alternation, no double-buffer
  prefetch) which avoids the desync.  Multi-round benchmarking is now possible.

NOTE on fresh fifo per call:
  Reusing the same fifo_mem tensor across repeated calls accumulates TPipe
  head/tail state stored inside fifo_mem.  The benchmark pre-allocates one fresh
  fifo per call to prevent this.

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
    load_matmul_add_c2v, load_add_matmul_v2c,
    BLOCK_DIM, TILE_SIZE,
    C2V_FIFO_ELEMS_PER_CORE, V2C_FIFO_ELEMS_PER_CORE,
)

RTOL = 1e-3
ATOL = 1e-3


# ── Correctness tests ──────────────────────────────────────────────────────────

def test_matmul_add_c2v(kernel) -> None:
    print("=" * 60)
    print("matmul_add_c2v  pushpop  (C = A @ B + D)")
    print("  TPUSH Cube(AccTile<float>) → TPOP Vec(VecTile<float>)")
    print("  Float32 slot; D:f32, C:f32, A/B:f16  |  FIFO_DEPTH=1")
    print("=" * 60)

    wave_rows = BLOCK_DIM * TILE_SIZE
    passed = failed = 0

    for seed in range(3):
        for num_rounds in range(1, 9):
            batch = num_rounds * wave_rows
            torch.manual_seed(seed)
            A = torch.randn(batch, TILE_SIZE, dtype=torch.float16, device=_DEVICE)
            B = torch.randn(TILE_SIZE, TILE_SIZE, dtype=torch.float16, device=_DEVICE)
            D = torch.randn(batch, TILE_SIZE, dtype=torch.float32, device=_DEVICE)
            C = torch.zeros(batch, TILE_SIZE, dtype=torch.float32, device=_DEVICE)
            fifo = torch.zeros(BLOCK_DIM * C2V_FIFO_ELEMS_PER_CORE,
                               dtype=torch.float32, device=_DEVICE)

            kernel(A, B, C, D, fifo)
            torch.npu.synchronize()

            ref = (A.float() @ B.float()) + D
            try:
                torch.testing.assert_close(C, ref, rtol=RTOL, atol=ATOL)
                passed += 1
            except AssertionError as e:
                failed += 1
                if failed <= 3:
                    print(f"  FAIL seed={seed} rounds={num_rounds}: {e}")

    total  = passed + failed
    status = "OK" if failed == 0 else f"FAILED ({failed}/{total})"
    print(f"Correctness: {passed}/{total} passed — {status}\n")
    if failed:
        sys.exit(1)


def test_add_matmul_v2c(kernel) -> None:
    print("=" * 60)
    print("add_matmul_v2c  pushpop  (C = (A + B) @ D)")
    print("  TPUSH Vec(VecTile<half>) → TPOP Cube(TileL1<half>)")
    print("  Half slot; all float16  |  FIFO_DEPTH=2  (num_rounds=1 scope)")
    print("  V2C: FIFO_DEPTH=2 needed; tileIndex desync breaks rounds>1.")
    print("=" * 60)

    wave_rows = BLOCK_DIM * TILE_SIZE
    passed = failed = 0

    # V2C remains scoped to num_rounds=1: FIFO_DEPTH=2 is required so both Vec
    # sub-blocks can write without blocking at allocate(), but tileIndex desync
    # (both sub-blocks share a single counter) breaks rounds > 1.
    for seed in range(5):
        batch = wave_rows  # num_rounds = 1
        torch.manual_seed(seed)
        A = torch.randn(batch, TILE_SIZE, dtype=torch.float16, device=_DEVICE)
        B = torch.randn(batch, TILE_SIZE, dtype=torch.float16, device=_DEVICE)
        D = torch.randn(TILE_SIZE, TILE_SIZE, dtype=torch.float16, device=_DEVICE)
        C = torch.zeros(batch, TILE_SIZE, dtype=torch.float16, device=_DEVICE)
        fifo = torch.zeros(BLOCK_DIM * V2C_FIFO_ELEMS_PER_CORE,
                           dtype=torch.float16, device=_DEVICE)

        kernel(A, B, C, D, fifo)
        torch.npu.synchronize()

        ref = ((A + B) @ D).half()
        try:
            torch.testing.assert_close(C, ref, rtol=1e-3, atol=1e-5)
            passed += 1
        except AssertionError as e:
            failed += 1
            if failed <= 3:
                print(f"  FAIL seed={seed}: {e}")

    total  = passed + failed
    status = "OK" if failed == 0 else f"FAILED ({failed}/{total})"
    print(f"Correctness (num_rounds=1): {passed}/{total} passed — {status}\n")
    if failed:
        sys.exit(1)


# ── Bandwidth benchmarks ───────────────────────────────────────────────────────

def _benchmark(kernel, name: str, fifo_dtype, fifo_elems_per_core: int,
               make_tensors, warmup: int = 10, repeats: int = 30,
               num_rounds_list: list | None = None) -> None:
    if num_rounds_list is None:
        num_rounds_list = [1, 2, 4, 8, 16, 32, 64]
    print("=" * 60)
    print(f"BENCHMARK  {name}  pushpop  (FIFO_DEPTH=1)")
    print(f"  warmup={warmup}  repeats={repeats}")
    print("=" * 60)

    wave_rows = BLOCK_DIM * TILE_SIZE
    hdr = f"{'batch':>10}  {'rounds':>6}  {'dur_us':>10}  {'bw_GB/s':>10}"
    print(hdr)
    print("-" * len(hdr))

    records = []
    for num_rounds in num_rounds_list:
        batch = num_rounds * wave_rows
        tensors = make_tensors(batch)
        A, B, D = tensors["A"], tensors["B"], tensors["D"]
        C = torch.zeros_like(tensors["C_ref"])

        # Pre-allocate fresh fifo per call — avoids TPipe head/tail accumulation
        n_calls = warmup + repeats
        fifos = [torch.zeros(BLOCK_DIM * fifo_elems_per_core,
                             dtype=fifo_dtype, device=_DEVICE)
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
        bytes_total = tensors["bytes"]
        bw_gbs = bytes_total / dur_us * 1e-3

        print(f"{batch:>10d}  {num_rounds:>6d}  {dur_us:>10.2f}  {bw_gbs:>10.2f}")
        records.append(dict(batch=batch, num_rounds=num_rounds,
                            dur_us=dur_us, bw_gbs=bw_gbs))

    peak_bw = max(r["bw_gbs"] for r in records)
    print(f"\nPeak bandwidth: {peak_bw:.1f} GB/s  "
          f"(910B2 HBM roofline ≈ 1500 GB/s)\n")


def benchmark_c2v(kernel) -> None:
    def make(batch):
        kw16 = dict(dtype=torch.float16, device=_DEVICE)
        kw32 = dict(dtype=torch.float32, device=_DEVICE)
        A = torch.randn(batch, TILE_SIZE, **kw16)
        B = torch.randn(TILE_SIZE, TILE_SIZE, **kw16)
        D = torch.randn(batch, TILE_SIZE, **kw32)
        C_ref = torch.zeros(batch, TILE_SIZE, **kw32)
        # bytes: A(f16) + B(f16) + D(f32) + C(f32)
        nb = (batch * TILE_SIZE * 2 + TILE_SIZE * TILE_SIZE * 2
              + batch * TILE_SIZE * 4 + batch * TILE_SIZE * 4)
        return dict(A=A, B=B, D=D, C_ref=C_ref, bytes=nb)

    _benchmark(kernel, "matmul_add_c2v  (C = A @ B + D)",
               torch.float32, C2V_FIFO_ELEMS_PER_CORE, make)


def benchmark_v2c(kernel) -> None:
    def make(batch):
        kw = dict(dtype=torch.float16, device=_DEVICE)
        A = torch.randn(batch, TILE_SIZE, **kw)
        B = torch.randn(batch, TILE_SIZE, **kw)
        D = torch.randn(TILE_SIZE, TILE_SIZE, **kw)
        C_ref = torch.zeros(batch, TILE_SIZE, **kw)
        # bytes: A + B + D + C  (all f16)
        nb = (batch * TILE_SIZE * 3 + TILE_SIZE * TILE_SIZE) * 2
        return dict(A=A, B=B, D=D, C_ref=C_ref, bytes=nb)

    _benchmark(kernel, "add_matmul_v2c  (C = (A + B) @ D)",
               torch.float16, V2C_FIFO_ELEMS_PER_CORE, make,
               num_rounds_list=[1])  # rounds=1 only; tileIndex desync breaks rounds>1


if __name__ == "__main__":
    print(f"BLOCK_DIM={BLOCK_DIM}\n")

    print("Compiling matmul_add_c2v ...")
    c2v = load_matmul_add_c2v(verbose=True)
    print()
    print("Compiling add_matmul_v2c ...")
    v2c = load_add_matmul_v2c(verbose=True)
    print()

    test_matmul_add_c2v(c2v)
    test_add_matmul_v2c(v2c)

    benchmark_c2v(c2v)
    benchmark_v2c(v2c)
