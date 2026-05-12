#!/usr/bin/env python3
"""
Correctness tests and bandwidth benchmark for matmul_add/pushpop kernels.

matmul_add_c2v  (pushpop):  C = A @ B + D
  TPUSH<C2VPipe, TileL0C, TILE_UP_DOWN>(pipe, c_l0)  on Cube
  TPOP<C2VPipe, VecTileFloat, TILE_UP_DOWN>(pipe, c_ub_float)  on Vec
  Float32 slot (AccTile::DType=float); D:f32, C:f32.

add_matmul_v2c  (pushpop):  C = (A + B) @ D
  TPUSH<V2CPipe, TileVecUB, TILE_UP_DOWN>(pipe, a_ub)  on Vec
  TPOP<V2CPipe, TileL1, TILE_UP_DOWN>(pipe, ab_l1)  on Cube
  Half slot; all float16.

NOTE on multi-round TPUSH/TPOP:
  The TileData TPUSH/TPOP API manages a FIFO with tileIndex shared between Vec
  sub-blocks.  With 2 sub-blocks, tileIndex advances by 2 per logical round,
  de-syncing producer and consumer slot indices for num_rounds > 1.
  Both kernels are therefore scoped to num_rounds=1 for correctness.
  For multi-round workloads, use the gm_pipe variant.

Bandwidth benchmark scope:
  Correctness is only guaranteed at num_rounds=1 (batch = BLOCK_DIM * T = 3072).
  The benchmark reports peak throughput at that batch size, which is kernel-launch-
  overhead dominated and not representative of the algorithm's peak memory bandwidth.
  For a like-for-like comparison to raw_flag / gm_pipe at large batches, see those
  variants which support up to 64 rounds.

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


def test_matmul_add_c2v(kernel) -> None:
    print("=" * 60)
    print("matmul_add_c2v  pushpop  (C = A @ B + D)")
    print("  TPUSH Cube(AccTile<float>) → TPOP Vec(VecTile<float>)")
    print("  Float32 slot; D:f32, C:f32, A/B:f16")
    print("=" * 60)

    wave_rows = BLOCK_DIM * TILE_SIZE
    passed = failed = 0

    # Test with num_rounds=1 (single-round scope, matching unit-test coverage).
    # For multi-round use cases, see gm_pipe variant.
    for seed in range(5):
        batch = wave_rows  # num_rounds=1
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
                print(f"  FAIL seed={seed}: {e}")

    status = "OK" if failed == 0 else f"FAILED ({failed}/{passed+failed})"
    print(f"Correctness (num_rounds=1): {passed}/{passed+failed} passed — {status}\n")
    if failed:
        sys.exit(1)


def test_add_matmul_v2c(kernel) -> None:
    print("=" * 60)
    print("add_matmul_v2c  pushpop  (C = (A + B) @ D)")
    print("  TPUSH Vec(VecTile<half>) → TPOP Cube(TileL1<half>)")
    print("  Half slot; all float16")
    print("=" * 60)

    wave_rows = BLOCK_DIM * TILE_SIZE
    passed = failed = 0

    for seed in range(5):
        batch = wave_rows  # num_rounds=1
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

    status = "OK" if failed == 0 else f"FAILED ({failed}/{passed+failed})"
    print(f"Correctness (num_rounds=1): {passed}/{passed+failed} passed — {status}\n")
    if failed:
        sys.exit(1)



# ── Bandwidth benchmarks ───────────────────────────────────────────────────────
# Both kernels are valid only at num_rounds=1 due to the TileData tileIndex
# sharing issue (see module docstring).  The benchmark therefore runs at
# batch = BLOCK_DIM * TILE_SIZE (one wave, 3072 rows) only.
#
# At this small batch the effective HBM bandwidth is dominated by kernel-launch
# overhead; the number is NOT comparable to the large-batch figures for raw_flag
# or gm_pipe.  It is shown here purely for completeness.

def _benchmark_c2v(kernel, warmup: int = 10, repeats: int = 100) -> None:
    print("=" * 60)
    print("BENCHMARK  matmul_add_c2v  pushpop  (C = A @ B + D)")
    print(f"  num_rounds=1 only (tileIndex API limitation)")
    print(f"  warmup={warmup}  repeats={repeats}")
    print("=" * 60)

    wave_rows = BLOCK_DIM * TILE_SIZE
    batch = wave_rows  # num_rounds = 1

    A = torch.randn(batch, TILE_SIZE, dtype=torch.float16, device=_DEVICE)
    B = torch.randn(TILE_SIZE, TILE_SIZE, dtype=torch.float16, device=_DEVICE)
    D = torch.randn(batch, TILE_SIZE, dtype=torch.float32, device=_DEVICE)
    C = torch.zeros(batch, TILE_SIZE, dtype=torch.float32, device=_DEVICE)

    # Pre-allocate fresh fifo per call to avoid TPipe state accumulation
    n_calls = warmup + repeats
    fifos = [torch.zeros(BLOCK_DIM * C2V_FIFO_ELEMS_PER_CORE,
                         dtype=torch.float32, device=_DEVICE)
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
    # A(f16) + B(f16) + D(f32) + C(f32): mixed dtype — raw bytes accessed
    bytes_total = (batch * TILE_SIZE * 2        # A fp16
                   + TILE_SIZE * TILE_SIZE * 2   # B fp16
                   + batch * TILE_SIZE * 4       # D fp32
                   + batch * TILE_SIZE * 4)      # C fp32
    bw_gbs = bytes_total / dur_us * 1e-3
    print(f"  batch={batch}  dur={dur_us:.2f} µs  bw={bw_gbs:.1f} GB/s")
    print(f"  (overhead-dominated at this small batch; see gm_pipe for large-batch numbers)\n")


def _benchmark_v2c(kernel, warmup: int = 10, repeats: int = 100) -> None:
    print("=" * 60)
    print("BENCHMARK  add_matmul_v2c  pushpop  (C = (A + B) @ D)")
    print(f"  num_rounds=1 only (tileIndex API limitation)")
    print(f"  warmup={warmup}  repeats={repeats}")
    print("=" * 60)

    wave_rows = BLOCK_DIM * TILE_SIZE
    batch = wave_rows  # num_rounds = 1

    A = torch.randn(batch, TILE_SIZE, dtype=torch.float16, device=_DEVICE)
    B = torch.randn(batch, TILE_SIZE, dtype=torch.float16, device=_DEVICE)
    D = torch.randn(TILE_SIZE, TILE_SIZE, dtype=torch.float16, device=_DEVICE)
    C = torch.zeros(batch, TILE_SIZE, dtype=torch.float16, device=_DEVICE)

    n_calls = warmup + repeats
    fifos = [torch.zeros(BLOCK_DIM * V2C_FIFO_ELEMS_PER_CORE,
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
    # All fp16
    bytes_total = (batch * TILE_SIZE * 3 + TILE_SIZE * TILE_SIZE) * 2
    bw_gbs = bytes_total / dur_us * 1e-3
    print(f"  batch={batch}  dur={dur_us:.2f} µs  bw={bw_gbs:.1f} GB/s")
    print(f"  (overhead-dominated at this small batch; see gm_pipe for large-batch numbers)\n")


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

    _benchmark_c2v(c2v)
    _benchmark_v2c(v2c)
