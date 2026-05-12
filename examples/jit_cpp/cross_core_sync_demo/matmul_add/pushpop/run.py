#!/usr/bin/env python3
"""
Correctness tests for matmul_add/pushpop kernels.

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
  The test is scoped to num_rounds=1 to match the unit-test coverage level.
  For multi-round correctness with 2 Vec sub-blocks, see the gm_pipe variant
  which uses explicit TSTORE/TLOAD with manual slot cycling.

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
