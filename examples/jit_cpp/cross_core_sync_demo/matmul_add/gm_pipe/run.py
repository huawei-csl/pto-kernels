#!/usr/bin/env python3
"""
Correctness tests for matmul_add/gm_pipe kernels.

matmul_add_c2v  (gm_pipe):  C = A @ B + D   — all float16, half FIFO slot
add_matmul_v2c  (gm_pipe):  C = (A + B) @ D — all float16, half FIFO slot

gm_pipe uses explicit TSTORE/TLOAD on GlobalTensor slot views, plus
raw ffts_cross_core_sync/wait_flag_dev for signaling — same as raw_flag
semantics but with FIFO_DEPTH=2 double-buffer slot cycling.

NOTE on FFTS signal accumulation:
  FIFO protocols leave FIFO_DEPTH "pipeline fill" signals in FFTS counters
  after each run.  Use a fresh fifo buffer PER CALL to avoid accumulation.

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
