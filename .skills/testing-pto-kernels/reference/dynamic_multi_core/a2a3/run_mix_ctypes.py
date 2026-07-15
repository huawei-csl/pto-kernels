#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import os
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401

HERE = Path(__file__).resolve().parent
TILE = 128

sys.path.insert(0, str(HERE.parents[1]))
from pto_demo_utils import (
    assert_close,
    compile_kernel,
    configure_torch_npu,
    cube_core_count,
    run_repeated,
    stream_ptr,
    tensor_ptr,
)  # noqa: E402


def load(kind: str):
    lib = ctypes.CDLL(str(compile_kernel(HERE / "compile.sh", kind)))
    lib.call.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
    ]
    lib.call.restype = None
    return lib


def workspace(block_dim: int, device: str) -> torch.Tensor:
    return torch.empty(block_dim * TILE, TILE, device=device, dtype=torch.float16)


def run_c2v(device: str, rounds: int) -> None:
    block_dim = cube_core_count(device)
    batch = rounds * block_dim * TILE
    torch.manual_seed(0)
    a = torch.randn(batch, TILE, device=device, dtype=torch.float16)
    b = torch.randn(TILE, TILE, device=device, dtype=torch.float16)
    d = torch.randn(batch, TILE, device=device, dtype=torch.float16)
    c = torch.empty_like(a)
    ws = workspace(block_dim, device)
    run_repeated(
        lambda: load("matmul_add").call(
            block_dim,
            stream_ptr(),
            tensor_ptr(a),
            tensor_ptr(b),
            tensor_ptr(c),
            tensor_ptr(d),
            tensor_ptr(ws),
            batch,
        )
    )
    assert_close(c, (a @ b + d).to(torch.float16))
    print(f"PASS A2A3 matmul_add_c2v rounds={rounds} batch={batch}")


def run_v2c(device: str, rounds: int) -> None:
    block_dim = cube_core_count(device)
    batch = rounds * block_dim * TILE
    torch.manual_seed(1)
    a = torch.randn(batch, TILE, device=device, dtype=torch.float16)
    b = torch.randn(batch, TILE, device=device, dtype=torch.float16)
    d = torch.randn(TILE, TILE, device=device, dtype=torch.float16)
    c = torch.empty_like(a)
    ws = workspace(block_dim, device)
    run_repeated(
        lambda: load("add_matmul").call(
            block_dim,
            stream_ptr(),
            tensor_ptr(a),
            tensor_ptr(b),
            tensor_ptr(c),
            tensor_ptr(d),
            tensor_ptr(ws),
            batch,
        )
    )
    assert_close(c, ((a + b) @ d).to(torch.float16))
    print(f"PASS A2A3 add_matmul_v2c rounds={rounds} batch={batch}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kernel", choices=("matmul_add", "add_matmul", "all"), default="all"
    )
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--device", default=os.environ.get("NPU_DEVICE", "npu:0"))
    args = parser.parse_args()
    configure_torch_npu()
    torch.npu.set_device(args.device)
    if args.kernel in ("matmul_add", "all"):
        run_c2v(args.device, args.rounds)
    if args.kernel in ("add_matmul", "all"):
        run_v2c(args.device, args.rounds)


if __name__ == "__main__":
    main()
