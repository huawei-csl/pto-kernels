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
sys.path.insert(0, str(HERE.parents[1]))
from pto_demo_utils import (  # noqa: E402
    assert_close,
    compile_kernel,
    configure_torch_npu,
    cube_core_count,
    run_repeated,
    stream_ptr,
    tensor_ptr,
    vector_core_count,
)


def run_add(device: str, n: int) -> None:
    lib = ctypes.CDLL(str(compile_kernel(HERE / "compile.sh", "add")))
    lib.call_add.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    lib.call_add.restype = None

    x = torch.randn(n, device=device, dtype=torch.float16)
    z = torch.randn(n, device=device, dtype=torch.float16)
    y = torch.empty_like(x)
    block_dim = vector_core_count(device)
    stream = stream_ptr()
    run_repeated(
        lambda: lib.call_add(block_dim, stream, tensor_ptr(y), tensor_ptr(x), tensor_ptr(z), n)
    )
    assert_close(y, x + z)
    print(f"PASS add n={n} block_dim={block_dim}")


def run_matmul(device: str, m: int) -> None:
    if m % 128 != 0:
        raise ValueError("m must be a multiple of 128")
    lib = ctypes.CDLL(str(compile_kernel(HERE / "compile.sh", "matmul")))
    lib.call_matmul.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    lib.call_matmul.restype = None

    torch.manual_seed(0)
    a = torch.randn(m, 128, device=device, dtype=torch.float16)
    b = torch.randn(128, 128, device=device, dtype=torch.float16)
    c = torch.empty(m, 128, device=device, dtype=torch.float16)
    block_dim = min(cube_core_count(device), max(1, m // 128))
    stream = stream_ptr()
    run_repeated(
        lambda: lib.call_matmul(block_dim, stream, tensor_ptr(a), tensor_ptr(b), tensor_ptr(c), m)
    )
    assert_close(c, torch.matmul(a, b))
    print(f"PASS simple_matmul m={m} block_dim={block_dim}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", choices=("add", "matmul", "all"), default="all")
    parser.add_argument("--device", default=os.environ.get("NPU_DEVICE", "npu:0"))
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--m", type=int, default=128)
    args = parser.parse_args()

    configure_torch_npu()
    torch.npu.set_device(args.device)
    if args.kernel in ("add", "all"):
        run_add(args.device, args.n)
    if args.kernel in ("matmul", "all"):
        run_matmul(args.device, args.m)


if __name__ == "__main__":
    main()
