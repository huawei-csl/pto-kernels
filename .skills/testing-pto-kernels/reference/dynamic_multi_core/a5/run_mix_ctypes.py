#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa: F401

HERE = Path(__file__).resolve().parent
TILE = 128

sys.path.insert(0, str(HERE.parents[1]))
from pto_demo_utils import assert_close, compile_kernel, configure_torch_npu, run_repeated, stream_ptr, tensor_ptr  # noqa: E402


def npu_from_cpu(arr: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(arr).to(device)


def load():
    lib = ctypes.CDLL(str(compile_kernel(HERE / "compile.sh", "mix")))
    for name in ("cv_matmul_add_c2v", "cv_add_matmul_v2c"):
        fn = getattr(lib, name)
        fn.argtypes = [
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_void_p,
        ]
        fn.restype = None
    return lib


def run_c2v(lib, device: str, block_dim: int, rounds: int) -> None:
    batch = rounds * block_dim * TILE
    rng = np.random.default_rng(2)
    a_cpu = rng.standard_normal((batch, TILE)).astype(np.float16)
    b_cpu = rng.standard_normal((TILE, TILE)).astype(np.float16)
    d_cpu = rng.standard_normal((batch, TILE)).astype(np.float32)
    a = npu_from_cpu(a_cpu, device)
    b = npu_from_cpu(b_cpu, device)
    d = npu_from_cpu(d_cpu, device)
    c = npu_from_cpu(np.zeros((batch, TILE), dtype=np.float32), device)
    run_repeated(
        lambda: lib.cv_matmul_add_c2v(block_dim, tensor_ptr(a), tensor_ptr(b), tensor_ptr(c), tensor_ptr(d), batch, stream_ptr())
    )
    ref = torch.from_numpy(a_cpu.astype(np.float32) @ b_cpu.astype(np.float32) + d_cpu)
    assert_close(c.cpu(), ref)
    print(f"PASS A5 matmul_add_c2v rounds={rounds} batch={batch} block_dim={block_dim}")


def run_v2c(lib, device: str, block_dim: int, rounds: int) -> None:
    batch = rounds * block_dim * TILE
    rng = np.random.default_rng(3)
    a_cpu = rng.standard_normal((batch, TILE)).astype(np.float16)
    b_cpu = rng.standard_normal((batch, TILE)).astype(np.float16)
    d_cpu = rng.standard_normal((TILE, TILE)).astype(np.float16)
    a = npu_from_cpu(a_cpu, device)
    b = npu_from_cpu(b_cpu, device)
    d = npu_from_cpu(d_cpu, device)
    c = npu_from_cpu(np.zeros((batch, TILE), dtype=np.float16), device)
    run_repeated(
        lambda: lib.cv_add_matmul_v2c(block_dim, tensor_ptr(a), tensor_ptr(b), tensor_ptr(c), tensor_ptr(d), batch, stream_ptr())
    )
    ref = torch.from_numpy(((a_cpu + b_cpu).astype(np.float32) @ d_cpu.astype(np.float32)).astype(np.float16))
    assert_close(c.cpu(), ref)
    print(f"PASS A5 add_matmul_v2c rounds={rounds} batch={batch} block_dim={block_dim}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", choices=("matmul_add", "add_matmul", "all"), default="all")
    parser.add_argument("--device", default=os.environ.get("NPU_DEVICE", "npu:0"))
    parser.add_argument("--block-dim", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=1)
    args = parser.parse_args()
    configure_torch_npu(simulator_safe=True)
    torch.npu.set_device(args.device)
    lib = load()
    if args.kernel in ("matmul_add", "all"):
        run_c2v(lib, args.device, args.block_dim, args.rounds)
    if args.kernel in ("add_matmul", "all"):
        run_v2c(lib, args.device, args.block_dim, args.rounds)


if __name__ == "__main__":
    main()
