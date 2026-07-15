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
from pto_demo_utils import (
    assert_close,
    compile_kernel,
    configure_torch_npu,
    run_repeated,
    stream_ptr,
    tensor_ptr,
)  # noqa: E402


def run_add(device: str) -> None:
    lib = ctypes.CDLL(str(compile_kernel(HERE / "compile.sh", "add")))
    lib.call_static_add.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.call_static_add.restype = None

    torch.manual_seed(0)
    x = torch.randn(64, 64, device=device, dtype=torch.float16)
    z = torch.randn(64, 64, device=device, dtype=torch.float16)
    out = torch.empty_like(x)
    run_repeated(
        lambda: lib.call_static_add(
            1, stream_ptr(), tensor_ptr(out), tensor_ptr(x), tensor_ptr(z)
        )
    )
    assert_close(out, x + z)
    print("PASS static_single_core/a2a3 add shape=(64,64)")


def run_matmul(device: str) -> None:
    lib = ctypes.CDLL(str(compile_kernel(HERE / "compile.sh", "matmul")))
    lib.call_matmul.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    torch.manual_seed(1)
    a = torch.randn(128, 128, device=device, dtype=torch.float16)
    b = torch.randn(128, 128, device=device, dtype=torch.float16)
    c = torch.empty(128, 128, device=device, dtype=torch.float16)
    run_repeated(
        lambda: lib.call_matmul(
            1, stream_ptr(), tensor_ptr(a), tensor_ptr(b), tensor_ptr(c), 128
        )
    )
    assert_close(c, torch.matmul(a, b))
    print("PASS static_single_core/a2a3 matmul shape=(128,128)x(128,128)")


def run_mix(device: str) -> None:
    lib = ctypes.CDLL(str(compile_kernel(HERE / "compile.sh", "matmul_add")))
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
    block_dim = 1
    batch = 128
    torch.manual_seed(2)
    a = torch.randn(batch, 128, device=device, dtype=torch.float16)
    b = torch.randn(128, 128, device=device, dtype=torch.float16)
    d = torch.randn(batch, 128, device=device, dtype=torch.float16)
    c = torch.empty_like(a)
    ws = torch.empty(128, 128, device=device, dtype=torch.float16)
    run_repeated(
        lambda: lib.call(
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
    print(f"PASS static_single_core/a2a3 matmul_add rounds=1 batch={batch}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kernel", choices=("add", "matmul", "matmul_add", "all"), default="all"
    )
    args = parser.parse_args()
    device = os.environ.get("NPU_DEVICE", "npu:0")
    configure_torch_npu()
    torch.npu.set_device(device)
    if args.kernel in ("add", "all"):
        run_add(device)
    if args.kernel in ("matmul", "all"):
        run_matmul(device)
    if args.kernel in ("matmul_add", "all"):
        run_mix(device)


if __name__ == "__main__":
    main()
