#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from pathlib import Path

import numpy as np
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


def npu_tensor(arr: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(arr).to(device)


def run_add(device: str) -> None:
    lib = ctypes.CDLL(str(compile_kernel(HERE / "compile.sh", "add")))
    lib.call_static_add.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    rng = np.random.default_rng(0)
    x_cpu = rng.standard_normal((64, 64)).astype(np.float16)
    z_cpu = rng.standard_normal((64, 64)).astype(np.float16)
    out = npu_tensor(np.zeros((64, 64), dtype=np.float16), device)
    x = npu_tensor(x_cpu, device)
    z = npu_tensor(z_cpu, device)
    run_repeated(
        lambda: lib.call_static_add(
            1, stream_ptr(), tensor_ptr(out), tensor_ptr(x), tensor_ptr(z)
        )
    )
    assert_close(out.cpu(), torch.from_numpy((x_cpu + z_cpu).astype(np.float16)))
    print("PASS static_single_core/a5 add shape=(64,64)")


def run_matmul(device: str) -> None:
    lib = ctypes.CDLL(str(compile_kernel(HERE / "compile.sh", "matmul")))
    lib.call_static_matmul.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    rng = np.random.default_rng(1)
    a_cpu = rng.standard_normal((16, 16)).astype(np.float16)
    b_cpu = rng.standard_normal((16, 16)).astype(np.float16)
    out = npu_tensor(np.zeros((16, 16), dtype=np.float32), device)
    a = npu_tensor(a_cpu, device)
    b = npu_tensor(b_cpu, device)
    run_repeated(
        lambda: lib.call_static_matmul(
            1, stream_ptr(), tensor_ptr(out), tensor_ptr(a), tensor_ptr(b)
        )
    )
    ref = torch.from_numpy(a_cpu.astype(np.float32) @ b_cpu.astype(np.float32))
    assert_close(out.cpu(), ref)
    print("PASS static_single_core/a5 matmul shape=(16,16)x(16,16)")


def run_matmul_add(device: str) -> None:
    lib = ctypes.CDLL(str(compile_kernel(HERE / "compile.sh", "matmul_add")))
    lib.call_static_matmul_add.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
    ]
    block_dim = 8
    batch = block_dim * 128
    rng = np.random.default_rng(2)
    a_cpu = rng.standard_normal((batch, 128)).astype(np.float16)
    b_cpu = rng.standard_normal((128, 128)).astype(np.float16)
    d_cpu = rng.standard_normal((batch, 128)).astype(np.float32)
    a = npu_tensor(a_cpu, device)
    b = npu_tensor(b_cpu, device)
    d = npu_tensor(d_cpu, device)
    c = npu_tensor(np.zeros((batch, 128), dtype=np.float32), device)
    run_repeated(
        lambda: lib.call_static_matmul_add(
            block_dim,
            stream_ptr(),
            tensor_ptr(a),
            tensor_ptr(b),
            tensor_ptr(c),
            tensor_ptr(d),
            batch,
        )
    )
    ref = torch.from_numpy(a_cpu.astype(np.float32) @ b_cpu.astype(np.float32) + d_cpu)
    assert_close(c.cpu(), ref)
    print(f"PASS static_single_core/a5 matmul_add rounds=1 batch={batch}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kernel", choices=("add", "matmul", "matmul_add", "all"), default="all"
    )
    parser.add_argument("--device", default=os.environ.get("NPU_DEVICE", "npu:0"))
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    configure_torch_npu(simulator_safe=True)
    torch.npu.set_device(args.device)
    if args.kernel in ("add", "all"):
        run_add(args.device)
    if args.kernel in ("matmul", "all"):
        run_matmul(args.device)
    if args.kernel in ("matmul_add", "all"):
        run_matmul_add(args.device)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps({"result": "PASS", "kernel": args.kernel}, indent=2)
        )


if __name__ == "__main__":
    main()
