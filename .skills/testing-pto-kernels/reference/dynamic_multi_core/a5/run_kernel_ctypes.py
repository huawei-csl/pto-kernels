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


def npu_from_cpu(arr: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(arr).to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", choices=("add", "matmul", "all"), default="add")
    parser.add_argument("--device", default=os.environ.get("NPU_DEVICE", "npu:0"))
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--block-dim", type=int, default=8)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    configure_torch_npu(simulator_safe=True)
    torch.npu.set_device(args.device)

    payloads = []
    if args.kernel in ("add", "all"):
        lib = ctypes.CDLL(str(compile_kernel(HERE / "compile.sh", "add")))
        lib.call_add.argtypes = [
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        rng = np.random.default_rng(0)
        x_cpu = rng.standard_normal(args.n).astype(np.float16)
        z_cpu = rng.standard_normal(args.n).astype(np.float16)
        x = npu_from_cpu(x_cpu, args.device)
        z = npu_from_cpu(z_cpu, args.device)
        y = npu_from_cpu(np.zeros(args.n, dtype=np.float16), args.device)
        stream = stream_ptr()
        run_repeated(
            lambda: lib.call_add(
                args.block_dim,
                stream,
                tensor_ptr(y),
                tensor_ptr(x),
                tensor_ptr(z),
                args.n,
            )
        )
        assert_close(y.cpu(), torch.from_numpy((x_cpu + z_cpu).astype(np.float16)))
        payloads.append(
            {
                "kernel": "add",
                "n": args.n,
                "block_dim": args.block_dim,
                "result": "PASS",
            }
        )
    if args.kernel in ("matmul", "all"):
        lib = ctypes.CDLL(str(compile_kernel(HERE / "compile.sh", "matmul")))
        lib.call_matmul.argtypes = [
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        rng = np.random.default_rng(1)
        a_cpu = rng.standard_normal((16, 16)).astype(np.float16)
        b_cpu = rng.standard_normal((16, 16)).astype(np.float16)
        a = npu_from_cpu(a_cpu, args.device)
        b = npu_from_cpu(b_cpu, args.device)
        y = npu_from_cpu(np.zeros((16, 16), dtype=np.float32), args.device)
        stream = stream_ptr()
        run_repeated(
            lambda: lib.call_matmul(
                1, stream, tensor_ptr(y), tensor_ptr(a), tensor_ptr(b)
            )
        )
        ref = torch.from_numpy(a_cpu.astype(np.float32) @ b_cpu.astype(np.float32))
        assert_close(y.cpu(), ref)
        payloads.append(
            {"kernel": "matmul", "shape": "16x16x16", "block_dim": 1, "result": "PASS"}
        )
    payload = {"result": "PASS", "results": payloads}
    print(json.dumps(payload, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
