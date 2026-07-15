#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa: F401
from torch.utils.cpp_extension import load

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"

sys.path.insert(0, str(HERE.parents[1]))
from pto_demo_utils import (
    assert_close,
    configure_torch_npu,
    run_repeated,
    stream_as_int,
)  # noqa: E402


def build_kernel(name: str) -> Path:
    out = subprocess.check_output(["bash", str(HERE / "compile.sh"), name], text=True)
    return Path(out.strip().splitlines()[-1])


def build_module():
    BUILD.mkdir(exist_ok=True)
    ext_dir = BUILD / "torch_ext_dynamic_a5"
    ext_dir.mkdir(exist_ok=True)
    add_lib = build_kernel("add")
    matmul_lib = build_kernel("matmul")
    print("Using torch extension pybind path: pto_dynamic_a5_demo")
    return load(
        name="pto_dynamic_a5_demo",
        sources=[str(HERE / "pybind.cpp")],
        extra_ldflags=[str(add_lib), str(matmul_lib), f"-Wl,-rpath,{add_lib.parent}"],
        build_directory=str(ext_dir),
        verbose=False,
    )


def npu_tensor(arr: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(arr).to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", choices=("add", "matmul", "all"), default="all")
    parser.add_argument("--device", default=os.environ.get("NPU_DEVICE", "npu:0"))
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--block-dim", type=int, default=8)
    args = parser.parse_args()

    configure_torch_npu(simulator_safe=True)
    torch.npu.set_device(args.device)
    mod = build_module()
    stream = stream_as_int()

    if args.kernel in ("add", "all"):
        rng = np.random.default_rng(0)
        x_cpu = rng.standard_normal(args.n).astype(np.float16)
        z_cpu = rng.standard_normal(args.n).astype(np.float16)
        x = npu_tensor(x_cpu, args.device)
        z = npu_tensor(z_cpu, args.device)
        y = npu_tensor(np.zeros(args.n, dtype=np.float16), args.device)
        run_repeated(lambda: mod.launch_add(y, x, z, args.n, args.block_dim, stream))
        assert_close(y.cpu(), torch.from_numpy((x_cpu + z_cpu).astype(np.float16)))
        print(
            f"PASS dynamic_multi_core/a5 pybind add n={args.n} block_dim={args.block_dim}"
        )

    if args.kernel in ("matmul", "all"):
        rng = np.random.default_rng(1)
        a_cpu = rng.standard_normal((16, 16)).astype(np.float16)
        b_cpu = rng.standard_normal((16, 16)).astype(np.float16)
        a = npu_tensor(a_cpu, args.device)
        b = npu_tensor(b_cpu, args.device)
        out = npu_tensor(np.zeros((16, 16), dtype=np.float32), args.device)
        run_repeated(lambda: mod.launch_matmul(out, a, b, 1, stream))
        ref = torch.from_numpy(a_cpu.astype(np.float32) @ b_cpu.astype(np.float32))
        assert_close(out.cpu(), ref)
        print("PASS dynamic_multi_core/a5 pybind matmul shape=16x16x16")


if __name__ == "__main__":
    main()
