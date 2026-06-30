#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from torch.utils.cpp_extension import load

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"

sys.path.insert(0, str(HERE.parents[1]))
from pto_demo_utils import (  # noqa: E402
    assert_close,
    configure_torch_npu,
    cube_core_count,
    run_repeated,
    stream_as_int,
    vector_core_count,
)


def build_kernel(name: str) -> Path:
    out = subprocess.check_output(["bash", str(HERE / "compile.sh"), name], text=True)
    return Path(out.strip().splitlines()[-1])


def build_module():
    BUILD.mkdir(exist_ok=True)
    ext_dir = BUILD / "torch_ext_dynamic"
    ext_dir.mkdir(exist_ok=True)
    add_lib = build_kernel("add")
    matmul_lib = build_kernel("matmul")
    print("Using pybind launch path: torch.utils.cpp_extension.load")
    return load(
        name="pto_dynamic_a2a3_demo",
        sources=[str(HERE / "pybind.cpp")],
        extra_ldflags=[str(add_lib), str(matmul_lib), f"-Wl,-rpath,{add_lib.parent}"],
        build_directory=str(ext_dir),
        verbose=False,
    )


def run_add(mod, device: str, n: int, stream: int) -> None:
    x = torch.randn(n, device=device, dtype=torch.float16)
    z = torch.randn(n, device=device, dtype=torch.float16)
    y = torch.empty_like(x)
    block_dim = vector_core_count(device)
    run_repeated(lambda: mod.launch_add(y, x, z, n, block_dim, stream))
    assert_close(y, x + z)
    print(f"PASS pybind add n={n} block_dim={block_dim}")


def run_matmul(mod, device: str, m: int, stream: int) -> None:
    if m % 128 != 0:
        raise ValueError("m must be a multiple of 128")
    torch.manual_seed(0)
    a = torch.randn(m, 128, device=device, dtype=torch.float16)
    b = torch.randn(128, 128, device=device, dtype=torch.float16)
    c = torch.empty(m, 128, device=device, dtype=torch.float16)
    block_dim = min(cube_core_count(device), max(1, m // 128))
    run_repeated(lambda: mod.launch_matmul(a, b, c, m, block_dim, stream))
    assert_close(c, torch.matmul(a, b))
    print(f"PASS pybind simple_matmul m={m} block_dim={block_dim}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", choices=("add", "matmul", "all"), default="all")
    parser.add_argument("--device", default=os.environ.get("NPU_DEVICE", "npu:0"))
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--m", type=int, default=128)
    args = parser.parse_args()

    configure_torch_npu()
    torch.npu.set_device(args.device)
    mod = build_module()
    stream = stream_as_int()
    if args.kernel in ("add", "all"):
        run_add(mod, args.device, args.n, stream)
    if args.kernel in ("matmul", "all"):
        run_matmul(mod, args.device, args.m, stream)


if __name__ == "__main__":
    main()
