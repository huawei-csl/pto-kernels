#!/usr/bin/env python3
from __future__ import annotations

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
from pto_demo_utils import (
    assert_close,
    configure_torch_npu,
    run_repeated,
    stream_as_int,
)  # noqa: E402


def build_kernel() -> Path:
    out = subprocess.check_output(["bash", str(HERE / "compile.sh"), "add"], text=True)
    return Path(out.strip().splitlines()[-1])


def build_module():
    BUILD.mkdir(exist_ok=True)
    ext_dir = BUILD / "torch_ext_static"
    ext_dir.mkdir(exist_ok=True)
    add_lib = build_kernel()
    print("Using pybind launch path: torch.utils.cpp_extension.load")
    return load(
        name="pto_static_a2a3_demo",
        sources=[str(HERE / "pybind.cpp")],
        extra_ldflags=[str(add_lib), f"-Wl,-rpath,{add_lib.parent}"],
        build_directory=str(ext_dir),
        verbose=False,
    )


def main() -> None:
    device = os.environ.get("NPU_DEVICE", "npu:0")
    configure_torch_npu()
    torch.npu.set_device(device)
    mod = build_module()
    torch.manual_seed(0)
    x = torch.randn(64, 64, device=device, dtype=torch.float16)
    z = torch.randn(64, 64, device=device, dtype=torch.float16)
    out = torch.empty_like(x)
    run_repeated(lambda: mod.launch_static_add(out, x, z, stream_as_int()))
    assert_close(out, x + z)
    print("PASS static_single_core/a2a3 pybind add shape=(64,64)")


if __name__ == "__main__":
    main()
