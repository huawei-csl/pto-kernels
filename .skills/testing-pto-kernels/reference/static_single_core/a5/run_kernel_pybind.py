#!/usr/bin/env python3
from __future__ import annotations

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
    ext_dir = BUILD / "torch_ext_static_a5"
    ext_dir.mkdir(exist_ok=True)
    add_lib = build_kernel("add")
    matmul_lib = build_kernel("matmul")
    return load(
        name="pto_static_a5_demo",
        sources=[str(HERE / "pybind.cpp")],
        extra_ldflags=[str(add_lib), str(matmul_lib)],
        build_directory=str(ext_dir),
        verbose=False,
    )


def npu_tensor(arr: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(arr).to(device)


def main() -> None:
    device = os.environ.get("NPU_DEVICE", "npu:0")
    configure_torch_npu(simulator_safe=True)
    torch.npu.set_device(device)
    try:
        import pto_static_a5_demo as pkg
    except ImportError:
        pkg = None
        print("Using pybind launch path: torch.utils.cpp_extension.load")
        mod = build_module()
    else:
        print("Using pybind launch path: installed pto_static_a5_demo package")
        mod = None
    stream = stream_as_int()

    rng = np.random.default_rng(0)
    x_cpu = rng.standard_normal((64, 64)).astype(np.float16)
    z_cpu = rng.standard_normal((64, 64)).astype(np.float16)
    x = npu_tensor(x_cpu, device)
    z = npu_tensor(z_cpu, device)
    out = npu_tensor(np.zeros((64, 64), dtype=np.float16), device)
    if pkg is not None:
        run_repeated(lambda: pkg.add(out, x, z, stream))
    else:
        run_repeated(lambda: mod.launch_static_add(out, x, z, stream))
    assert_close(out.cpu(), torch.from_numpy((x_cpu + z_cpu).astype(np.float16)))
    print("PASS static_single_core/a5 pybind add")

    a_cpu = rng.standard_normal((16, 16)).astype(np.float16)
    b_cpu = rng.standard_normal((16, 16)).astype(np.float16)
    a = npu_tensor(a_cpu, device)
    b = npu_tensor(b_cpu, device)
    mat_out = npu_tensor(np.zeros((16, 16), dtype=np.float32), device)
    if pkg is not None:
        run_repeated(lambda: pkg.matmul(mat_out, a, b, stream))
    else:
        run_repeated(lambda: mod.launch_static_matmul(mat_out, a, b, stream))
    assert_close(
        mat_out.cpu(),
        torch.from_numpy(a_cpu.astype(np.float32) @ b_cpu.astype(np.float32)),
    )
    print("PASS static_single_core/a5 pybind matmul")


if __name__ == "__main__":
    main()
