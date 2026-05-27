#!/usr/bin/env python3
"""Bisheng build helper for A5 pure-vector kernels (dav-c310-vec, REGISTER_BASE)."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

A5_SIM_ROOT = Path(__file__).resolve().parent.parent
KERNEL_DIR = A5_SIM_ROOT / "kernels"
BUILD_DIR = A5_SIM_ROOT / "build"

KERNELS = {
    "silu": {
        "source": "silu_a5.cpp",
        "lib": "libsilu_a5.so",
    },
    "swiglu": {
        "source": "swiglu_a5.cpp",
        "lib": "libswiglu_a5.so",
    },
}


def _pto_include_root() -> Path:
    env = os.environ.get("PTO_LIB_PATH")
    if env:
        candidate = Path(env)
        if (candidate / "include" / "pto" / "pto-inst.hpp").is_file():
            return candidate / "include"
        if (candidate / "pto" / "pto-inst.hpp").is_file():
            return candidate
    ascend = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME")
    if ascend:
        candidate = Path(ascend)
        if (candidate / "include" / "pto" / "pto-inst.hpp").is_file():
            return candidate / "include"
    fallback = Path("/workdir/megagdn-pto/third_party/pto-isa/include")
    if (fallback / "pto" / "pto-inst.hpp").is_file():
        return fallback
    raise EnvironmentError(
        "PTO headers not found. Set PTO_LIB_PATH or source CANN setenv.bash."
    )


def _ascend_home() -> Path:
    home = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME")
    if not home:
        raise EnvironmentError("ASCEND_HOME_PATH is not set. Source CANN setenv.bash first.")
    return Path(home)


def _bisheng() -> str:
    ascend = _ascend_home()
    candidate = ascend / "bin" / "bisheng"
    if candidate.is_file():
        return str(candidate)
    found = shutil.which("bisheng")
    if found:
        return found
    raise FileNotFoundError("bisheng compiler not found")


def _common_includes() -> list[str]:
    ascend = _ascend_home()
    driver = os.environ.get("ASCEND_DRIVER_PATH", "/usr/local/Ascend/driver")
    pto_root = _pto_include_root()
    return [
        f"-I{pto_root}",
        f"-I{ascend}/include",
        f"-I{driver}/kernel/inc",
        f"-I{KERNEL_DIR}",
    ]


def _kernel_flags() -> list[str]:
    ascend = _ascend_home()
    return (
        _common_includes()
        + [
            f"-I{ascend}/pkg_inc",
            f"-I{ascend}/pkg_inc/profiling",
            f"-I{ascend}/pkg_inc/runtime/runtime",
            "-std=gnu++17",
            "-O2",
            "-Wno-macro-redefined",
            "-Wno-ignored-attributes",
            "-Wno-unknown-attributes",
            "-fPIC",
            "-xcce",
            "-Xhost-start",
            "-Xhost-end",
            "-mllvm",
            "-cce-aicore-stack-size=0x8000",
            "-mllvm",
            "-cce-aicore-function-stack-size=0x8000",
            "-mllvm",
            "-cce-aicore-record-overflow=true",
            "-mllvm",
            "-cce-aicore-addr-transform",
            "-mllvm",
            "-cce-aicore-dcci-insert-for-scalar=false",
            "--cce-aicore-arch=dav-c310-vec",
            "-DREGISTER_BASE",
        ]
    )


def _run(cmd: list[str], cwd: Path) -> None:
    print("==>", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def build_kernel(name: str, force: bool = False) -> Path:
    if name not in KERNELS:
        raise ValueError(f"unknown kernel: {name}")
    spec = KERNELS[name]
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    out = BUILD_DIR / spec["lib"]
    if out.is_file() and not force:
        return out

    src_path = KERNEL_DIR / spec["source"]
    obj = BUILD_DIR / f"{src_path.stem}.o"
    bisheng = _bisheng()
    _run([bisheng, *_kernel_flags(), "-c", str(src_path), "-o", str(obj)], cwd=BUILD_DIR)
    _run(
        [
            bisheng,
            "-fPIC",
            "-shared",
            "--cce-fatobj-link",
            "-Wl,-soname," + spec["lib"],
            str(obj),
            "-o",
            str(out),
        ],
        cwd=BUILD_DIR,
    )
    print(f"Built {out}")
    return out


def build_all(force: bool = False) -> dict[str, Path]:
    return {name: build_kernel(name, force=force) for name in KERNELS}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build A5 pure-vector example kernels")
    parser.add_argument("--kernel", choices=tuple(KERNELS.keys()))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        if args.all:
            build_all(force=args.force)
        elif args.kernel:
            build_kernel(args.kernel, force=args.force)
        else:
            parser.print_help()
            raise SystemExit(1)
    except (EnvironmentError, FileNotFoundError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"build failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
