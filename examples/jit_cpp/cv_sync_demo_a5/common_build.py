#!/usr/bin/env python3
"""Build helper for the A5 Cube/Vector sync demo."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
BUILD_DIR = HERE / "build"
KERNEL_LIB = "libcv_sync_kernels.so"
OUT_LIB = "libcv_sync_demo_a5.so"
KERNEL_SOURCES = [
    "stream_c2v.cpp",
    "stream_v2c.cpp",
    "matmul_add_c2v.cpp",
    "add_matmul_v2c.cpp",
]


def _ascend_home() -> Path:
    home = os.environ.get("ASCEND_HOME_PATH") or "/usr/local/Ascend/cann-9.0.0"
    return Path(home)


def _bisheng() -> str:
    candidate = _ascend_home() / "bin" / "bisheng"
    if candidate.is_file():
        return str(candidate)
    found = shutil.which("bisheng")
    if found:
        return found
    raise FileNotFoundError("bisheng compiler not found; source CANN set_env.sh first")


def _pto_isa_root() -> Path:
    override = os.environ.get("PTO_ISA_ROOT")
    if override:
        return Path(override)
    vendored = REPO_ROOT / "third_party" / "pto-isa"
    if vendored.is_dir():
        return vendored
    return Path("/home/jzhuang/pto-isa")


def _includes() -> list[str]:
    ascend = _ascend_home()
    driver = Path(os.environ.get("ASCEND_DRIVER_PATH", "/usr/local/Ascend/driver"))
    return [
        f"-I{_pto_isa_root()}/include",
        f"-I{ascend}/include",
        f"-I{driver}/kernel/inc",
        f"-I{HERE}",
        f"-I{ascend}/pkg_inc",
        f"-I{ascend}/pkg_inc/profiling",
        f"-I{ascend}/pkg_inc/runtime/runtime",
    ]


def _kernel_flags() -> list[str]:
    return _includes() + [
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
        "--cce-aicore-arch=dav-c310",
        "-DREGISTER_BASE",
    ]


def _host_flags() -> list[str]:
    return _includes() + [
        "-std=gnu++17",
        "-O2",
        "-Wno-macro-redefined",
        "-Wno-ignored-attributes",
        "-Wno-unknown-attributes",
        "-xc++",
        "-include",
        "stdint.h",
        "-include",
        "stddef.h",
        "-fPIC",
    ]


def _run(cmd: list[str]) -> None:
    print("==>", " ".join(cmd))
    subprocess.run(cmd, cwd=BUILD_DIR, check=True)


def build() -> Path:
    bisheng = _bisheng()
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    kernel_objects: list[Path] = []
    for src in KERNEL_SOURCES:
        src_path = HERE / src
        obj = BUILD_DIR / f"{src_path.stem}.o"
        _run([bisheng, *_kernel_flags(), "-c", str(src_path), "-o", str(obj)])
        kernel_objects.append(obj)

    kernel_so = BUILD_DIR / KERNEL_LIB
    _run(
        [
            bisheng,
            "-fPIC",
            "-shared",
            "--cce-fatobj-link",
            f"-Wl,-soname,{KERNEL_LIB}",
            *[str(o) for o in kernel_objects],
            "-o",
            str(kernel_so),
        ]
    )

    launch_obj = BUILD_DIR / "launch_api.o"
    _run([bisheng, *_host_flags(), "-c", str(HERE / "launch_api.cpp"), "-o", str(launch_obj)])

    out = BUILD_DIR / OUT_LIB
    _run(
        [
            bisheng,
            str(launch_obj),
            "-shared",
            "-o",
            str(out),
            f"-L{BUILD_DIR}",
            "-lcv_sync_kernels",
            "-lstdc++",
            f"-Wl,-rpath,{BUILD_DIR}",
        ]
    )
    return out


if __name__ == "__main__":
    print(build())

