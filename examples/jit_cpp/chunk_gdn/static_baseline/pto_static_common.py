"""
Shared PTO static-kernel build helpers (bisheng, include order, compiled_lib output).
"""
from __future__ import annotations

import os
import subprocess
from functools import lru_cache

ASCEND_TOOLKIT_HOME = os.environ.get("ASCEND_TOOLKIT_HOME") or os.environ.get(
    "ASCEND_HOME_PATH", ""
)
if not ASCEND_TOOLKIT_HOME:
    raise RuntimeError("Set ASCEND_TOOLKIT_HOME or ASCEND_HOME_PATH")

PTO_LIB_PATH = os.environ.get("PTO_LIB_PATH", ASCEND_TOOLKIT_HOME)
_pto_inc = os.path.join(PTO_LIB_PATH, "include")
if not os.path.isdir(_pto_inc):
    raise RuntimeError(
        f"PTO include directory missing: {_pto_inc!r} (set PTO_LIB_PATH; must be before CANN -I)."
    )

_HERE = os.path.dirname(os.path.abspath(__file__))
INCLUDE_DIR = os.path.join(_HERE, "include")
COMPILED_DIR = os.path.join(_HERE, "compiled_lib")
_DRIVER_INC = "/usr/local/Ascend/driver/kernel/inc"


@lru_cache(maxsize=32)
def compile_pto_kernel(kernel_cpp_basename: str, so_basename: str) -> str:
    """Compile ``kernel_cpp_basename`` under this directory to ``compiled_lib/so_basename``."""
    os.makedirs(COMPILED_DIR, exist_ok=True)
    cpp_path = os.path.join(_HERE, kernel_cpp_basename)
    lib_path = os.path.join(COMPILED_DIR, so_basename)
    extra = os.environ.get("PTO_STATIC_EXTRA_FLAGS", "").split()
    flags = [
        "-fPIC",
        "-shared",
        "-xcce",
        "-DMEMORY_BASE",
        "-O2",
        "-std=gnu++17",
        "--cce-aicore-arch=dav-c220",
        "-mllvm",
        "-cce-aicore-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-function-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-record-overflow=true",
        "-mllvm",
        "-cce-aicore-dcci-insert-for-scalar=false",
        "-Wno-macro-redefined",
        "-Wno-ignored-attributes",
        f"-I{INCLUDE_DIR}",
        f"-I{_pto_inc}",
        f"-I{ASCEND_TOOLKIT_HOME}/include",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc/runtime",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc/profiling",
    ]
    if os.path.isdir(_DRIVER_INC):
        flags.append(f"-I{_DRIVER_INC}")
    flags.extend(extra)
    cmd = ["bisheng", *flags, cpp_path, "-o", lib_path]
    if os.environ.get("VERBOSE_COMPILE"):
        print("compile:", " ".join(cmd))
    subprocess.run(cmd, check=True, timeout=300)
    return lib_path
