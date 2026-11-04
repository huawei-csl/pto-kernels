# Copyright (c) 2026 Huawei Technologies Co., Ltd. See repository LICENSE.
"""Cached device/native builds and the single-call C ABI; no tensor computations."""
import ctypes
import functools
import hashlib
import os
from pathlib import Path
import subprocess
import time

import torch
import torch_npu


@functools.lru_cache(None)
def build():
    root = Path(__file__).resolve().parents[3]
    source = Path(__file__).with_name("kernel.cpp")
    flags = [
        "-DMEMORY_BASE",
        "-O2",
        "-std=c++17",
        "-fPIC",
        "--npu-arch=dav-2201",
        "--asc-aicore-lang",
        "-shared",
    ]
    key = hashlib.sha256(
        source.read_bytes()
        + b"".join(p.read_bytes() for p in sorted(source.parent.glob("*.h")))
        + (root / "csrc/kernel/kernel_utils.h").read_bytes()
        + repr(flags).encode()
    ).hexdigest()[:16]
    target = root / "build/gdn_mega_bwd" / key / "kernel.so"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        temporary = target.with_suffix(".tmp.so")
        subprocess.run(
            ["bisheng", *flags, str(source), "-o", str(temporary)], check=True
        )
        temporary.replace(target)
        print(f"GDN backward build: {time.perf_counter()-start:.2f}s")
    native = source.with_name("launch.cpp")
    torch_root, npu_root = Path(torch.__file__).parent, Path(torch_npu.__file__).parent
    cann = Path(os.environ["ASCEND_HOME_PATH"])
    abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
    native_key = hashlib.sha256(
        native.read_bytes()
        + source.with_name("layout.h").read_bytes()
        + repr((torch.__version__, torch_npu.__version__, abi)).encode()
    ).hexdigest()[:16]
    library = root / "build/gdn_mega_bwd" / native_key / "launch.so"
    if not library.exists():
        library.parent.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        temporary = library.with_suffix(".tmp.so")
        includes = [
            torch_root / "include",
            torch_root / "include/torch/csrc/api/include",
            npu_root / "include",
            cann / "include",
        ]
        libs = [torch_root / "lib", npu_root / "lib"]
        subprocess.run(
            [
                "g++",
                "-O2",
                "-std=c++17",
                "-fPIC",
                "-shared",
                f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
                *[f"-I{x}" for x in includes],
                str(native),
                *[f"-L{x}" for x in libs],
                *[f"-Wl,-rpath,{x}" for x in libs],
                "-ltorch_npu",
                "-ltorch_cpu",
                "-lc10",
                "-o",
                str(temporary),
            ],
            check=True,
        )
        temporary.replace(library)
        print(f"GDN native launcher build: {time.perf_counter()-start:.2f}s")
    lib = ctypes.CDLL(str(library))
    lib.device_library = ctypes.CDLL(str(target))
    lib.stage_launcher = ctypes.cast(lib.device_library.launch_stages, ctypes.c_void_p)
    lib.workspace_elements.argtypes = [
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.workspace_elements.restype = ctypes.c_int64
    lib.gdn_backward.argtypes = [
        ctypes.POINTER(Input),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_float,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    lib.gdn_backward.restype = None
    return lib


class Input(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("stride", ctypes.c_int64 * 4),
        ("dtype", ctypes.c_int64),
    ]

    @classmethod
    def from_tensor(cls, x):
        if x is None:
            return cls(None, (ctypes.c_int64 * 4)(), 2)
        strides = (*x.stride(), *([0] * (4 - x.ndim)))
        return cls(
            x.data_ptr(),
            (ctypes.c_int64 * 4)(*strides),
            (torch.float16, torch.bfloat16, torch.float32).index(x.dtype),
        )
