"""Shared bisheng JIT build for the A5 kernels in this directory.

The repo-wide ``examples/jit_cpp`` helper targets dav-c220 (``-DMEMORY_BASE``);
these are A5 (``dav-c310-vec``, ``-DREGISTER_BASE``), so bisheng is invoked
directly. Building needs a working bisheng (``ASCEND_HOME_PATH`` or
``ASCEND_TOOLKIT_HOME``); running additionally needs ``torch_npu``.
"""

import ctypes
import os
import subprocess
from pathlib import Path

BLOCK_DIM = 64  # 64 AIC -> 128 AIV, optimal on this device
NBUF = 4  # pipeline depth (UB buffers)
PREFETCH = 2  # tiles prefetched ahead
# every launcher here takes (block_dim, stream, ptr, batch)
KERNEL_ARGS = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
# Flags that never vary. Written as one string and split: black re-explodes a list
# of short strings to one entry per line, but leaves this alone.
FIXED_FLAGS = (
    "-O2 -std=c++17 -fPIC -Wno-ignored-attributes -Wno-macro-redefined "
    "-mllvm -cce-aicore-stack-size=0x8000 "
    "-mllvm -cce-aicore-function-stack-size=0x8000 "
    "-mllvm -cce-aicore-addr-transform "
    "-mllvm -cce-aicore-dcci-insert-for-scalar=false "
    "-Xhost-start -Xhost-end"
).split()


def ascend_home() -> str:
    for key in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        if os.environ.get(key):
            return os.environ[key]
    raise RuntimeError("set ASCEND_HOME_PATH or ASCEND_TOOLKIT_HOME")


def compile_so(src, tag, defs=(), out_dir=None, verbose=True, force=False) -> Path:
    """Compile and link one .cpp to a device .so.

    ``tag`` names the artifact and must carry every macro that changes codegen, or
    distinct configurations collide on one file. An up-to-date .so is reused
    unless ``force``.
    """
    src = Path(src)
    out_dir = Path(out_dir) if out_dir else src.parent / "build"
    out_dir.mkdir(parents=True, exist_ok=True)
    obj, so = out_dir / f"{tag}.o", out_dir / f"{tag}.so"
    if not force and so.exists() and so.stat().st_mtime >= src.stat().st_mtime:
        if verbose:
            print(f"[compile] up-to-date, reusing {so.name}")
        return so
    home = ascend_home()
    bisheng = f"{home}/bin/bisheng"
    arch = ["--cce-aicore-arch=dav-c310-vec", "-DREGISTER_BASE"]
    inc = [f"-I{home}/aarch64-linux/include", f"-I{home}/include"]
    if verbose:
        print(f"[compile] {tag}: {' '.join(defs)}")
    cmd = [bisheng, "-xcce", *arch, *defs, *FIXED_FLAGS, *inc]
    subprocess.run([*cmd, "-c", str(src), "-o", str(obj)], check=True)
    link = f"-fPIC -shared --cce-fatobj-link -Wl,-soname,{so.name}".split()
    subprocess.run([bisheng, *link, str(obj), "-o", str(so)], check=True)
    return so


def entry(so_path, name, argtypes=None):
    """ctypes-load ``so_path`` and bind the launcher ``name``."""
    fn = getattr(ctypes.CDLL(str(so_path)), name)
    fn.argtypes = list(KERNEL_ARGS if argtypes is None else argtypes)
    fn.restype = None
    return fn


def stream_ptr(ptr=None):
    if ptr is not None:
        return ptr
    import torch  # noqa: F401

    resolved = getattr(torch.npu.current_stream(), "_as_parameter_", None)
    if resolved is None:
        raise RuntimeError("could not resolve the current NPU stream pointer")
    return resolved
