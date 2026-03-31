import os
import subprocess
from pathlib import Path

from jit_util_common import (
    DEFAULT_DEVICE,
    PTO_LIB_PATH,
    chmod_output_path,
    normalize_npu_device,
    resolve_block_dim,
)

from copy_vs_hadamard.jit_util_copy_common import load_copy_lib


def compile_cpp(
    kernel_cpp: str,
    verbose: bool = False,
    timeout: int = 120,
    so_dir: str | None = None,
) -> str:
    kernel_path = Path(kernel_cpp).resolve()
    out_dir = (
        Path(so_dir) if so_dir is not None else kernel_path.parent / "outputs" / "so"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    chmod_output_path(out_dir)
    lib_path = out_dir / f"{kernel_path.stem}_jit.so"

    flags = [
        "-fPIC",
        "-shared",
        "-xcce",
        "-DMEMORY_BASE",
        "-O3",
        "-std=c++17",
        "-Wno-ignored-attributes",
        "--cce-aicore-arch=dav-c220-vec",
        "-isystem",
        f"{PTO_LIB_PATH}/include",
    ]
    command = ["bisheng", *flags, str(kernel_path), "-o", str(lib_path)]
    if verbose:
        print("compile command:", " ".join(command))

    try:
        subprocess.run(command, timeout=timeout, check=True)
    except Exception as e:
        raise RuntimeError(f"Compile failed: {e}") from e

    if verbose:
        print(f"generated {lib_path}")
    chmod_output_path(lib_path)
    return str(lib_path)


def jit_compile(
    src_path,
    verbose=True,
    clean_up=False,
    so_dir=None,
    device: str | int = DEFAULT_DEVICE,
    block_dim=None,
):
    normalize_npu_device(device)
    resolved_block_dim = resolve_block_dim(device, block_dim)
    lib_path = compile_cpp(src_path, verbose=verbose, so_dir=so_dir)
    func = load_copy_lib(lib_path, block_dim=resolved_block_dim)
    if clean_up:
        os.remove(lib_path)
    return func
