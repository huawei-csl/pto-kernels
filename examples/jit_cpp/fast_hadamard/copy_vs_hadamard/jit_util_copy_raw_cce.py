import os
import subprocess
from pathlib import Path

from jit_util_common import DEFAULT_DEVICE, chmod_output_path, normalize_npu_device

from copy_vs_hadamard.jit_util_copy_common import load_copy_lib

ASCEND_TOOLKIT_HOME = os.environ["ASCEND_TOOLKIT_HOME"]
ASCEND_C_ROOT = Path(ASCEND_TOOLKIT_HOME) / "aarch64-linux" / "asc"
ASCENDC_ROOT = Path(ASCEND_TOOLKIT_HOME) / "aarch64-linux" / "ascendc"
ASCEND_C_INCLUDE_DIRS = [
    ASCEND_C_ROOT,
    ASCEND_C_ROOT / "include",
    ASCEND_C_ROOT / "include" / "basic_api",
    ASCEND_C_ROOT / "include" / "interface",
    ASCEND_C_ROOT / "include" / "utils",
    ASCEND_C_ROOT / "impl" / "basic_api",
    ASCEND_C_ROOT / "impl" / "utils",
    ASCENDC_ROOT / "include",
    ASCENDC_ROOT / "include" / "highlevel_api",
]


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
        "-O3",
        "-std=c++17",
        "--cce-aicore-arch=dav-c220-vec",
    ]
    for include_dir in ASCEND_C_INCLUDE_DIRS:
        flags.extend(["-isystem", str(include_dir)])
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
    lib_path = compile_cpp(src_path, verbose=verbose, so_dir=so_dir)
    func = load_copy_lib(lib_path, block_dim=block_dim or 1)
    if clean_up:
        os.remove(lib_path)
    return func
