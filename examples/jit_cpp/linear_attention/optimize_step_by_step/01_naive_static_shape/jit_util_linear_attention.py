from pathlib import Path
import sys

COMMON_DIR = Path(__file__).resolve().parents[1] / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from functools import lru_cache

from jit_shared import compile_cpp as shared_compile_cpp
from jit_shared import load_static_nomask_lib


def compile_cpp(kernel_cpp: str, verbose: bool = False, timeout: int = 180) -> str:
    return shared_compile_cpp(
        kernel_cpp,
        output_name="linear_attention_jit.so",
        std="c++17",
        verbose=verbose,
        timeout=timeout,
    )


def load_lib(lib_path: str):
    return load_static_nomask_lib(lib_path)


@lru_cache(maxsize=None)
def jit_compile(src_path: str, verbose: bool = True, clean_up: bool = False):
    lib_path = compile_cpp(src_path, verbose=verbose)
    func = load_lib(lib_path)
    if clean_up:
        Path(lib_path).unlink(missing_ok=True)
    return func
