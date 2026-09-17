from pathlib import Path
import sys

COMMON_DIR = Path(__file__).resolve().parents[1] / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from functools import lru_cache

from jit_shared import BLOCK_DIM, STEP03_KERNEL_FLAGS, compile_cpp as shared_compile_cpp
from jit_shared import load_dynamic_nomask_lib


def compile_cpp(
    kernel_cpp: str,
    num_heads: int,
    hidden_size: int,
    chunk_size: int,
    verbose: bool = False,
    timeout: int = 180,
) -> str:
    return shared_compile_cpp(
        kernel_cpp,
        output_name=f"linear_attention_H{num_heads}_D{hidden_size}_C{chunk_size}_jit.so",
        std="gnu++17",
        defines=[
            f"-DLINEAR_ATTN_H={num_heads}",
            f"-DLINEAR_ATTN_D={hidden_size}",
            f"-DLINEAR_ATTN_C={chunk_size}",
        ],
        extra_flags=STEP03_KERNEL_FLAGS,
        verbose=verbose,
        timeout=timeout,
    )


def load_lib(lib_path: str):
    return load_dynamic_nomask_lib(lib_path)


@lru_cache(maxsize=None)
def jit_compile(
    src_path: str,
    num_heads: int,
    hidden_size: int,
    chunk_size: int,
    verbose: bool = True,
    clean_up: bool = False,
):
    lib_path = compile_cpp(
        src_path,
        num_heads=num_heads,
        hidden_size=hidden_size,
        chunk_size=chunk_size,
        verbose=verbose,
    )
    func = load_lib(lib_path)
    if clean_up:
        Path(lib_path).unlink(missing_ok=True)
    return func
