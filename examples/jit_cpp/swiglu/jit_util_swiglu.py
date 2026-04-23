# pylint: disable=wrong-import-position
import ctypes
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
FAST_HADAMARD_DIR = THIS_DIR.parent / "fast_hadamard"
if str(FAST_HADAMARD_DIR) not in sys.path:
    sys.path.insert(0, str(FAST_HADAMARD_DIR))

from jit_util_common import (  # noqa: E402
    BLOCK_DIM,
    DEFAULT_DEVICE,
    jit_compile_with_loader,
    load_cdll,
    load_required_symbol,
    resolve_launch_block_dim,
    resolve_stream_ptr,
    torch_to_ctypes,
)

UINT32_MAX = (1 << 32) - 1
SWIGLU_ARGTYPES = [
    ctypes.c_uint32,  # blockDim
    ctypes.c_void_p,  # stream
    ctypes.c_void_p,  # x
    ctypes.c_void_p,  # y
    ctypes.c_uint32,  # batch
    ctypes.c_uint32,  # input_n
]


def _validate_swiglu_io(x, y):
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError("x and y must be 2D tensors.")
    if x.dtype != torch.float16:
        raise TypeError("x must use torch.float16.")
    if y.dtype != torch.float16:
        raise TypeError("y must use torch.float16.")
    if x.device != y.device:
        raise ValueError("x and y must be on the same device.")
    if not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("x and y must be contiguous.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same batch size.")
    if x.shape[1] <= 0 or (x.shape[1] & 1):
        raise ValueError("x.shape[1] must be a positive even integer.")
    if x.shape[0] > UINT32_MAX or x.shape[1] > UINT32_MAX:
        raise ValueError("x dimensions must fit uint32_t kernel arguments.")
    output_n = x.shape[1] // 2
    if y.shape[1] != output_n:
        raise ValueError("y must have shape [batch, x.shape[1] // 2].")


def load_lib(lib_path, block_dim=BLOCK_DIM):
    lib = load_cdll(lib_path)
    resolved_block_dim = max(1, int(block_dim))

    kernel = load_required_symbol(
        lib,
        "call_swiglu_kernel",
        SWIGLU_ARGTYPES,
    )

    def swiglu_func(x, y, *, block_dim=resolved_block_dim, stream_ptr=None):
        _validate_swiglu_io(x, y)
        kernel(
            resolve_launch_block_dim(block_dim, resolved_block_dim),
            resolve_stream_ptr(stream_ptr),
            torch_to_ctypes(x),
            torch_to_ctypes(y),
            x.shape[0],
            x.shape[1],
        )

    swiglu_func.block_dim = resolved_block_dim
    return swiglu_func


def jit_compile(
    src_path,
    verbose=True,
    clean_up=False,
    so_dir=None,
    device: str | int = DEFAULT_DEVICE,
    block_dim=None,
):
    if so_dir is None:
        so_dir = THIS_DIR / "outputs" / "so"
    return jit_compile_with_loader(
        src_path,
        load_lib,
        verbose=verbose,
        clean_up=clean_up,
        so_dir=so_dir,
        device=device,
        block_dim=block_dim,
    )
