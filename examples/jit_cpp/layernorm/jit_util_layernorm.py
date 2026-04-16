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

LAYERNORM_ARGTYPES = [
    ctypes.c_uint32,   # blockDim
    ctypes.c_void_p,   # stream
    ctypes.c_void_p,   # x
    ctypes.c_void_p,   # gamma
    ctypes.c_void_p,   # beta
    ctypes.c_void_p,   # y
    ctypes.c_uint32,   # rows
    ctypes.c_uint32,   # hidden
    ctypes.c_float,    # eps
    ctypes.c_float,    # inv_hidden
]


def _validate_layernorm_io(x, gamma, beta, y):
    if x.dim() != 2:
        raise ValueError("x must be a 2D tensor.")
    if gamma.dim() != 1 or beta.dim() != 1:
        raise ValueError("gamma and beta must be 1D tensors.")
    if y.dim() != 2:
        raise ValueError("y must be a 2D tensor.")
    for name, t in [("x", x), ("gamma", gamma), ("beta", beta), ("y", y)]:
        if t.dtype != torch.float16:
            raise TypeError(f"{name} must use torch.float16.")
    if not (x.is_contiguous() and gamma.is_contiguous()
            and beta.is_contiguous() and y.is_contiguous()):
        raise ValueError("All tensors must be contiguous.")
    rows, hidden = x.shape
    if gamma.shape[0] != hidden or beta.shape[0] != hidden:
        raise ValueError("gamma and beta must have shape [hidden].")
    if y.shape != x.shape:
        raise ValueError("y must have the same shape as x.")


def load_lib(lib_path, block_dim=BLOCK_DIM):
    lib = load_cdll(lib_path)
    resolved_block_dim = max(1, int(block_dim))

    kernel = load_required_symbol(
        lib,
        "call_layernorm_kernel",
        LAYERNORM_ARGTYPES,
    )

    def layernorm_func(x, gamma, beta, y, *, eps=1e-5,
                       block_dim=resolved_block_dim, stream_ptr=None):
        _validate_layernorm_io(x, gamma, beta, y)
        rows, hidden = x.shape
        kernel(
            resolve_launch_block_dim(block_dim, resolved_block_dim),
            resolve_stream_ptr(stream_ptr),
            torch_to_ctypes(x),
            torch_to_ctypes(gamma),
            torch_to_ctypes(beta),
            torch_to_ctypes(y),
            rows,
            hidden,
            float(eps),
            1.0 / float(hidden),
        )

    layernorm_func.block_dim = resolved_block_dim
    return layernorm_func


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
