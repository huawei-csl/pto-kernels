import ctypes

import torch

from jit_util_common import (
    BLOCK_DIM,
    DEFAULT_DEVICE,
    load_cdll,
    load_required_symbol,
    resolve_launch_block_dim,
    resolve_stream_ptr,
    torch_to_ctypes,
)


def validate_copy_tensors(x, y, *, batch=None, n=None):
    if x.dtype != torch.float16 or y.dtype != torch.float16:
        raise TypeError("x and y must use torch.float16.")
    if x.device != y.device:
        raise ValueError("x and y must be on the same device.")
    if not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("x and y must be contiguous.")
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError("x and y must be 2D tensors.")
    if x.shape != y.shape:
        raise ValueError("x and y must have matching shapes.")

    resolved_batch = x.shape[0] if batch is None else int(batch)
    resolved_n = x.shape[1] if n is None else int(n)
    if x.shape != (resolved_batch, resolved_n):
        raise ValueError("batch and n must match the input tensor shape.")
    return resolved_batch, resolved_n


def load_copy_lib(lib_path, block_dim=BLOCK_DIM):
    lib = load_cdll(lib_path)
    resolved_block_dim = max(1, int(block_dim))

    kernel = load_required_symbol(
        lib,
        "call_kernel",
        [
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ],
    )

    def launch_copy(
        x,
        y,
        batch,
        n,
        *,
        block_dim=resolved_block_dim,
        stream_ptr=None,
    ):
        kernel(
            resolve_launch_block_dim(block_dim, resolved_block_dim),
            resolve_stream_ptr(stream_ptr),
            torch_to_ctypes(x),
            torch_to_ctypes(y),
            int(batch),
            int(n),
        )

    def copy_func(
        x,
        y,
        batch=None,
        n=None,
        block_dim=resolved_block_dim,
        stream_ptr=None,
    ):
        batch, n = validate_copy_tensors(x, y, batch=batch, n=n)
        launch_copy(x, y, batch, n, block_dim=block_dim, stream_ptr=stream_ptr)

    copy_func.block_dim = resolved_block_dim
    copy_func.fast = launch_copy
    return copy_func


__all__ = ["DEFAULT_DEVICE", "load_copy_lib", "validate_copy_tensors"]
