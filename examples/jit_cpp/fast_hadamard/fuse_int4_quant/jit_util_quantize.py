import ctypes

import torch

from jit_util_common import (
    BLOCK_DIM,
    DEFAULT_DEVICE,
    jit_compile_with_loader,
    load_cdll,
    load_required_symbol,
    resolve_launch_block_dim,
    resolve_stream_ptr,
    torch_to_ctypes,
)


def _validate_packed_int4_io(x, y):
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError("x and y must be 2D tensors.")
    if x.dtype != torch.float16:
        raise TypeError("x must use torch.float16.")
    if y.dtype != torch.int8:
        raise TypeError("y must use torch.int8 packed-byte storage.")
    if x.device != y.device:
        raise ValueError("x and y must be on the same device.")
    if not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("x and y must be contiguous.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same batch size.")
    if x.shape[1] <= 0 or (x.shape[1] & 1):
        raise ValueError("x.shape[1] must be a positive even integer.")
    if y.shape[1] != x.shape[1] // 2:
        raise ValueError("Packed int4 y must have shape [batch, x.shape[1] // 2].")


def load_lib(lib_path, block_dim=None):
    lib = load_cdll(lib_path)
    resolved_block_dim = max(1, int(BLOCK_DIM if block_dim is None else block_dim))

    kernel = load_required_symbol(
        lib,
        "call_quantize_kernel",
        [
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_float,
        ],
    )

    def quantize_func(x, y, scale, block_dim=resolved_block_dim, stream_ptr=None):
        _validate_packed_int4_io(x, y)
        kernel(
            resolve_launch_block_dim(block_dim, resolved_block_dim),
            resolve_stream_ptr(stream_ptr),
            torch_to_ctypes(x),
            torch_to_ctypes(y),
            x.shape[0],
            x.shape[1],
            float(scale),
        )

    quantize_func.block_dim = resolved_block_dim
    return quantize_func


def jit_compile(
    src_path,
    verbose=True,
    clean_up=False,
    so_dir=None,
    device: str | int = DEFAULT_DEVICE,
    block_dim=None,
):
    return jit_compile_with_loader(
        src_path,
        load_lib,
        verbose=verbose,
        clean_up=clean_up,
        so_dir=so_dir,
        device=device,
        block_dim=block_dim,
    )
