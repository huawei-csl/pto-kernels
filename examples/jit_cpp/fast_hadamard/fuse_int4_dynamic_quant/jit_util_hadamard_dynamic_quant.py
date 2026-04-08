import ctypes
import math

import torch

from jit_util_common import (
    DEFAULT_DEVICE,
    jit_compile_with_loader,
    load_cdll,
    load_required_symbol,
    resolve_stream_ptr,
    torch_to_ctypes,
)

DYNAMIC_QUANT_ARGTYPES = [
    ctypes.c_uint32,  # blockDim
    ctypes.c_void_p,  # stream
    ctypes.c_void_p,  # x (fp16 scratch, in-place hadamard)
    ctypes.c_void_p,  # y (packed int4 output)
    ctypes.c_void_p,  # row_scales (float32 output, one per row)
    ctypes.c_uint32,  # batch
    ctypes.c_uint32,  # full_n
    ctypes.c_uint32,  # hadamard_n
    ctypes.c_uint32,  # log2_hadamard_n
    ctypes.c_float,  # inv_sqrt_hadamard_n
]

DEFAULT_BLOCK_DIM = 4


def load_lib(lib_path, block_dim=DEFAULT_BLOCK_DIM):
    lib = load_cdll(lib_path)
    resolved_block_dim = max(1, int(block_dim))

    kernel = load_required_symbol(
        lib,
        "call_dynamic_quant_kernel",
        DYNAMIC_QUANT_ARGTYPES,
    )

    def dynamic_quant_func(
        x,
        y,
        row_scales,
        batch=None,
        full_n=None,
        hadamard_n=None,
        log2_hadamard_n=None,
        *,
        block_dim=resolved_block_dim,
        stream_ptr=None,
    ):
        if x.dim() != 2 or y.dim() != 2:
            raise ValueError("x and y must be 2D tensors.")
        if x.dtype != torch.float16:
            raise TypeError("x must be torch.float16.")
        if y.dtype != torch.int8:
            raise TypeError("y must be torch.int8 (packed int4).")
        if row_scales.dtype != torch.float32:
            raise TypeError("row_scales must be torch.float32.")

        if batch is None:
            batch = x.shape[0]
        if full_n is None:
            full_n = x.shape[1]
        if hadamard_n is None:
            hadamard_n = full_n
        if log2_hadamard_n is None:
            log2_hadamard_n = int(math.log2(hadamard_n))

        if hadamard_n & (hadamard_n - 1) != 0:
            raise ValueError(f"hadamard_n must be a power of two, got {hadamard_n}")
        if full_n % hadamard_n != 0:
            raise ValueError(
                f"full_n ({full_n}) must be divisible by hadamard_n ({hadamard_n})"
            )

        kernel(
            max(1, int(block_dim)),
            resolve_stream_ptr(stream_ptr),
            torch_to_ctypes(x),
            torch_to_ctypes(y),
            torch_to_ctypes(row_scales),
            batch,
            full_n,
            hadamard_n,
            log2_hadamard_n,
            1.0 / math.sqrt(float(hadamard_n)),
        )

    dynamic_quant_func.block_dim = resolved_block_dim
    return dynamic_quant_func


def jit_compile(
    src_path,
    verbose=True,
    clean_up=False,
    so_dir=None,
    device=DEFAULT_DEVICE,
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
