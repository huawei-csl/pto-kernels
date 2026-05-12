import torch

from jit_util_common import (
    BLOCK_DIM,
    DEFAULT_DEVICE,
    FUSED_HADAMARD_QUANT_ARGTYPES,
    jit_compile_with_loader,
    load_cdll,
    load_required_symbol,
    optional_torch_to_ctypes,
    resolve_grouped_quant_config,
    resolve_hadamard_call_shape,
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


def _validate_packed_int4_output_shape(y, batch, n):
    if (n & 1) != 0:
        raise ValueError("n must be even for packed int4 output.")
    if y.shape[0] != batch or y.shape[1] != n // 2:
        raise ValueError("Packed int4 y must have shape [batch, n // 2].")


def load_lib(lib_path, block_dim=BLOCK_DIM):
    lib = load_cdll(lib_path)
    resolved_block_dim = max(1, int(block_dim))

    kernel = load_required_symbol(
        lib,
        "call_fused_kernel",
        FUSED_HADAMARD_QUANT_ARGTYPES,
    )

    def fused_func(
        x,
        y,
        batch=None,
        n=None,
        log2_n=None,
        scale=1.0,
        *,
        group_size=None,
        q_offset=0.0,
        q_scales=None,
        q_offsets=None,
        block_dim=resolved_block_dim,
        stream_ptr=None,
    ):
        _validate_packed_int4_io(x, y)
        batch, n, log2_n = resolve_hadamard_call_shape(
            x,
            batch=batch,
            n=n,
            log2_n=log2_n,
        )
        _validate_packed_int4_output_shape(y, batch, n)
        resolved_group_size, scale_group_stride, offset_group_stride = (
            resolve_grouped_quant_config(
                x,
                batch,
                n,
                group_size,
                q_scales,
                q_offsets,
            )
        )

        kernel(
            resolve_launch_block_dim(block_dim, resolved_block_dim),
            resolve_stream_ptr(stream_ptr),
            torch_to_ctypes(x),
            torch_to_ctypes(y),
            optional_torch_to_ctypes(q_scales),
            optional_torch_to_ctypes(q_offsets),
            scale_group_stride,
            offset_group_stride,
            batch,
            n,
            log2_n,
            float(scale),
            resolved_group_size,
            float(q_offset),
        )

    fused_func.block_dim = resolved_block_dim
    fused_func.supports_grouped = True
    return fused_func


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
