import ctypes
import math
import os

import torch

from jit_util_hadamard import (
    BLOCK_DIM,
    DEFAULT_DEVICE,
    compile_cpp,
    get_cube_block_dim,
    normalize_npu_device,
    torch_to_ctypes,
)


def _optional_torch_to_ctypes(tensor):
    if tensor is None:
        return None
    return torch_to_ctypes(tensor)


def _validate_group_param_tensor(params, name, x, batch, groups_per_row):
    if params is None:
        return 0
    if params.dtype != torch.float16:
        raise TypeError(f"{name} must use torch.float16.")
    if params.device != x.device:
        raise ValueError(f"{name} must be on the same device as x.")
    if not params.is_contiguous():
        raise ValueError(f"{name} must be contiguous.")

    if params.dim() == 1:
        if params.shape[0] != groups_per_row:
            raise ValueError(f"1D {name} must have shape [num_groups].")
        return 0

    if params.dim() == 2:
        if params.shape[1] != groups_per_row:
            raise ValueError(f"2D {name} must have shape [batch|1, num_groups].")
        if params.shape[0] == 1:
            return 0
        if params.shape[0] == batch:
            return groups_per_row
        raise ValueError(f"2D {name} must have a leading dim of 1 or batch.")

    raise ValueError(f"{name} must be 1D or 2D.")


def _infer_group_size(n, group_size, q_scales):
    if group_size is not None:
        resolved = int(group_size)
        if resolved <= 0 or n % resolved != 0:
            raise ValueError("group_size must be a positive divisor of n.")
        return resolved

    if q_scales is None:
        return n

    groups_per_row = int(q_scales.shape[-1])
    if groups_per_row <= 0 or n % groups_per_row != 0:
        raise ValueError("Could not infer a valid group_size from q_scales.")
    return n // groups_per_row


def load_lib(lib_path, block_dim=BLOCK_DIM):
    lib_path = os.path.abspath(lib_path)
    lib = ctypes.CDLL(lib_path)
    resolved_block_dim = max(1, int(block_dim))

    if not hasattr(lib, "call_fused_kernel"):
        raise AttributeError(
            "Could not find call_fused_kernel in the compiled library."
        )
    kernel = lib.call_fused_kernel
    kernel.argtypes = [
        ctypes.c_uint32,  # blockDim
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # x
        ctypes.c_void_p,  # y
        ctypes.c_void_p,  # group_scales
        ctypes.c_void_p,  # group_offsets
        ctypes.c_uint32,  # scale_group_stride
        ctypes.c_uint32,  # offset_group_stride
        ctypes.c_uint32,  # batch
        ctypes.c_uint32,  # n
        ctypes.c_uint32,  # log2_n
        ctypes.c_float,  # scale
        ctypes.c_uint32,  # group_size
        ctypes.c_float,  # q_offset
    ]
    kernel.restype = None

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
        if x.dim() != 2 or y.dim() != 2:
            raise ValueError("x and y must be 2D tensors.")
        if x.shape != y.shape:
            raise ValueError("x and y must have matching shapes.")

        batch = x.shape[0] if batch is None else int(batch)
        n = x.shape[1] if n is None else int(n)
        if batch != x.shape[0] or n != x.shape[1]:
            raise ValueError("batch and n must match the input tensor shape.")
        if log2_n is None:
            log2_n = int(math.log2(n))
        else:
            log2_n = int(log2_n)

        if stream_ptr is None:
            stream = torch.npu.current_stream()
            stream_ptr = getattr(stream, "_as_parameter_", None)
        launch_block_dim = resolved_block_dim if block_dim is None else int(block_dim)

        resolved_group_size = _infer_group_size(n, group_size, q_scales)
        groups_per_row = n // resolved_group_size
        scale_group_stride = _validate_group_param_tensor(
            q_scales, "q_scales", x, batch, groups_per_row
        )
        offset_group_stride = _validate_group_param_tensor(
            q_offsets, "q_offsets", x, batch, groups_per_row
        )

        kernel(
            launch_block_dim,
            stream_ptr,
            torch_to_ctypes(x),
            torch_to_ctypes(y),
            _optional_torch_to_ctypes(q_scales),
            _optional_torch_to_ctypes(q_offsets),
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
):
    resolved_device = normalize_npu_device(device)
    block_dim = get_cube_block_dim(resolved_device)
    lib_path = compile_cpp(src_path, verbose=verbose, so_dir=so_dir)
    func = load_lib(lib_path, block_dim=block_dim)
    if clean_up:
        os.remove(lib_path)
    return func
