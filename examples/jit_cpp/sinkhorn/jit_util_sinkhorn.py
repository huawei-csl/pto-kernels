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

MAX_DIM = 128

SINKHORN_DS_ARGTYPES = [
    ctypes.c_uint32,  # blockDim
    ctypes.c_void_p,  # stream
    ctypes.c_void_p,  # input
    ctypes.c_void_p,  # output
    ctypes.c_uint32,  # N
    ctypes.c_uint32,  # K
    ctypes.c_uint32,  # repeat
    ctypes.c_float,  # eps
]


def _validate(input_tensor, output_tensor, K):
    if input_tensor.dim() != 3:
        raise ValueError("input must be 3D (N, K, K).")
    if input_tensor.shape[1] != K or input_tensor.shape[2] != K:
        raise ValueError(f"input must have shape (N, {K}, {K}).")
    if output_tensor.shape != input_tensor.shape:
        raise ValueError("output must have the same shape as input.")
    if input_tensor.dtype != torch.float16:
        raise TypeError("input must use torch.float16.")
    if not input_tensor.is_contiguous() or not output_tensor.is_contiguous():
        raise ValueError("tensors must be contiguous.")
    if input_tensor.device != output_tensor.device:
        raise ValueError("tensors must be on the same device.")
    if K > MAX_DIM:
        raise ValueError(f"K must be <= {MAX_DIM}.")


def load_lib(lib_path, block_dim=BLOCK_DIM):
    lib = load_cdll(lib_path)
    resolved_block_dim = max(1, int(block_dim))

    kernel = load_required_symbol(
        lib,
        "call_sinkhorn_ds_kernel",
        SINKHORN_DS_ARGTYPES,
    )

    def sinkhorn_ds_func(
        input_tensor,
        output_tensor,
        *,
        repeat=10,
        eps=1e-6,
        block_dim=resolved_block_dim,
        stream_ptr=None,
    ):
        N, K, _ = input_tensor.shape
        _validate(input_tensor, output_tensor, K)
        kernel(
            resolve_launch_block_dim(block_dim, resolved_block_dim),
            resolve_stream_ptr(stream_ptr),
            torch_to_ctypes(input_tensor),
            torch_to_ctypes(output_tensor),
            N,
            K,
            repeat,
            float(eps),
        )

    sinkhorn_ds_func.block_dim = resolved_block_dim
    return sinkhorn_ds_func


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
