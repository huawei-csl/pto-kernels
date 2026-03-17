import ctypes

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


def load_lib(lib_path, block_dim=None):
    lib = load_cdll(lib_path)
    resolved_block_dim = max(1, int(BLOCK_DIM if block_dim is None else block_dim))

    kernel = load_required_symbol(
        lib,
        "call_quantize_kernel",
        [
            ctypes.c_uint32,  # blockDim
            ctypes.c_void_p,  # stream
            ctypes.c_void_p,  # x
            ctypes.c_void_p,  # y
            ctypes.c_uint32,  # batch
            ctypes.c_uint32,  # n
            ctypes.c_float,  # scale
        ],
    )

    def quantize_func(x, y, scale, block_dim=resolved_block_dim, stream_ptr=None):
        kernel(
            resolve_launch_block_dim(block_dim, resolved_block_dim),
            resolve_stream_ptr(stream_ptr),
            torch_to_ctypes(x),
            torch_to_ctypes(y),
            x.shape[0],
            x.shape[1],
            scale,
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
