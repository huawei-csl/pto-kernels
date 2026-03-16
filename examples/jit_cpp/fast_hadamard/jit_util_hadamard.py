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


def load_lib(lib_path, block_dim=BLOCK_DIM):
    lib = load_cdll(lib_path)
    resolved_block_dim = max(1, int(block_dim))

    kernel = load_required_symbol(
        lib,
        "call_kernel",
        [
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ],
    )

    def hadamard_func(
        x,
        batch,
        n,
        log2_n,
        block_dim=resolved_block_dim,
        stream_ptr=None,
    ):
        kernel(
            resolve_launch_block_dim(block_dim, resolved_block_dim),
            resolve_stream_ptr(stream_ptr),
            torch_to_ctypes(x),
            batch,
            n,
            log2_n,
        )

    hadamard_func.block_dim = resolved_block_dim
    return hadamard_func


def jit_compile(
    src_path,
    verbose=True,
    clean_up=False,
    so_dir=None,
    device: str | int = DEFAULT_DEVICE,
):
    return jit_compile_with_loader(
        src_path,
        load_lib,
        verbose=verbose,
        clean_up=clean_up,
        so_dir=so_dir,
        device=device,
    )
