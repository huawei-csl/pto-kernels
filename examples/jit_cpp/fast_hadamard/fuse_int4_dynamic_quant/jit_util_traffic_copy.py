import ctypes

from jit_util_common import (
    DEFAULT_DEVICE,
    jit_compile_with_loader,
    load_cdll,
    load_required_symbol,
    resolve_stream_ptr,
    torch_to_ctypes,
)

TRAFFIC_COPY_ARGTYPES = [
    ctypes.c_uint32,  # blockDim
    ctypes.c_void_p,  # stream
    ctypes.c_void_p,  # src
    ctypes.c_void_p,  # dst
    ctypes.c_uint32,  # byte_count
]


def load_lib(lib_path, block_dim=1):
    lib = load_cdll(lib_path)
    resolved_block_dim = max(1, int(block_dim))
    kernel = load_required_symbol(
        lib,
        "call_traffic_copy_kernel",
        TRAFFIC_COPY_ARGTYPES,
    )

    def traffic_copy_func(
        src, dst, byte_count=None, *, block_dim=resolved_block_dim, stream_ptr=None
    ):
        if src.dtype != dst.dtype:
            raise TypeError("src and dst must have the same dtype.")
        if src.device != dst.device:
            raise ValueError("src and dst must be on the same device.")
        if not src.is_contiguous() or not dst.is_contiguous():
            raise ValueError("src and dst must be contiguous.")
        if src.numel() != dst.numel():
            raise ValueError("src and dst must have the same number of elements.")

        if byte_count is None:
            byte_count = src.numel() * src.element_size()
        expected_bytes = src.numel() * src.element_size()
        if int(byte_count) != expected_bytes:
            raise ValueError(
                f"byte_count must match src/dst storage bytes ({expected_bytes}), got {byte_count}."
            )

        kernel(
            max(1, int(block_dim)),
            resolve_stream_ptr(stream_ptr),
            torch_to_ctypes(src),
            torch_to_ctypes(dst),
            int(byte_count),
        )

    traffic_copy_func.block_dim = resolved_block_dim
    return traffic_copy_func


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
