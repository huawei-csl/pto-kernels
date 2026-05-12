from jit_util_common import (
    BLOCK_DIM,
    DEFAULT_DEVICE,
    FUSED_HADAMARD_QUANT_ARGTYPES,
    jit_compile_with_loader,
    load_cdll,
    load_required_symbol,
    make_fused_hadamard_quant_func,
)


def load_lib(lib_path, block_dim=BLOCK_DIM):
    lib = load_cdll(lib_path)
    resolved_block_dim = max(1, int(block_dim))

    kernel = load_required_symbol(
        lib,
        "call_fused_kernel",
        FUSED_HADAMARD_QUANT_ARGTYPES,
    )
    return make_fused_hadamard_quant_func(kernel, resolved_block_dim)


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
