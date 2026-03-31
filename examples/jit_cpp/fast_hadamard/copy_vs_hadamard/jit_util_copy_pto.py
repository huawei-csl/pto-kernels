from jit_util_common import DEFAULT_DEVICE, jit_compile_with_loader

from copy_vs_hadamard.jit_util_copy_common import load_copy_lib


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
        load_copy_lib,
        verbose=verbose,
        clean_up=clean_up,
        so_dir=so_dir,
        device=device,
        block_dim=block_dim,
    )
