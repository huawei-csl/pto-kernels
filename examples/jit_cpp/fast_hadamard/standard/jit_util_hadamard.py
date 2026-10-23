"""Host side of the vector-unit Hadamard kernel.

The argument restrictions live HERE, not in the test suite, because the kernel
cannot report any of them. Its only guard is `n == 0 || n > ELEMENTS_PER_TILE`,
and that one returns silently with the input untouched; everything else fails
quietly and wrongly. `log2_n` in particular is passed in rather than derived, so
a wrong value makes the specialised dispatch fall through to the generic body
and run the wrong number of butterfly stages, with no diagnostic at all. A
non-power-of-two `n` likewise runs the generic body and returns garbage, because
the even/odd gather assumes a halvable row.
"""

import ctypes
import math

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


# One row is transformed inside a single 32 KB UB tile, so the whole transform
# length has to fit there: X_BUFFER_BYTES / sizeof(half). Mirrors
# ELEMENTS_PER_TILE in fast_hadamard.cpp -- keep the two in step.
MAX_HADAMARD_N = 16 * 1024

# The kernel carries a specialised body per N in this set, which butterflies a
# whole multi-sample tile at once; every other power of two falls back to a
# correct but slower row-at-a-time body. The tested grid is 128..16384.
SPECIALISED_HADAMARD_N = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)

# Any batch is accepted, including 0: the kernel splits by samples, lets cores
# with no work return, and chunks each core's share into tiles of
# ELEMENTS_PER_TILE / n samples with a ragged tail.


def validate_hadamard_args(x, batch, n, log2_n):
    """Check everything the kernel cannot, and normalise to ints."""
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.device.type != "npu":
        raise ValueError(f"x must be on an NPU device, got {x.device}")
    if x.dtype != torch.float16:
        raise TypeError(f"x must have dtype torch.float16, got {x.dtype}")
    if x.dim() != 2:
        raise ValueError(f"x must be 2D with shape [batch, n], got {tuple(x.shape)}")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")

    batch = int(batch)
    n = int(n)
    log2_n = int(log2_n)

    if batch < 0:
        raise ValueError(f"batch must be >= 0, got {batch}")
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    if n > MAX_HADAMARD_N:
        raise ValueError(f"n must be <= {MAX_HADAMARD_N}, got {n}")
    if x.shape != (batch, n):
        raise ValueError(f"x shape {tuple(x.shape)} must match batch={batch} and n={n}")
    if n & (n - 1):
        raise ValueError(f"n must be a power of two, got {n}")

    expected_log2_n = int(math.log2(n))
    if log2_n != expected_log2_n:
        raise ValueError(
            f"log2_n must equal int(log2(n))={expected_log2_n}, got {log2_n}"
        )

    return batch, n, log2_n


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
        batch=None,
        n=None,
        log2_n=None,
        block_dim=resolved_block_dim,
        stream_ptr=None,
    ):
        """Transform x IN PLACE: x <- x * H_n, for 2D contiguous fp16.

        batch, n and log2_n are derived from x when not given; passing them is
        supported so a caller can assert what it thinks it is transforming.
        """
        if batch is None or n is None:
            if x.dim() != 2:
                raise ValueError(
                    f"x must be 2D with shape [batch, n], got {tuple(x.shape)}"
                )
            batch = x.shape[0] if batch is None else batch
            n = x.shape[1] if n is None else n
        if log2_n is None:
            log2_n = int(n).bit_length() - 1

        batch, n, log2_n = validate_hadamard_args(x, batch, n, log2_n)
        if batch == 0:
            return

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
