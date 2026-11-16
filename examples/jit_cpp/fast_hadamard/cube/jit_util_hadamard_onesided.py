"""Host side of the one-sided cube-unit Hadamard kernel.

The argument restrictions live HERE, not in the test suite, because the kernel
cannot report any of them: its dispatch treats an N it has no case for as a
reason to do nothing. This side also builds the +-H_M operand buffer and picks
block_dim per launch, so its tile tables and block size MUST mirror the
kernel's -- a disagreement either sizes the schedule for a tile the kernel is
not running, or hands the matmul a wrong-sized operand and it silently
computes garbage.
"""

import ctypes
import functools
import math
import os

import torch

from jit_util_common import (
    CUBE_ARCH,
    DEFAULT_DEVICE,
    get_cube_block_dim,
    jit_compile_with_loader,
    load_cdll,
    load_required_symbol,
    normalize_npu_device,
    resolve_launch_block_dim,
    resolve_stream_ptr,
    torch_to_ctypes,
)

# Y = X * H_N as a single right-multiplication, with H_N factored
# H_B (x) H_M and applied as fixed-size M x M matmuls accumulated in L0C.
# N must be 16-aligned for the cube. 512 is a COMPUTE crossover, not a capacity
# limit -- see the kernel header and validate_onesided_hadamard_args.
MIN_ONESIDED_N = 64
MAX_ONESIDED_N = 512

# PTO requires 16-aligned tile rows (TMOV's TExtract rejects even Rows == 1),
# so the kernel's smallest tile is 16 samples. The TRANSFERS around a tile need
# not be, so no batch is padded: a short span runs in one masked tile. Nothing
# here depends on this value; the tests place batches either side of it.
MIN_ONESIDED_BATCH = 16

_H_CACHE: dict = {}

# Samples per tile, mirroring FAST_HADAMARD_ONESIDED_CASES in the kernel. Only
# N <= 128 follows S * N == PTO_OS_TILE_ELEMS; 256 and 512 run the blocked form
# and override it with a SMALLER tile (see PTO_OS_S512). Keep in step with the
# kernel -- the tests import this to place their tile-boundary batches.
_TILE_ELEMS = 16384
_SAMPLES_PER_TILE = {64: 256, 128: 128, 256: 32, 512: 16}

# The kernel carries a second, shorter tile for the resident sizes and picks it
# when the tall one would not yield PTO_OS_TILES_TO_PIPELINE tiles -- below
# that the tile height, not the pipeline, is what caps the core count. These
# three MUST mirror PTO_OS_S64_SHALLOW, PTO_OS_S128_SHALLOW and
# PTO_OS_TILES_TO_PIPELINE, or the schedule sizes a tile the kernel is not
# running and picks the wrong block_dim.
_SAMPLES_PER_TILE_SHALLOW = {64: 128, 128: 64, 256: 32, 512: 16}
_TILES_TO_PIPELINE = 32


def onesided_samples_per_tile(n: int, batch: int) -> int:
    """S the kernel will run for this shape, mirroring its dispatch."""
    tall = _SAMPLES_PER_TILE.get(int(n), 1)
    if int(batch) >= _TILES_TO_PIPELINE * tall:
        return tall
    return _SAMPLES_PER_TILE_SHALLOW.get(int(n), tall)


# bd = sqrt(k * tiles) trades core count against keeping each core's pipes
# full, with k fitted per (N, tile). Calibration, not a rule -- resweep it
# whenever the tile geometry, the crossover or PTO_OS_BLOCK changes.
_TILE_RUN_K = {64: 7.51, 128: 7.51, 256: 15.02, 512: 23.02}
# Same fit for the shorter tile. The blocked sizes have one tile, so both
# tables agree for them.
_TILE_RUN_K_SHALLOW = {64: 5.52, 128: 7.51, 256: 15.02, 512: 23.02}
# Only reached if the case table grows a size this has not been fitted for --
# sweep block_dim and add the entry.
_DEFAULT_TILE_RUN_K = 8.0


# Memoised: a pure function of its arguments, on the per-launch path where the
# host is the bottleneck for most shapes. The cache is keyed on the arguments
# only, so the tile tables and _TILE_RUN_K must stay constants -- make either
# settable at run time and this needs a cache_clear.
@functools.lru_cache(maxsize=4096)
def onesided_block_dim_schedule(batch: int, n: int, max_block_dim: int) -> int:
    """Pick block_dim for one launch: enough cores to use the part, but not so
    many that each core's tile run is too short to fill its pipes.

    block_dim cannot change results, so an explicit block_dim always overrides
    this; test_block_dim_does_not_change_result is the gate on that.
    """
    samples = onesided_samples_per_tile(n, batch)
    tiles = -(-int(batch) // samples)
    if tiles == 0:  # the launcher short-circuits batch 0; this keeps it total
        return 1
    table = (
        _TILE_RUN_K
        if samples == _SAMPLES_PER_TILE.get(int(n), 1)
        else _TILE_RUN_K_SHALLOW
    )
    k = table.get(int(n), _DEFAULT_TILE_RUN_K)
    cores = max(1, min(int(max_block_dim), round(math.sqrt(k * tiles))))
    # Drop the cores that would be launched with nothing to do: the kernel
    # splits the batch as tiles_per_core = ceil(tiles / block_num), so every
    # core from ceil(tiles / tiles_per_core) on finds begin >= batch and
    # returns. Re-deriving block_num from that count leaves tiles_per_core --
    # and so every core's range -- bit-identical.
    return -(-tiles // -(-tiles // cores))


def hadamard_matrix(size: int, device, dtype=torch.float16):
    """Sylvester (natural-order) Hadamard matrix with +-1 entries."""
    h = torch.ones((1, 1), dtype=torch.float32)
    while h.shape[0] < size:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h.to(device=device, dtype=dtype).contiguous()


# Block size M of the kernel's blocked-accumulation form, mirroring
# PTO_OS_BLOCK. The buffer below carries +-H_M for it, so the two MUST agree:
# a mismatch hands the kernel a wrong-sized operand and it computes garbage.
_BLOCK = int(os.environ.get("PTO_OS_BLOCK", 64))


def hadamard_operands(n: int, block: int | None = None) -> tuple[int, int]:
    """(M, B) of the kernel's blocked form: H_n = H_B (x) H_M, M = min(64, n)."""
    m = min(_BLOCK if block is None else int(block), n)
    return m, n // m


def hadamard_full(n: int, device, block: int | None = None):
    """Cached device operand buffer: H_n, then H_M, then -H_M, flat.

    H_n is symmetric, so the kernel uses it as the right operand with no
    transpose. The +-H_M pair at offset n*n is what the blocked form multiplies
    by -- with H_n = H_B (x) H_M and H_B's entries +-1, those two are the only
    right operands it needs -- and is APPENDED rather than substituted so one
    buffer feeds either form. The tail is 2*M*M <= 32768 halves whatever n is.

    Keyed on the torch device object directly: this sits on the per-call path,
    and stringifying the device costs more than the lookup it feeds.
    """
    key = (n, device, block)
    cached = _H_CACHE.get(key)
    if cached is None:
        m, _ = hadamard_operands(n, block)
        h_m = hadamard_matrix(m, device)
        cached = torch.cat(
            (hadamard_matrix(n, device).view(-1), h_m.view(-1), -h_m.view(-1))
        ).contiguous()
        _H_CACHE[key] = cached
    return cached


def clear_operand_cache():
    """Drop every cached operand buffer, so the next call rebuilds one.

    Only a caller that needs a COLD build has any use for this: the buffers are
    pure functions of (n, device, block) and are never invalidated otherwise.
    Building one enqueues torch ops, which is the window the stream pointer has
    to be resolved after, so this is what lets that ordering be tested.
    """
    _H_CACHE.clear()


def validate_onesided_hadamard_args(x, y):
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError("x and y must be 2D tensors with shape [batch, n].")
    if x.shape != y.shape:
        raise ValueError("x and y must have matching shapes.")
    if x.dtype != torch.float16 or y.dtype != torch.float16:
        raise TypeError("x and y must use torch.float16.")
    if not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("x and y must be contiguous.")

    n = int(x.shape[1])
    if n < MIN_ONESIDED_N or n > MAX_ONESIDED_N:
        raise ValueError(
            f"n must be in [{MIN_ONESIDED_N}, {MAX_ONESIDED_N}]; got {n}. "
            "The one-sided product costs n*n MACs per sample however H is "
            "factored, so past 512 the MACs grow 4x per doubling while the "
            "bytes only double -- use the two-sided kernel in cube/, which is "
            "1.7x faster at 1024 and 4.8x at 2048."
        )
    if n & (n - 1):
        raise ValueError(f"n must be a power of two; got {n}.")
    return int(x.shape[0]), n


def load_lib(lib_path, block_dim=None, device: str | int = DEFAULT_DEVICE):
    lib = load_cdll(lib_path)
    resolved_block_dim = max(
        1, int(block_dim) if block_dim is not None else get_cube_block_dim(device)
    )

    kernel = load_required_symbol(
        lib,
        "call_kernel",
        [
            ctypes.c_uint32,  # blockDim
            ctypes.c_void_p,  # stream
            ctypes.c_void_p,  # x
            ctypes.c_void_p,  # y
            ctypes.c_void_p,  # h  (N x N, row-major)
            ctypes.c_uint32,  # batch
            ctypes.c_uint32,  # n
        ],
    )

    def hadamard_func(
        x,
        y,
        batch=None,
        n=None,
        block_dim=None,
        stream_ptr=None,
    ):
        resolved_batch, resolved_n = validate_onesided_hadamard_args(x, y)
        if batch is not None and int(batch) != resolved_batch:
            raise ValueError("batch must match the input tensor shape.")
        if n is not None and int(n) != resolved_n:
            raise ValueError("n must match the input tensor shape.")

        if resolved_batch == 0:
            return

        launch_block_dim = (
            onesided_block_dim_schedule(resolved_batch, resolved_n, resolved_block_dim)
            if block_dim is None
            else resolve_launch_block_dim(block_dim, resolved_block_dim)
        )

        # Build H BEFORE resolving the stream pointer, on its own line: on a
        # cache miss this enqueues torch ops, and resolve_stream_ptr is what
        # drains torch_npu's task queue so the raw ctypes launch is ordered
        # behind them. Inlined after it, the kernel could read H unwritten.
        h = hadamard_full(resolved_n, x.device)
        kernel(
            launch_block_dim,
            resolve_stream_ptr(stream_ptr),
            torch_to_ctypes(x),
            torch_to_ctypes(y),
            torch_to_ctypes(h),
            resolved_batch,
            resolved_n,
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
    resolved_device = normalize_npu_device(device)
    return jit_compile_with_loader(
        src_path,
        lambda lib_path, block_dim=None: load_lib(
            lib_path, block_dim=block_dim, device=resolved_device
        ),
        verbose=verbose,
        clean_up=clean_up,
        so_dir=so_dir,
        device=resolved_device,
        arch=CUBE_ARCH,
    )
