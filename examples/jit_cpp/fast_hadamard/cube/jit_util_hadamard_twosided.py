"""Host side of the two-sided cube-unit Hadamard kernel.

The argument restrictions live HERE, not in the test suite, because the kernel
cannot report any of them. This side also builds the two operand buffers for
the Kronecker split and picks block_dim per launch, so its split, tile and
block tables MUST mirror the kernel's case table -- a disagreement either
sizes the schedule for a tile the kernel is not running, or hands the matmul a
wrong-shaped H and it silently computes garbage.
`test_tile_table_mirrors_the_kernel` is the gate on that.
"""

import ctypes
import functools
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

# The cube path needs N = R * C with both factors 16-aligned, so N >= 256; the
# top is the last row of FAST_HADAMARD_TWOSIDED_CASES.
MIN_TWOSIDED_N = 256
MAX_TWOSIDED_N = 32768

_FACTOR_CACHE: dict = {}

# Samples per tile, mirroring FAST_HADAMARD_TWOSIDED_CASES in the kernel, so a
# launch has ceil(batch / S) tiles to spread over the cores. The tests import it
# to place their tile-boundary batches.
_SAMPLES_PER_TILE = {
    256: 32,
    512: 16,
    1024: 8,
    2048: 4,
    4096: 2,
    8192: 1,
    16384: 1,
    32768: 1,
}

# The kernel picks between S, S/2 and S/4 on the tile count. These two MUST
# mirror PTO_TS_TILES_TALL and PTO_TS_TILES_MID in the kernel: sizing the
# schedule for a tile the kernel is not running picks the wrong block_dim,
# silently, since block_dim can change only the time and not the answer.
_TILES_TALL = 16
_TILES_MID = 6


def twosided_samples_per_tile(batch: int, n: int) -> int:
    """S the kernel will run for this shape, mirroring its dispatch exactly,
    including taking the mid tile untested when S/2 == S/4 (at S == 2 the two
    would otherwise disagree)."""
    tall = _SAMPLES_PER_TILE.get(int(n), 1)
    if tall == 1 or int(batch) >= _TILES_TALL * tall:
        return tall
    mid = max(1, tall // 2)
    if mid == 1 or int(batch) >= _TILES_MID * mid:
        return mid
    return max(1, mid // 2)


# TILES PER CORE, not the core count, is what says whether another core is
# worth adding: the kernel splits by tiles, so the reachable schedules are
# ceil(tiles / j), and this takes j = 1 while there are at most _SOLO_TILES[n]
# tiles and j = 2 after that. The sizes with no deep ring (N >= 16384) gain no
# overlap from pairing a core's tiles, so they want every core they can get.
_SOLO_TILES = {256: 1, 512: 1, 1024: 1, 2048: 1, 4096: 1, 8192: 6, 16384: 24, 32768: 24}
# Only reached if FAST_HADAMARD_TWOSIDED_CASES grows a size this table has not
# been fitted for -- sweep block_dim and add the entry.
_DEFAULT_SOLO_TILES = 1


# Memoised: a pure function of its arguments, on the per-launch path where the
# host is the bottleneck for most shapes. The cache is keyed on the arguments
# only, so _SAMPLES_PER_TILE, the tier thresholds and _SOLO_TILES must stay
# constants -- make any of them settable at run time and this needs a
# cache_clear.
@functools.lru_cache(maxsize=4096)
def twosided_block_dim_schedule(batch: int, n: int, max_block_dim: int) -> int:
    """Pick block_dim for one launch: enough cores to use the part, but not so
    many that each core's tile run is too short to fill its pipes.

    The tile tier moves with the batch, so `tiles` has to be computed from the
    tile the kernel will actually run, not from the tall one.

    block_dim cannot change results, so an explicit block_dim always overrides
    this; test_block_dim_does_not_change_result is the gate on that.
    """
    samples = twosided_samples_per_tile(batch, n)
    tiles = -(-int(batch) // samples)
    if tiles == 0:  # the launcher short-circuits batch 0; this keeps it total
        return 1
    solo = _SOLO_TILES.get(int(n), _DEFAULT_SOLO_TILES)
    wanted = tiles if tiles <= solo else -(-tiles // 2)
    cores = max(1, min(int(max_block_dim), wanted))
    # Drop the cores that would be launched with nothing to do: the kernel
    # splits the batch as tiles_per_core = ceil(tiles / block_num), so every
    # core from ceil(tiles / tiles_per_core) on finds begin >= batch and
    # returns. Re-deriving block_num from that count leaves tiles_per_core --
    # and so every core's range -- bit-identical.
    return -(-tiles // -(-tiles // cores))


def _parse_split_env(raw: str) -> dict:
    """Parse PTO_TS_SPLIT, e.g. "1024:64x16,2048:128x16", into {n: (R, C)}."""
    out = {}
    for item in raw.split(","):
        if not item.strip():
            continue
        n, _, rc = item.partition(":")
        r, _, c = rc.partition("x")
        out[int(n)] = (int(r), int(c))
    return out


# Per-N override of the Kronecker split. The balanced split minimises MACs --
# the cost is N * (R + C) per sample -- but C == 16 is what makes BOTH GM legs
# conversion-free at any S (Layout::kNzGm), so the two objectives compete and
# which wins is a measurement.
#
# MUST mirror FAST_HADAMARD_TWOSIDED_CASES (test_tile_table_mirrors_the_kernel
# is the gate): a disagreement sizes the operand buffers for a split the kernel
# is not running and it computes garbage.
_SPLIT_DEFAULT = {512: (32, 16), 1024: (64, 16)}
_SPLIT = _parse_split_env(os.environ.get("PTO_TS_SPLIT", "")) or dict(_SPLIT_DEFAULT)


def hadamard_split(n: int) -> tuple[int, int]:
    """Kronecker split H_n = H_rows (x) H_cols; balanced unless overridden."""
    if int(n) in _SPLIT:
        return _SPLIT[int(n)]
    log2_n = n.bit_length() - 1
    rows_log2 = log2_n // 2
    return 1 << rows_log2, 1 << (log2_n - rows_log2)


def hadamard_matrix(size: int, device, dtype=torch.float16):
    """Sylvester (natural-order) Hadamard matrix with +-1 entries."""
    h = torch.ones((1, 1), dtype=torch.float32)
    while h.shape[0] < size:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h.to(device=device, dtype=dtype).contiguous()


# Mat-factor of each stage, mirroring PTO_TS_BLOCK in the kernel: H_f is applied
# as H_acc (x) H_mat with H_mat of size min(_BLOCK, f). The two MUST agree or
# the kernel multiplies by a wrong-sized operand and computes garbage.
_BLOCK = int(os.environ.get("PTO_TS_BLOCK", 64))


def hadamard_mat_factor(f: int, block: int | None = None) -> int:
    """Size of the matmul sub-factor of H_f. A factor already at or below the
    block size is not split (M == f), which is the dense case."""
    return min(_BLOCK if block is None else int(block), f)


def _to_nz(m):
    """Row-major (W x W) -> its NZ image: block-column major over 16x16
    row-major fractals. `loadHadamardFactors` reads every operand through a
    `Layout::NZ` view, so every operand must be stored this way.

    At W == 16 this is the identity, so N=256 cannot catch a mistake here; the
    first size that can is N=512, and only on its H_C.
    """
    w = m.shape[0]
    if w % 16 or m.shape[1] != w:
        raise ValueError(f"NZ needs a square 16-aligned operand, got {tuple(m.shape)}")
    return m.view(w // 16, 16, w // 16, 16).permute(2, 0, 1, 3).reshape(-1)


def _factor_buffer(f: int, device, block: int | None = None):
    """H_f, then +H_M and -H_M, flat -- the layout loadHadamardFactors reads.

    Appended rather than substituted, so one buffer feeds either form. The tail
    is 2*M*M <= 32768 halves whatever f is, and is unused when M == f.
    """
    m = hadamard_mat_factor(f, block)
    h_m = hadamard_matrix(m, device)
    return torch.cat(
        (_to_nz(hadamard_matrix(f, device)), _to_nz(h_m), _to_nz(-h_m))
    ).contiguous()


def hadamard_factors(n: int, device, block: int | None = None):
    """Cached (H_rows, H_cols) device operand buffers for the split of H_n.

    `block` MUST match whatever the kernel was compiled with (PTO_TS_BLOCK):
    the buffers carry +-H_M, so a disagreement hands the kernel the wrong-sized
    operand. The env default exists so one setting drives both sides.

    Keyed on the torch device object directly: this sits on the per-call path,
    and stringifying the device costs more than the lookup it feeds.
    """
    b = _BLOCK if block is None else int(block)
    key = (n, device, b)
    cached = _FACTOR_CACHE.get(key)
    if cached is None:
        rows, cols = hadamard_split(n)
        cached = (_factor_buffer(rows, device, b), _factor_buffer(cols, device, b))
        _FACTOR_CACHE[key] = cached
    return cached


def validate_twosided_hadamard_args(x, y):
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError("x and y must be 2D tensors with shape [batch, n].")
    if x.shape != y.shape:
        raise ValueError("x and y must have matching shapes.")
    if x.dtype != torch.float16 or y.dtype != torch.float16:
        raise TypeError("x and y must use torch.float16.")
    if not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("x and y must be contiguous.")
    if x.device != y.device:
        raise ValueError("x and y must be on the same device.")

    n = int(x.shape[1])
    if n & (n - 1):
        raise ValueError(f"n must be a power of two, got {n}.")
    if not MIN_TWOSIDED_N <= n <= MAX_TWOSIDED_N:
        raise ValueError(f"n must be in [{MIN_TWOSIDED_N}, {MAX_TWOSIDED_N}], got {n}.")
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
            ctypes.c_void_p,  # h_left  (H_rows,  R x R)
            ctypes.c_void_p,  # h_right (H_cols,  C x C)
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
        resolved_batch, resolved_n = validate_twosided_hadamard_args(x, y)
        if batch is not None and int(batch) != resolved_batch:
            raise ValueError("batch must match the input tensor shape.")
        if n is not None and int(n) != resolved_n:
            raise ValueError("n must match the input tensor shape.")

        # block_dim is a per-launch argument, not a property of the compiled
        # kernel: the best value moves with the shape.
        launch_block_dim = (
            twosided_block_dim_schedule(resolved_batch, resolved_n, resolved_block_dim)
            if block_dim is None
            else resolve_launch_block_dim(block_dim, resolved_block_dim)
        )

        # Build the operands BEFORE resolving the stream pointer: resolving is
        # what drains torch_npu's task queue, so a hoisted pointer lets the raw
        # ctypes launch jump ahead of the torch ops that build them -- wrong
        # answers on the first call at each N.
        h_left, h_right = hadamard_factors(resolved_n, x.device)
        kernel(
            launch_block_dim,
            resolve_stream_ptr(stream_ptr),
            torch_to_ctypes(x),
            torch_to_ctypes(y),
            torch_to_ctypes(h_left),
            torch_to_ctypes(h_right),
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
