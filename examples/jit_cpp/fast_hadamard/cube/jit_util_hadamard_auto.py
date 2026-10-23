"""One Hadamard entry point that branches on N to the right cube kernel.

Two cube-unit implementations exist in this example, and which one applies is
decided almost entirely by N:

  - `cube/fast_hadamard_onesided.cpp`  Y = X * H_N as a single right
        multiplication, H_N = H_B (x) H_M applied as fixed-size matmuls
        accumulated in L0C. Costs N*N MACs per sample but touches each pipe
        once per tile. Runs N in [64, 512].
  - `cube/fast_hadamard_twosided.cpp`  Y = H_R * X * H_C, the balanced
        Kronecker split. Costs only N*(R+C) MACs per sample but needs a
        turnaround between the two multiplies. Runs N in [256, 32768], since
        it needs N = R * C with both factors 16-aligned.

The one-sided product's MAC count grows 4x per doubling of N against 2x for its
bytes, so the split takes over as soon as it can run at all. What is left to
the one-sided kernel is N < 256, which the split cannot reach, and one corner
of N=256, where both kernels are already at the memory roofline and the
one-sided store pattern is the better of the two.

The vector-unit butterfly in `standard/` is not a branch here. It stays where
it is: the baseline these kernels are measured against, and the only one of
the three that runs N below 64.

Supported N is the union of the two cube kernels: powers of two in
[64, 32768]. The grid this example holds itself to,
`bench_common.BENCH_HIDDEN_DIMS` = [128 .. 16384], sits strictly inside that,
with N=64 reachable only by the one-sided kernel and N=32768 only by the
two-sided one.
"""

from pathlib import Path

from cube.jit_util_hadamard_onesided import (
    MAX_ONESIDED_N,
    MIN_ONESIDED_N,
    jit_compile as jit_compile_onesided,
)
from cube.jit_util_hadamard_twosided import (
    MAX_TWOSIDED_N,
    MIN_TWOSIDED_N,
    jit_compile as jit_compile_twosided,
)
from jit_util_common import DEFAULT_DEVICE, normalize_npu_device

MIN_AUTO_N = MIN_ONESIDED_N  # 64
MAX_AUTO_N = MAX_TWOSIDED_N  # 32768

# Only N >= 256 factors as R * C with both 16-aligned, so below this the
# one-sided kernel is the only one that runs -- not a crossover, a capability.
TWOSIDED_MIN_N = MIN_TWOSIDED_N  # 256

# Above this the one-sided kernel refuses: its N*N MACs per sample grow 4x per
# doubling against 2x for the bytes.
ONESIDED_MAX_N = MAX_ONESIDED_N  # 512

# N=256 is the one size where the batch moves the choice. Both kernels reach
# the memory roofline there, so the MAC count stops deciding and the one-sided
# kernel's cheaper store pattern can show; from this batch up it does. Below
# it, and at every other size both kernels run, the split wins.
ONESIDED_BATCH_MIN = {256: 32768}


def supported_auto_sizes() -> list[int]:
    """Every N this dispatcher accepts, ascending."""
    sizes, n = [], MIN_AUTO_N
    while n <= MAX_AUTO_N:
        sizes.append(n)
        n *= 2
    return sizes


def pick_implementation(n: int, batch: int | None = None) -> str:
    """'onesided' or 'twosided' -- the faster kernel at this N (and batch).

    N decides everywhere except N=256, where both kernels are memory-bound and
    the one-sided one edges ahead at very large batch; see ONESIDED_BATCH_MIN.
    Omitting `batch` keeps the small-batch answer, which is the common one.
    """
    n = int(n)
    if not MIN_AUTO_N <= n <= MAX_AUTO_N:
        raise ValueError(
            f"n must be in [{MIN_AUTO_N}, {MAX_AUTO_N}]; got {n}. Below "
            f"{MIN_AUTO_N} only the vector kernel in standard/ applies; above "
            f"{MAX_AUTO_N} no kernel here does -- the two-sided split would "
            "need an accumulator larger than L0C."
        )
    if n & (n - 1):
        raise ValueError(f"n must be a power of two; got {n}.")
    if n < TWOSIDED_MIN_N:
        return "onesided"
    if n > ONESIDED_MAX_N:
        return "twosided"
    floor = ONESIDED_BATCH_MIN.get(n)
    if floor is not None and batch is not None and int(batch) >= floor:
        return "onesided"
    return "twosided"


def jit_compile(
    base_dir=None,
    verbose: bool = True,
    clean_up: bool = False,
    device: str | int = DEFAULT_DEVICE,
):
    """Compile on demand and return one `fn(x, y)` that routes on N.

    Each implementation is compiled the first time an N that needs it is seen,
    not up front: both are cube-arch kernels but they are separate translation
    units, and a caller that only ever uses one size should only pay for one
    compile.
    """
    resolved_device = normalize_npu_device(device)
    base = Path(base_dir) if base_dir else Path(__file__).resolve().parents[1]
    builders = {
        "onesided": (
            jit_compile_onesided,
            base / "cube" / "fast_hadamard_onesided.cpp",
        ),
        "twosided": (
            jit_compile_twosided,
            base / "cube" / "fast_hadamard_twosided.cpp",
        ),
    }
    compiled: dict = {}

    def kernel_for(impl: str):
        if impl not in compiled:
            build, src = builders[impl]
            compiled[impl] = build(
                str(src),
                verbose=verbose,
                clean_up=clean_up,
                device=resolved_device,
            )
        return compiled[impl]

    def hadamard_func(
        x,
        y,
        batch=None,
        n=None,
        block_dim=None,
        stream_ptr=None,
    ):
        """Y = X * H_N, out of place, for 2D contiguous fp16 [batch, n].

        Both back ends take this signature and validate the tensors
        themselves, so the only thing done here is the choice between them.
        Each applies its own block_dim schedule unless one is passed.
        """
        if x.dim() != 2:
            raise ValueError(
                f"x must be 2D with shape [batch, n]; got {tuple(x.shape)}."
            )
        resolved_n = int(x.shape[1])
        impl = pick_implementation(resolved_n, int(x.shape[0]))
        return kernel_for(impl)(
            x,
            y,
            batch=batch,
            n=n,
            block_dim=block_dim,
            stream_ptr=stream_ptr,
        )

    hadamard_func.pick_implementation = pick_implementation
    hadamard_func.compiled = compiled
    hadamard_func.kernel_for = kernel_for
    return hadamard_func


# Re-exported so callers can check a size without compiling anything.
__all__ = [
    "MIN_AUTO_N",
    "MAX_AUTO_N",
    "ONESIDED_MAX_N",
    "ONESIDED_BATCH_MIN",
    "TWOSIDED_MIN_N",
    "MIN_TWOSIDED_N",
    "jit_compile",
    "pick_implementation",
    "supported_auto_sizes",
]
