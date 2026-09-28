"""Build and load the two fused Hadamard + MXFP4 quantize kernels.

Both are in this directory and share fused_hadamard_quant_common.hpp; they
differ in the rotation -- one over the whole row, one over independent
32-element blocks -- and so in which row widths have an instantiation. Pick one
with ``kind``: "full" or "b32".

The butterfly is the UNNORMALISED Sylvester matrix, so its output is sqrt(32)
larger than an orthogonal block Hadamard's. That is left to the caller: MXFP4's
E8M0 scale is a power of two and sqrt(32) is not, so the scale cannot absorb it
and the nibbles would genuinely differ. Scale x by 1/sqrt(32) going in for
orthogonal semantics.
"""

import ctypes
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
import torch_npu  # noqa

HERE = Path(__file__).resolve().parent
BUILDDIR = HERE / "build"
COMMON_HEADER = HERE / "fused_hadamard_quant_common.hpp"

MX_BLOCK = 32
VECTOR_CORES = 64  # vector cores on an A5

# The rotation is order K for "full", so K must be a power of two. For "b32" it
# is always 32 wide, which frees K from that -- but not from RowsFor, which
# needs Rows*K to be a whole 1024-element grain, so 11008 is absent. Both lists
# must match SUPPORTED_K in the matching .cpp.
SUPPORTED_K = {
    "full": (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384),
    "b32": (
        32,
        64,
        96,
        128,
        192,
        256,
        512,
        768,
        896,
        1024,
        1152,
        1280,
        1408,
        1536,
        1664,
        1792,
        2048,
        2560,
        2816,
        3072,
        3584,
        4096,
        5120,
        6144,
        7168,
        8192,
        14336,
        16384,
    ),
}


@dataclass(frozen=True)
class KernelSpec:
    """What differs between the two kernels.

    ``width_rule`` is appended to the error a bad ``k`` raises, because the two
    reject a width for different reasons and a caller needs to know which.
    """

    source: Path
    lib_stem: str
    launch_symbol: str
    rows_symbol: str
    supported_k: Tuple[int, ...]
    width_rule: str

    @property
    def build_dir(self) -> Path:
        return BUILDDIR

    @property
    def newest_input(self) -> float:
        """mtime of the newest thing the .so is built from.

        The .cpp is not the only input: most of each kernel is in
        COMMON_HEADER, and a cache keyed on the .cpp alone would serve a stale
        .so after a header-only edit, silently.
        """
        return max(self.source.stat().st_mtime, COMMON_HEADER.stat().st_mtime)


SPECS = {
    "full": KernelSpec(
        source=HERE / "fused_hadamard_quant_a5.cpp",
        lib_stem="fused_full",
        launch_symbol="call_hadamard_mxfp4_full",
        rows_symbol="hadamard_mxfp4_full_rows_for",
        supported_k=SUPPORTED_K["full"],
        width_rule="The rotation is row wide, so K must be a power of two.",
    ),
    "b32": KernelSpec(
        source=HERE / "fused_hadamard_quant_b32_a5.cpp",
        lib_stem="fused_b32",
        launch_symbol="call_hadamard_mxfp4_b32",
        rows_symbol="hadamard_mxfp4_b32_rows_for",
        supported_k=SUPPORTED_K["b32"],
        width_rule=(
            "Widths must be a multiple of 32 whose Rows*K is a whole "
            "1024-element grain, and have an instantiation."
        ),
    ),
}


def spec(kind):
    try:
        return SPECS[kind]
    except KeyError:
        raise ValueError(f"kind must be one of {sorted(SPECS)}, got {kind!r}") from None


def _flags(home):
    return (
        f"-xcce --cce-aicore-arch=dav-c310-vec -DREGISTER_BASE "
        f"-std=c++17 -O2 -fPIC -Wno-ignored-attributes -Wno-macro-redefined "
        f"-mllvm -cce-aicore-stack-size=0x8000 "
        f"-mllvm -cce-aicore-function-stack-size=0x8000 "
        f"-mllvm -cce-aicore-addr-transform "
        f"-mllvm -cce-aicore-dcci-insert-for-scalar=false -Xhost-start -Xhost-end "
        f"-I{home}/aarch64-linux/include -I{home}/include"
    ).split()


def _compile(spec, verbose=True, extra_defs=()):
    """Compile the fused kernel to a .so. One .so serves every supported K.

    extra_defs are extra -D tokens for a tuning or A/B variant. They go into the
    .so NAME as well as the command line, so a variant can never be served from
    the default build's cache -- silently timing the wrong binary is the failure
    this guards.
    """
    home = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME")
    if not home:
        raise RuntimeError("source a CANN set_env.sh first: ASCEND_HOME_PATH is unset")
    build_dir = spec.build_dir
    build_dir.mkdir(parents=True, exist_ok=True)
    tag = "".join("_" + d.lstrip("-D").replace("=", "") for d in sorted(extra_defs))
    # Reuse an .so newer than its source. These kernels unroll to hundreds of
    # tile instructions and a rebuild can outlast the task queue's 600 s cap, so
    # recompiling per call is not merely wasteful.
    cached = build_dir / f"{spec.lib_stem}{tag}.so"
    if cached.exists() and cached.stat().st_mtime > spec.newest_input:
        if verbose:
            print("reusing", cached)
        return cached
    obj = build_dir / f"{spec.lib_stem}{tag}.o"
    lib = cached
    for step in (
        [
            f"{home}/bin/bisheng",
            *_flags(home),
            *extra_defs,
            "-c",
            str(spec.source),
            "-o",
            str(obj),
        ],
        [
            f"{home}/bin/bisheng",
            "-fPIC",
            "-shared",
            "--cce-fatobj-link",
            f"-Wl,-soname,{lib.name}",
            str(obj),
            "-o",
            str(lib),
        ],
    ):
        if verbose:
            print("compile:", " ".join(step[:3]), "...")
        subprocess.run(step, check=True)
    return lib


def current_stream_ptr():
    return ctypes.c_void_p(torch.npu.current_stream().npu_stream)


def _load(spec, k=256, verbose=True, extra_defs=()):
    """Return `fused(x) -> (nibbles, scales)` for row width `k`.

    Allocates its outputs, mirroring `torch_npu.npu_dynamic_mx_quant`, so the two
    are comparable on the same call path.

    extra_defs reaches the compiler, so the reduced builds the benchmark's ladder
    needs come from this one source: FUSED_ROTATE_ONLY leaves the butterfly
    alone, FUSED_NO_ROTATE leaves the quantizer alone.
    """
    if k not in spec.supported_k:
        raise ValueError(
            f"K={k} has no instantiation; supported: {sorted(spec.supported_k)}. "
            + spec.width_rule
        )
    lib = ctypes.CDLL(str(_compile(spec, verbose=verbose, extra_defs=extra_defs)))
    launch = getattr(lib, spec.launch_symbol)
    launch.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    launch.restype = None
    rows_for = getattr(lib, spec.rows_symbol)
    rows_for.argtypes = [ctypes.c_uint32]
    rows_for.restype = ctypes.c_uint32

    def fused(x, out=None):
        if x.dtype != torch.bfloat16:
            raise TypeError(f"expected bfloat16, got {x.dtype}")
        if x.shape[-1] != k:
            raise ValueError(f"expected last dim {k}, got {tuple(x.shape)}")
        if not x.is_contiguous():
            raise ValueError("expected a contiguous tensor; call .contiguous()")
        batch = x.numel() // k
        if out is None:
            q = torch.empty((batch, k // 2), dtype=torch.uint8, device=x.device)
            s = torch.empty((batch, k // MX_BLOCK), dtype=torch.uint8, device=x.device)
        else:
            q, s = out
        launch(
            VECTOR_CORES,
            current_stream_ptr(),
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(q.data_ptr()),
            ctypes.c_void_p(s.data_ptr()),
            batch,
            k,
        )
        return q, s

    fused.rows_for = lambda: rows_for(k)
    fused.k = k
    return fused


def compile_kernel(kind, verbose=True, extra_defs=()):
    return _compile(spec(kind), verbose=verbose, extra_defs=extra_defs)


def build_and_load(kind, k=256, verbose=True, extra_defs=()):
    return _load(spec(kind), k=k, verbose=verbose, extra_defs=extra_defs)
