"""Build and load the A5 MXFP4 matmul kernel.

A full ``y = A @ B`` with both operands MXFP4 -- E2M1 nibbles carrying one
E8M0 scale per 32 elements along K -- accumulated in fp32 and stored bf16, on
the cube's native microscaled path. The 32 is the scale granularity; the
matmul is not blocked.

K and N are compile-time template arguments, because the GM strides are
per-layer constants and static addressing is what keeps the inner loop tight.
One built ``.so`` therefore serves exactly one (K, N) pair and the build cache
is keyed on both. M is a runtime argument: ``block_dim`` is
``m_tiles * n_tiles``, so a compile-time M would cap parallelism at
``N / n_tile`` -- four cores at N=1024.

The kernel cannot report an error from the device. Handed a shape it was not
built for, its launcher returns having written nothing, and the caller gets an
untouched output buffer back rather than a failure. Every argument is checked
here instead. Use :func:`load_matmul`, which validates and then launches.
"""

import ctypes
import functools
import os
import subprocess
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "mxfp4_matmul_a5.cpp"

MX_BLOCK = 32  # elements per E8M0 scale
BASE_K = 256  # the cube's K tile; must match BASE_K in the kernel
K_L1 = 512  # default L1 slab width; must match MXMM_K_L1
N_ALIGN = 64  # TMATMUL_MX needs N 64-aligned for fp4
M_ALIGN = 16  # and M 16-aligned
DEFAULT_CUBE_CORES = 32  # what an A5 reports, and the fallback


@functools.lru_cache(maxsize=1)
def cube_core_count() -> int:
    """Blocks a launch should ask for: one per cube core.

    Read from the device rather than assumed. Asking for more than the device
    has still computes the right answer -- the kernel's work loop strides by
    the block count -- but it measured 1.00-1.05x slower than one block per
    core, and the tile picker uses the same number to decide when a shape
    fills the machine. Cached, because a property query per launch is the
    per-call cost ``current_stream_ptr`` warns about.

    A device that will not answer falls back to ``DEFAULT_CUBE_CORES``, which
    is right for the A5 this was written on but is a guess anywhere else, and
    a wrong guess is silent: too low leaves cores idle, too high oversubscribes
    them. The fallback warns for that reason -- the launch is still correct
    either way, only slower.
    """
    import torch

    try:
        properties = torch.npu.get_device_properties(torch.npu.current_device())
    except (AttributeError, RuntimeError) as exc:
        warnings.warn(
            f"could not read cube_core_num from the device ({exc!r}); "
            f"assuming {DEFAULT_CUBE_CORES} cube cores. The result is still "
            f"correct, but block_dim may not match this part.",
            RuntimeWarning,
            stacklevel=2,
        )
        return DEFAULT_CUBE_CORES
    cores = int(getattr(properties, "cube_core_num", DEFAULT_CUBE_CORES))
    if cores < 1:
        warnings.warn(
            f"device reported cube_core_num={cores}; assuming "
            f"{DEFAULT_CUBE_CORES} cube cores instead.",
            RuntimeWarning,
            stacklevel=2,
        )
        return DEFAULT_CUBE_CORES
    return cores


# (block_dim, stream, a, a_scale, b, b_scale, out, m, k, n)
KERNEL_ARGS = [
    ctypes.c_uint32,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint32,
]
# Flags that never vary; one string so black leaves the wrapping alone.
FIXED_FLAGS = (
    "-O2 -std=c++17 -fPIC -Wno-ignored-attributes -Wno-macro-redefined "
    "-mllvm -cce-aicore-stack-size=0x8000 "
    "-mllvm -cce-aicore-function-stack-size=0x8000 "
    "-mllvm -cce-aicore-addr-transform "
    "-mllvm -cce-aicore-dcci-insert-for-scalar=false "
    "-Xhost-start -Xhost-end"
).split()


def ascend_home() -> str:
    for key in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        if os.environ.get(key):
            return os.environ[key]
    raise RuntimeError("set ASCEND_HOME_PATH or ASCEND_TOOLKIT_HOME")


def check_shape(k: int, n: int, k_l1: int = K_L1) -> None:
    """Validate (K, N) before any arithmetic that could fail on a bad value."""
    if k <= 0 or n <= 0:
        raise ValueError(f"k and n must be positive, got k={k}, n={n}")
    if k % k_l1:
        raise ValueError(
            f"k must be whole L1 slabs of {k_l1} elements, got k={k}. "
            f"Build with -DMXMM_K_L1 to change the slab width."
        )
    if n % N_ALIGN:
        raise ValueError(f"n must be a multiple of {N_ALIGN} for fp4, got n={n}")


def check_rows(m: int, m_max: int, m_tile: int) -> None:
    """Validate M against what this build accepts."""
    if m <= 0:
        raise ValueError(f"m must be positive, got m={m}")
    if m % m_tile:
        raise ValueError(f"m must be a multiple of {m_tile}, got m={m}")
    if m > m_max:
        raise ValueError(f"m must be at most {m_max}, got m={m}")


def compile_kernel(
    k: int,
    n: int,
    extra_defs=(),
    force: bool = False,
    verbose: bool = True,
) -> Path:
    """Compile and link to a device ``.so``, reusing an up-to-date one.

    K and N are compile-time, so they go in the name. Without that the first
    shape built wins the cache and every later caller is silently handed a
    binary for the wrong dimensions -- and the kernel's launcher answers a
    mismatched shape by doing nothing at all.
    """
    check_shape(k, n)
    out_dir = HERE / "build"
    out_dir.mkdir(parents=True, exist_ok=True)
    cores = cube_core_count()
    stem = f"mxfp4_matmul_a5_k{k}_n{n}_tb{cores}"
    if extra_defs:
        tag = "_".join(sorted(d.lstrip("-D").replace("=", "") for d in extra_defs))
        stem = f"{stem}_{tag}"
    obj, so = out_dir / f"{stem}.o", out_dir / f"{stem}.so"
    if not force and so.exists() and so.stat().st_mtime >= SRC.stat().st_mtime:
        if verbose:
            print(f"[compile] up-to-date, reusing {so.name}")
        return so
    home = ascend_home()
    bisheng = f"{home}/bin/bisheng"
    arch = ["--cce-aicore-arch=dav-c310", "-DREGISTER_BASE"]
    inc = [f"-I{home}/aarch64-linux/include", f"-I{home}/include"]
    shape = [
        f"-DMXMM_TEST_K={k}",
        f"-DMXMM_TEST_N={n}",
        f"-DMXMM_TARGET_BLOCKS={cores}",
    ]
    cmd = [bisheng, "-xcce", *arch, *FIXED_FLAGS, *inc, *shape, *extra_defs]
    subprocess.run([*cmd, "-c", str(SRC), "-o", str(obj)], check=True)
    link = f"-fPIC -shared --cce-fatobj-link -Wl,-soname,{so.name}".split()
    subprocess.run([bisheng, *link, str(obj), "-o", str(so)], check=True)
    return so


def bind_launcher(so_path, name: str, argtypes=None):
    """ctypes-load ``so_path`` and bind the launcher ``name``."""
    fn = getattr(ctypes.CDLL(str(so_path)), name)
    fn.argtypes = list(KERNEL_ARGS if argtypes is None else argtypes)
    fn.restype = None
    return fn


def current_stream_ptr():
    """Raw pointer for the ACTIVE stream, resolved per launch.

    Never cached: a cached pointer sends later launches to whichever stream was
    current first -- a race, not an error. The raw accessor is used instead of
    current_stream() because it is far cheaper and follows a
    `with torch.npu.stream(...)` block identically.
    """
    import torch
    import torch_npu

    # pylint: disable=protected-access  # the public accessor costs 8.9 us
    raw = getattr(torch_npu._C, "_npu_getCurrentRawStream", None)
    if raw is not None:
        return ctypes.c_void_p(raw(torch.npu.current_device()))
    handle = getattr(torch.npu.current_stream(), "npu_stream", None)
    if handle is None:
        raise RuntimeError("could not resolve the NPU stream pointer")
    return ctypes.c_void_p(int(handle))


def kernel_shape(so_path):
    """What this build was compiled for, read back from the binary itself.

    Cheaper than trusting the filename, and the only way to catch a stale
    ``.so`` whose name says one shape while its template arguments say another.
    """
    lib = ctypes.CDLL(str(so_path))
    out = {}
    for name in ("m_tile", "m_max", "k", "n"):
        fn = getattr(lib, f"mxfp4_matmul_{name}")
        fn.argtypes = []
        fn.restype = ctypes.c_uint32
        out[name] = int(fn())
    return out


def tile_plan(so_path):
    """The kernel's own M-rounding and tile choice, as plain callables.

    The tile is picked from M so that ``m_tiles * n_tiles`` fills the device;
    a caller that guesses a multiple of its own instead can land on a launch
    the kernel silently declines.
    """
    lib = ctypes.CDLL(str(so_path))
    fns = {}
    for name in ("m_round", "m_tile_for", "n_tile_for"):
        fn = getattr(lib, f"mxfp4_matmul_{name}")
        fn.argtypes = [ctypes.c_uint32]
        fn.restype = ctypes.c_uint32
        fns[name] = fn

    def plan(m: int):
        if m <= 0:
            raise ValueError(f"m must be positive, got m={m}")
        m_run = int(fns["m_round"](m))
        m_tile = int(fns["m_tile_for"](m_run))
        n_tile = int(fns["n_tile_for"](m_run))
        if 0 in (m_run, m_tile, n_tile):
            raise ValueError(f"kernel has no tile plan for m={m}")
        return m_run, m_tile, n_tile

    return plan


def load_matmul(so_path, k: int, n: int):
    """Return a validated callable for one built (K, N).

    The returned function takes the packed operands and writes into ``out``:
    ``a`` and ``b`` are E2M1 nibbles two per byte, ``b`` stored DN -- that is,
    (N, K) row-major -- and the scales are E8M0 bytes, ``a_scale`` (M, K/32)
    row-major and ``b_scale`` (N, K/32), matching B's transpose.

    M is rounded UP to whole tiles, because the kernel launches on whole tiles.
    So the operands and the result are ``m_round(m)`` rows tall, not ``m``, and
    the rows past ``m`` hold whatever the padded operands produced. Read the
    height off the returned tensor rather than assuming it; a caller that
    supplies ``out`` at the unrounded height is refused rather than having the
    extra rows written past the end of it.
    """
    import torch

    check_shape(k, n)
    built = kernel_shape(so_path)
    if (built["k"], built["n"]) != (k, n):
        raise ValueError(
            f"{Path(so_path).name} was built for k={built['k']} n={built['n']}, "
            f"asked for k={k} n={n}"
        )
    plan = tile_plan(so_path)
    kernel = bind_launcher(so_path, "call_mxfp4_matmul")

    def run(a, a_scale, b, b_scale, out=None, m=None, stream_ptr=None):
        rows = int(a.shape[0]) if m is None else int(m)
        m_run, m_tile, n_tile = plan(rows)
        check_rows(m_run, built["m_max"], built["m_tile"])
        # The kernel reads each buffer as one flat contiguous run and can
        # report neither a wrong dtype nor a strided view: a wider dtype is
        # reinterpreted and a non-contiguous tensor is read as if packed.
        for name, buf, shape in (
            ("a", a, (m_run, k // 2)),
            ("a_scale", a_scale, (m_run, k // MX_BLOCK)),
            ("b", b, (n, k // 2)),
            ("b_scale", b_scale, (n, k // MX_BLOCK)),
        ):
            if buf.dtype != torch.uint8:
                raise TypeError(f"{name} must be uint8, got {buf.dtype}")
            if not buf.is_contiguous():
                raise ValueError(f"{name} must be contiguous; call .contiguous()")
            if tuple(buf.shape) != shape:
                raise ValueError(f"{name} must be {shape}, got {tuple(buf.shape)}")
        if out is None:
            out = torch.empty((m_run, n), dtype=torch.bfloat16, device=a.device)
        elif out.dtype != torch.bfloat16 or tuple(out.shape) != (m_run, n):
            raise ValueError(
                f"out must be bfloat16 {(m_run, n)}, got "
                f"{out.dtype} {tuple(out.shape)}"
            )
        blocks = (m_run // m_tile) * (n // n_tile)
        kernel(
            min(cube_core_count(), max(1, blocks)),
            current_stream_ptr() if stream_ptr is None else stream_ptr,
            ctypes.c_void_p(a.data_ptr()),
            ctypes.c_void_p(a_scale.data_ptr()),
            ctypes.c_void_p(b.data_ptr()),
            ctypes.c_void_p(b_scale.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            m_run,
            k,
            n,
        )
        return out

    def prepare(a, a_scale, b, b_scale, out=None, m=None, stream_ptr=None):
        """Validate once, then return a launcher that only launches.

        ``run`` re-derives the tile plan on every call, and the plan costs
        three ctypes round-trips into the ``.so``. At M=16 K=N=1024 the whole
        matmul is about 30 us, so that bookkeeping is a measurable part of it
        and timing ``run`` in a loop reports the wrapper as much as the kernel.
        Anything measuring this kernel, or calling it at a fixed shape in a
        hot loop, should bind the shape once through here.

        Two consequences of binding. The stream pointer is resolved now, not
        per launch, which is the one thing ``current_stream_ptr`` warns against
        -- so a launcher prepared outside a ``with torch.npu.stream(...)``
        block keeps firing at the stream that was current when it was
        prepared. Re-prepare inside the block instead. And preparing runs the
        validating path once, so ``out`` has already been written when this
        returns.
        """
        rows = int(a.shape[0]) if m is None else int(m)
        m_run, m_tile, n_tile = plan(rows)
        prepared_out = run(a, a_scale, b, b_scale, out=out, m=rows)
        blocks = min(cube_core_count(), max(1, (m_run // m_tile) * (n // n_tile)))
        stream = current_stream_ptr() if stream_ptr is None else stream_ptr
        pointers = tuple(
            ctypes.c_void_p(buf.data_ptr())
            for buf in (a, a_scale, b, b_scale, prepared_out)
        )

        # The launcher holds the operand TENSORS, not only their addresses.
        # Without this the caller's tensors can fall out of scope while the
        # launcher lives on, the caching allocator hands that memory to
        # whatever allocates next, and the kernel silently reads it -- which is
        # a wrong answer with no error, and it cost a bitwise A/B a false
        # mismatch before the cause was found.
        held = (a, a_scale, b, b_scale, prepared_out)

        def launch():
            kernel(blocks, stream, *pointers, m_run, k, n)
            return held[-1]

        return launch, prepared_out

    run.prepare = prepare
    return run
