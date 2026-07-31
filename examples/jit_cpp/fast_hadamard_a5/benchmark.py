#!/usr/bin/env python3
"""Grid-benchmark fast_hadamard_a5 over (batch x ROWS_PER_TILE).

The transform (fast_hadamard_a5.cpp) and the copy-floor reference
(copy_ref_a5.cpp) are separate translation units, built here from their own
sources with a matching ROWS_PER_TILE.

Methodology notes:
  * the copy floor is measured ONCE per batch from a fixed, UB-valid ROWS=64
    build, not recompiled per ROWS -- the copy is a 2-buffer ping/pong, so at
    ROWS_PER_TILE=256 it would need 2*128 KB and overflow UB (its own TU now
    static_asserts this rather than producing garbage timings).
  * MEDIAN of several trials, which rejects the occasional event-timer glitch
    that reads ~2x too fast.
  * the buffer pool is sized past the measured cache knee (64..128 MiB at batch
    16384), so the copy floor is HBM bandwidth and not cache bandwidth.
  * ROWS_PER_TILE=256 is not swept (NBUF=1, buffering-limited, not useful).

Emits CSV: rows,nbuf,batch,had_gbs,copy_gbs,ratio -> build/grid.csv (+ stdout).
`--nsweep --repeat K` medians K measurements per N; a single sweep varies by up
to 0.06 in the ratio, so published figures should use K > 1.
copy_gbs is the fixed ROWS=64 reference for that batch (same across the ROWS axis)."""
import ctypes
import functools
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa

from jit_a5 import compile_so, entry, stream_ptr
from jit_util_a5 import DISPATCH_ARGS, N, rows_for

HERE = Path(__file__).resolve().parent
SRC = HERE / "fast_hadamard_a5.cpp"
CSRC = HERE / "copy_ref_a5.cpp"
BUILDDIR = HERE / "build"
ROWS_LIST = [16, 32, 64, 128]
BATCHES = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]  # 2^10..2^18
COPY_ROWS = 64  # fixed, UB-valid tiling for the copy-floor reference
# Round-robin working set. Must clear cache or the measured copy floor is cache
# bandwidth: at batch 16384 the knee is between 64 and 128 MiB, and a fixed
# 8-buffer pool sat at 64 MiB -- inside it. Buffer count is derived per batch.
WORKING_SET_BYTES = 384 * 1024 * 1024
TRIALS = 7  # median over trials rejects timer glitches


def cfg(rows):
    """(nbuf, prefetch) for a tiling. prefetch < nbuf or the pipeline deadlocks."""
    nbuf = max(1, min(4, (192 * 1024) // (rows * N * 2)))
    return nbuf, min(2, nbuf - 1)


@functools.lru_cache(maxsize=1)
def dispatch_lib():
    """The one .so that serves every N; its launcher takes n as a 5th argument."""
    so = compile_so(
        SRC, "bench_dispatch", (), out_dir=BUILDDIR, verbose=False, force=True
    )
    return entry(so, "call_hadamard", DISPATCH_ARGS)


def build(rows, nbuf, pf, tag, src=None, extra=()):
    """Build one variant; returns the ctypes handle with its launcher bound.

    force=True: a benchmark must never time a stale .so.
    """
    src = Path(src) if src else SRC
    # derived, not passed: a per-call-site launcher argument silently bound the
    # transform's symbol against the copy .so whenever a call site omitted it
    copy = "copy_ref" in src.name
    launcher = "call_copy" if copy else "call_hadamard_tuned"
    defs = (f"-DROWS_PER_TILE={rows}", f"-DNBUF={nbuf}", f"-DPREFETCH={pf}", *extra)
    if not copy:
        # The tuning entry point is opt-in and deliberately has no
        # defaults, so every shape macro must be passed; the grid only
        # ever sweeps N=256.
        defs = ("-DHAD_TUNE", f"-DHAD_N={N}", *defs)
    so = compile_so(
        src,
        f"bench_{tag}",
        defs,
        out_dir=BUILDDIR,
        verbose=False,
        force=True,
    )
    return entry(so, launcher)


@functools.lru_cache(maxsize=1)
def stream():
    """Resolved once per process, but lazily: re-resolving inside a timed loop
    skews the measurement, and doing it at import time would make importing this
    module require a live device."""
    return stream_ptr()


def gbs_median(fn, bd, batch, n=N, dispatch_n=None):
    """Median bandwidth over TRIALS, each trial = r reps round-robin over the pool."""
    data = 2 * batch * n * 2
    r = 50
    # one allocation, sliced: at small batches the pool needs hundreds of
    # buffers to clear cache, and that many separate allocations is slow
    depth = max(2, -(-WORKING_SET_BYTES // (batch * n * 2)))
    block = torch.randn(depth * batch, n, dtype=torch.float16).npu()
    pool = [block[i * batch : (i + 1) * batch] for i in range(depth)]
    torch.npu.synchronize()
    it = {"k": 0}

    def one():
        b = pool[it["k"] % depth]
        it["k"] += 1
        args = (bd, stream(), ctypes.c_void_p(b.data_ptr()), batch)
        if dispatch_n:
            args += (dispatch_n,)
        fn(*args)

    for _ in range(8):
        one()
    torch.npu.synchronize()
    gs = []
    for _ in range(TRIALS):
        s, e = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        s.record()
        for _ in range(r):
            one()
        e.record()
        torch.npu.synchronize()
        us = s.elapsed_time(e) * 1e3 / r
        gs.append(data / 1e9 / (us / 1e6))
    gs.sort()
    return gs[len(gs) // 2]


# ---- block-size sweep -------------------------------------------------------
# Up to N=256 each half-row fits one 128-element fp16
# vector; beyond that the row is split into CHUNKS = (N/2)/128 pieces. Hold the
# tile size and the total bytes moved constant so N is the only variable, and
# rebuild the copy floor at each N so every ratio is against its own DMA ceiling.
NSWEEP_NS = [32, 64, 128, 256, 512, 1024, 2048]
NSWEEP_TILE_BYTES = 32 * 1024
NSWEEP_TOTAL_ELEMS = 1 << 24


def hadamard_matrix(n):
    matrix = np.array([[1.0]])
    while matrix.shape[0] < n:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix


def rel_err(fn, bd, n, rows, dispatch_n=None):
    """Max relative error vs a torch reference; gates every timing below."""
    batch = rows * 4
    rng = np.random.default_rng(n)
    x_np = rng.standard_normal((batch, n)).astype(np.float16)
    ref = x_np.astype(np.float32) @ hadamard_matrix(n)
    x = torch.from_numpy(x_np).npu()
    args = (bd, stream(), ctypes.c_void_p(x.data_ptr()), batch)
    if dispatch_n:
        args += (dispatch_n,)
    fn(*args)
    torch.npu.synchronize()
    out = x.cpu().numpy().astype(np.float32)
    return float(np.abs(out - ref).max()) / (float(np.abs(ref).max()) or 1.0)


def nsweep(bd, repeat=1):
    print("n,chunks,rows,batch,rel_err,had_gbs,copy_gbs,ratio")
    out = ["n,chunks,rows,batch,rel_err,had_gbs,copy_gbs,ratio"]
    for n in NSWEEP_NS:
        rows = rows_for(n)
        batch = (NSWEEP_TOTAL_ELEMS // n // rows) * rows
        chunks = max(1, (n // 2) // 128)
        lib = dispatch_lib()
        err = rel_err(lib, bd, n, rows, n)
        if err >= 0.03:
            line = f"{n},{chunks},{rows},{batch},{err:.5f},,,"
            print(line + "   # WRONG -- not timed")
            out.append(line)
            continue
        hg = statistics.median(
            gbs_median(lib, bd, batch, n=n, dispatch_n=n) for _ in range(repeat)
        )
        cl = build(
            rows,
            4,
            2,
            f"cpn{n}",
            src=CSRC,
            extra=(f"-DCOPY_N={n}",),
        )
        cg = statistics.median(gbs_median(cl, bd, batch, n=n) for _ in range(repeat))
        line = f"{n},{chunks},{rows},{batch},{err:.5f},{hg:.1f},{cg:.1f},{hg / cg:.4f}"
        print(line)
        sys.stdout.flush()
        out.append(line)
    (BUILDDIR / "nsweep.csv").write_text("\n".join(out) + "\n")
    print("NSWEEP DONE")


def main():
    flags = {"--nsweep"}
    repeat = 1
    argv = list(sys.argv[1:])
    if "--repeat" in argv:
        i = argv.index("--repeat")
        repeat = int(argv[i + 1])
        del argv[i : i + 2]
    args = [a for a in argv if a not in flags]
    bd = int(args[0]) if args else 64
    if "--nsweep" in argv:
        nsweep(bd, repeat)
        return

    # ---- fixed copy-floor reference (ROWS=64), measured once per batch ----
    # built from its own TU (copy_ref_a5.cpp) so the transform stays standalone
    cref_lib = build(COPY_ROWS, *cfg(COPY_ROWS), "copyref", src=CSRC)
    copy_ref = {}
    for batch in BATCHES:
        copy_ref[batch] = gbs_median(cref_lib, bd, batch)

    print("rows,nbuf,batch,had_gbs,copy_gbs,ratio")
    out = ["rows,nbuf,batch,had_gbs,copy_gbs,ratio"]
    for rows in ROWS_LIST:
        nbuf, pf = cfg(rows)
        lib = build(rows, nbuf, pf, str(rows))
        for batch in BATCHES:
            if batch % rows != 0:
                continue
            hg = gbs_median(lib, bd, batch, dispatch_n=None)
            cg = copy_ref[batch]
            line = f"{rows},{nbuf},{batch},{hg:.1f},{cg:.1f},{hg/cg:.4f}"
            print(line)
            sys.stdout.flush()
            out.append(line)
    (BUILDDIR / "grid.csv").write_text("\n".join(out) + "\n")
    lo, hi = min(copy_ref.values()), max(copy_ref.values())
    print(f"# copy floor {lo:.1f}..{hi:.1f} GB/s (read+write) across batches")
    print("GRID DONE")


if __name__ == "__main__":
    main()
