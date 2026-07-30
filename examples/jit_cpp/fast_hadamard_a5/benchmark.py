#!/usr/bin/env python3
"""Grid-benchmark fast_hadamard_256_a5 over (batch x ROWS_PER_TILE).

The transform (fast_hadamard_256_a5.cpp) and the copy-floor reference
(copy_ref_256_a5.cpp) are separate translation units, built here from their own
sources with a matching ROWS_PER_TILE.

Methodology notes:
  * the copy floor is measured ONCE per batch from a fixed, UB-valid ROWS=64
    build, not recompiled per ROWS -- the copy is a 2-buffer ping/pong, so at
    ROWS_PER_TILE=256 it would need 2*128 KB and overflow UB (its own TU now
    static_asserts this rather than producing garbage timings).
  * MEDIAN of several trials, which rejects the occasional event-timer glitch
    that reads ~2x too fast.
  * the buffer pool is larger than L2, so the copy hits HBM rather than cache.
  * ROWS_PER_TILE=256 is not swept (NBUF=1, buffering-limited, not useful).

Emits CSV: rows,nbuf,batch,had_gbs,copy_gbs,ratio -> build/grid256.csv (+ stdout).
copy_gbs is the fixed ROWS=64 reference for that batch (same across the ROWS axis)."""
import ctypes
import functools
import sys
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa

from jit_a5 import compile_so, entry, stream_ptr

HERE = Path(__file__).resolve().parent
N = 256
ROWS_LIST = [16, 32, 64, 128]
BATCHES = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]  # 2^10..2^18
COPY_ROWS = 64  # fixed, UB-valid tiling for the copy-floor reference
POOL = 8  # working set >> L2 to avoid cache-resident (too-fast) copies
TRIALS = 7  # median over trials rejects timer glitches


def nbuf_for(rows):
    return max(1, min(4, (192 * 1024) // (rows * N * 2)))


def build(rows, nbuf, pf, tag, src=None, extra=()):
    """Build one variant; returns the ctypes handle with its launcher bound.

    force=True: a benchmark must never time a stale .so.
    """
    src = Path(src) if src else HERE / "fast_hadamard_256_a5.cpp"
    # derived, not passed: a per-call-site launcher argument silently bound the
    # transform's symbol against the copy .so whenever a call site omitted it
    launcher = "call_copy256" if "copy_ref" in src.name else "call_hadamard256"
    defs = (f"-DROWS_PER_TILE={rows}", f"-DNBUF={nbuf}", f"-DPREFETCH={pf}", *extra)
    so = compile_so(
        src,
        f"g256_{tag}",
        defs,
        out_dir=HERE / "build",
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


def gbs_median(fn, bd, batch, n=N):
    """Median bandwidth over TRIALS, each trial = r reps over a POOL round-robin."""
    data = 2 * batch * n * 2
    r = 50
    pool = [torch.randn(batch, n, dtype=torch.float16).npu() for _ in range(POOL)]
    torch.npu.synchronize()
    it = {"k": 0}

    def one():
        b = pool[it["k"] % POOL]
        it["k"] += 1
        fn(bd, stream(), ctypes.c_void_p(b.data_ptr()), batch)

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
# N is a build-time macro. Up to 256 each half-row fits one 128-element fp16
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


def rel_err(fn, bd, n, rows):
    """Max relative error vs a torch reference; gates every timing below."""
    batch = rows * 4
    rng = np.random.default_rng(n)
    x_np = rng.standard_normal((batch, n)).astype(np.float16)
    ref = x_np.astype(np.float32) @ hadamard_matrix(n)
    x = torch.from_numpy(x_np).npu()
    fn(bd, stream(), ctypes.c_void_p(x.data_ptr()), batch)
    torch.npu.synchronize()
    out = x.cpu().numpy().astype(np.float32)
    return float(np.abs(out - ref).max()) / (float(np.abs(ref).max()) or 1.0)


def nsweep(bd):
    print("n,chunks,rows,batch,rel_err,had_gbs,copy_gbs,ratio")
    out = ["n,chunks,rows,batch,rel_err,had_gbs,copy_gbs,ratio"]
    for n in NSWEEP_NS:
        rows = max(8, NSWEEP_TILE_BYTES // (n * 2))
        batch = (NSWEEP_TOTAL_ELEMS // n // rows) * rows
        chunks = max(1, (n // 2) // 128)
        defs = (f"-DHAD_N={n}",)
        lib = build(rows, 4, 2, f"n{n}", extra=defs)
        err = rel_err(lib, bd, n, rows)
        if err >= 0.03:
            line = f"{n},{chunks},{rows},{batch},{err:.5f},,,"
            print(line + "   # WRONG -- not timed")
            out.append(line)
            continue
        hg = gbs_median(lib, bd, batch, n=n)
        cl = build(
            rows,
            4,
            2,
            f"cpn{n}",
            src=HERE / "copy_ref_256_a5.cpp",
            extra=(f"-DCOPY_N={n}",),
        )
        cg = gbs_median(cl, bd, batch, n=n)
        line = f"{n},{chunks},{rows},{batch},{err:.5f},{hg:.1f},{cg:.1f},{hg / cg:.4f}"
        print(line)
        sys.stdout.flush()
        out.append(line)
    (HERE / "build/nsweep256.csv").write_text("\n".join(out) + "\n")
    print("NSWEEP DONE")


def main():
    args = [a for a in sys.argv[1:] if a != "--nsweep"]
    bd = int(args[0]) if args else 64
    if "--nsweep" in sys.argv:
        nsweep(bd)
        return

    # ---- fixed copy-floor reference (ROWS=64), measured once per batch ----
    # built from its own TU (copy_ref_256_a5.cpp) so the transform stays standalone
    cref_lib = build(
        COPY_ROWS,
        nbuf_for(COPY_ROWS),
        min(2, nbuf_for(COPY_ROWS) - 1),
        "copyref",
        src=HERE / "copy_ref_256_a5.cpp",
    )
    copy_ref = {}
    for batch in BATCHES:
        copy_ref[batch] = gbs_median(cref_lib, bd, batch)

    print("rows,nbuf,batch,had_gbs,copy_gbs,ratio")
    out = ["rows,nbuf,batch,had_gbs,copy_gbs,ratio"]
    for rows in ROWS_LIST:
        nbuf = nbuf_for(rows)
        pf = min(2, max(0, nbuf - 1))
        lib = build(rows, nbuf, pf, str(rows))
        for batch in BATCHES:
            if batch % rows != 0:
                continue
            hg = gbs_median(lib, bd, batch)
            cg = copy_ref[batch]
            line = f"{rows},{nbuf},{batch},{hg:.1f},{cg:.1f},{hg/cg:.4f}"
            print(line)
            sys.stdout.flush()
            out.append(line)
    (HERE / "build/grid256.csv").write_text("\n".join(out) + "\n")
    print(
        f"# copy-floor peak = {max(copy_ref.values()):.1f} GB/s (should be < ~3300 = HBM ceiling)"
    )
    print("GRID256 DONE")


if __name__ == "__main__":
    main()
