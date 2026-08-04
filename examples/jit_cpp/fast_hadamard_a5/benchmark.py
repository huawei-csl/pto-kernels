#!/usr/bin/env python3
"""Benchmark fast_hadamard_a5 against a torch device-to-device copy.

Sweeps ROWS_PER_TILE against batch; with --tiles, GM<->UB tile size against
block size; with --nsweep, block size alone. Bandwidth counts read + write
traffic.

Every measurement holds the memory footprint at WORKING_SET_BYTES by deriving the
pool depth, and moves TRIAL_BYTES per trial by deriving the rep count. Each row
reports the pool depth, rep count and microseconds per launch behind it, and a
status column: `ok`, or why the row is not a usable bandwidth ratio.

Emits build/grid.csv (ROWS_PER_TILE x batch), build/tiles.csv under --tiles
(tile size x N) and build/nsweep.csv under --nsweep."""
import argparse
import ctypes
import functools
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa

from jit_a5 import compile_so, entry, stream_ptr
from jit_util_a5 import DISPATCH_ARGS, N, rows_for

HERE = Path(__file__).resolve().parent
SRC = HERE / "fast_hadamard_a5.cpp"
BUILDDIR = HERE / "build"
ROWS_LIST = [16, 32, 64, 128]  # ROWS_PER_TILE values swept by the grid
TILE_KIBS = [8, 16, 32, 64]  # tile sizes swept by --tiles
BATCHES = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
WORKING_SET_BYTES = 256 * 1024 * 1024  # footprint held constant across batches
# The cap has to be loose enough that the smallest batch still reaches
# WORKING_SET_BYTES, or the footprint grows with batch and the sweep varies two
# things at once: batch 1024 at N=256 needs 512 buffers of 0.5 MB. A cap of 16
# pinned the footprint only from batch 32768 up, so smaller batches were partly
# cache-resident and the bandwidth-vs-batch curve kinked at the cache knee.
POOL_MIN, POOL_MAX = 2, 512  # round-robin needs two; 512 bounds allocation
TRIAL_BYTES = 8 * 1024**3  # bytes moved per trial
REPS_MIN, REPS_MAX = 20, 2000
MIN_DEVICE_MICROS = 20.0  # under this, a launch is host-bound rather than timed
HBM_BOUND = 3300.0  # a copy reading above this is not a usable measurement
TRIALS = 9  # the median of these is reported


def pool_depth(batch, n):
    """Buffers needed to hold WORKING_SET_BYTES at this batch."""
    per_buffer = batch * n * 2
    return max(POOL_MIN, min(POOL_MAX, WORKING_SET_BYTES // per_buffer))


def reps_for(batch, n):
    """Reps that move TRIAL_BYTES, counting read + write."""
    per_rep = 2 * batch * n * 2
    return int(max(REPS_MIN, min(REPS_MAX, TRIAL_BYTES // per_rep)))


UB_BYTES = 256 * 1024  # A5 (dav-c310)


def cfg(rows, n=N):
    """(nbuf, prefetch) for a tiling: as many buffers as UB holds, up to 4.

    prefetch < nbuf or the pipeline deadlocks.
    """
    nbuf = max(1, min(4, UB_BYTES // (rows * n * 2)))
    return nbuf, min(2, nbuf - 1)


@functools.lru_cache(maxsize=1)
def dispatch_lib():
    """The one .so that serves every N; its launcher takes n as a 5th argument."""
    so = compile_so(
        SRC, "bench_dispatch", (), out_dir=BUILDDIR, verbose=False, force=True
    )
    return entry(so, "call_hadamard", DISPATCH_ARGS)


def build(rows, nbuf, pf, tag, n=N, src=None, extra=()):
    """Build one tuned variant and bind call_hadamard_tuned from it.

    The tuning entry point has no default shape, so all four macros are required.
    force=True so a stale .so is never timed.
    """
    src = Path(src) if src else SRC
    defs = (f"-DROWS_PER_TILE={rows}", f"-DNBUF={nbuf}", f"-DPREFETCH={pf}", *extra)
    defs = ("-DHAD_TUNE", f"-DHAD_N={n}", *defs)
    so = compile_so(
        src,
        f"bench_{tag}",
        defs,
        out_dir=BUILDDIR,
        verbose=False,
        force=True,
    )
    return entry(so, "call_hadamard_tuned")


@functools.lru_cache(maxsize=1)
def stream():
    """The current NPU stream pointer, resolved once per process and lazily."""
    return stream_ptr()


def _time_loop(launch, depth, data, reps):
    """(median GB/s, median microseconds per launch) over TRIALS."""
    samples = []
    counter = 0
    for _ in range(8):  # warm
        launch(counter % depth)
        counter += 1
    torch.npu.synchronize()
    for _ in range(TRIALS):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(reps):
            launch(counter % depth)
            counter += 1
        end.record()
        torch.npu.synchronize()
        samples.append(start.elapsed_time(end) * 1e3 / reps)
    samples.sort()
    micros = samples[len(samples) // 2]
    return data / 1e9 / (micros / 1e6), micros


def torch_copy_gbs(batch, n):
    """Reference bandwidth: a torch device-to-device copy of the same bytes.

    Same read+write accounting, pool sizing and timing loop as the transform. It
    is out-of-place where the transform is in-place, so its footprint is two
    pools rather than one.
    """
    depth = pool_depth(batch, n)
    reps = reps_for(batch, n)
    src = [torch.randn(batch, n, dtype=torch.float16).npu() for _ in range(depth)]
    dst = [torch.empty_like(src[0]) for _ in range(depth)]
    gbs, micros = _time_loop(
        lambda k: dst[k].copy_(src[k]), depth, 2 * batch * n * 2, reps
    )
    src.clear()
    dst.clear()
    torch.npu.empty_cache()
    return gbs, micros


def gbs_median(fn, bd, batch, n=N, dispatch_n=None):
    """(median GB/s, median microseconds per launch) for the transform."""
    depth = pool_depth(batch, n)
    reps = reps_for(batch, n)
    pool = [torch.randn(batch, n, dtype=torch.float16).npu() for _ in range(depth)]
    torch.npu.synchronize()

    def one(k):
        args = (bd, stream(), ctypes.c_void_p(pool[k].data_ptr()), batch)
        if dispatch_n:
            args += (dispatch_n,)
        fn(*args)

    gbs, micros = _time_loop(one, depth, 2 * batch * n * 2, reps)
    pool.clear()
    torch.npu.empty_cache()
    return gbs, micros


def verdict(had, copy, micros):
    """Why this measurement is not a usable bandwidth ratio, or 'ok'."""
    if micros < MIN_DEVICE_MICROS:
        return f"launch-bound({micros:.0f}us)"
    if copy > HBM_BOUND:
        return f"copy-over-hbm({copy:.0f})"
    if had / copy > 1.0:
        return f"ratio-over-one({had / copy:.3f})"
    return "ok"


# ---- block-size sweep -------------------------------------------------------
# Total elements moved is held constant, so N is the only variable and the memory
# footprint is the same at every N.
NSWEEP_NS = [32, 64, 128, 256, 512, 1024, 2048]
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


NSWEEP_HEADER = (
    "n,chunks,rows,batch,pool,reps,micros,rel_err,had_gbs,copy_gbs,ratio,status"
)
GRID_HEADER = "rows,nbuf,batch,pool,reps,micros,had_gbs,copy_gbs,ratio,status"
TILE_HEADER = "n,tile_kib,rows,nbuf,batch,micros,had_gbs,copy_gbs,ratio,status"


def median_of(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def nsweep(bd, repeat):
    print(NSWEEP_HEADER)
    out = [NSWEEP_HEADER]
    for n in NSWEEP_NS:
        rows = rows_for(n)
        batch = (NSWEEP_TOTAL_ELEMS // n // rows) * rows
        chunks = max(1, (n // 2) // 128)
        lib = dispatch_lib()
        err = rel_err(lib, bd, n, rows, n)
        prefix = (
            f"{n},{chunks},{rows},{batch},{pool_depth(batch, n)},"
            f"{reps_for(batch, n)}"
        )
        if err >= 0.03:
            line = f"{prefix},,{err:.5f},,,,wrong"
            print(line + "   # WRONG -- not timed")
            out.append(line)
            continue
        # repeat the whole pair so the two sides stay interleaved
        pairs = [
            (
                gbs_median(lib, bd, batch, n=n, dispatch_n=n),
                torch_copy_gbs(batch, n),
            )
            for _ in range(repeat)
        ]
        hg = median_of([had for (had, _), _ in pairs])
        micros = median_of([us for (_, us), _ in pairs])
        cg = median_of([copy for _, (copy, _) in pairs])
        line = (
            f"{prefix},{micros:.1f},{err:.5f},{hg:.1f},{cg:.1f},"
            f"{hg / cg:.4f},{verdict(hg, cg, micros)}"
        )
        print(line)
        sys.stdout.flush()
        out.append(line)
    (BUILDDIR / "nsweep.csv").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("NSWEEP DONE")


def grid(bd, repeat):
    """Sweep ROWS_PER_TILE against batch at the default N, with the ratio."""
    print(GRID_HEADER)
    out = [GRID_HEADER]
    broken = []
    copy_ref = {}
    for rows in ROWS_LIST:
        nbuf, pf = cfg(rows, N)
        lib = build(rows, nbuf, pf, f"grid{rows}", n=N)
        for batch in BATCHES:
            if batch % rows != 0:
                continue
            if batch not in copy_ref:
                copy_ref[batch] = torch_copy_gbs(batch, N)[0]
            trials = [
                gbs_median(lib, bd, batch, dispatch_n=None) for _ in range(repeat)
            ]
            hg = median_of([had for had, _ in trials])
            micros = median_of([us for _, us in trials])
            cg = copy_ref[batch]
            status = verdict(hg, cg, micros)
            if status != "ok":
                broken.append((rows, batch, status))
            line = (
                f"{rows},{nbuf},{batch},{pool_depth(batch, N)},"
                f"{reps_for(batch, N)},{micros:.1f},{hg:.1f},{cg:.1f},"
                f"{hg / cg:.4f},{status}"
            )
            print(line)
            sys.stdout.flush()
            out.append(line)
    (BUILDDIR / "grid.csv").write_text("\n".join(out) + "\n", encoding="utf-8")
    floors = list(copy_ref.values())
    print(f"# copy floor {min(floors):.1f}..{max(floors):.1f} GB/s (read+write)")
    if broken:
        print(f"# {len(broken)} cell(s) flagged: {broken}")
    print("GRID DONE")


def tiles(bd, repeat):
    """Sweep GM<->UB tile size against block size at one memory footprint.

    Each cell moves NSWEEP_TOTAL_ELEMS, so the reference depends only on N and is
    measured once per N. A tiling the kernel's static_asserts reject is reported
    as unbuildable rather than predicted here.
    """
    print(TILE_HEADER)
    out = [TILE_HEADER]
    unbuildable = []
    for n in NSWEEP_NS:
        copy_gbs = None
        for tile_kib in TILE_KIBS:
            rows = tile_kib * 1024 // (n * 2)
            if rows < 1:
                unbuildable.append((n, tile_kib, "tile is under one row"))
                continue
            nbuf, pf = cfg(rows, n)
            try:
                lib = build(rows, nbuf, pf, f"tile{n}_{tile_kib}", n=n)
            except subprocess.CalledProcessError:
                unbuildable.append((n, tile_kib, f"rows={rows} rejected by kernel"))
                continue
            batch = (NSWEEP_TOTAL_ELEMS // n // rows) * rows
            if copy_gbs is None:
                copy_gbs = torch_copy_gbs(batch, n)[0]
            trials = [
                gbs_median(lib, bd, batch, n=n, dispatch_n=n) for _ in range(repeat)
            ]
            hg = median_of([had for had, _ in trials])
            micros = median_of([us for _, us in trials])
            line = (
                f"{n},{tile_kib},{rows},{nbuf},{batch},{micros:.1f},"
                f"{hg:.1f},{copy_gbs:.1f},{hg / copy_gbs:.4f},"
                f"{verdict(hg, copy_gbs, micros)}"
            )
            print(line)
            sys.stdout.flush()
            out.append(line)
    (BUILDDIR / "tiles.csv").write_text("\n".join(out) + "\n", encoding="utf-8")
    if unbuildable:
        print(f"# {len(unbuildable)} tiling(s) not buildable: {unbuildable}")
    print("TILES DONE")


def parse_args(argv=None):
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "block_dim", nargs="?", type=int, default=64, help="AI cores (default 64)"
    )
    parser.add_argument(
        "--nsweep", action="store_true", help="sweep block size N instead of the grid"
    )
    parser.add_argument(
        "--tiles", action="store_true", help="sweep tile size against block size"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="repeat each measurement and take the median (default 1)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.repeat < 1:
        raise ValueError(f"--repeat must be >= 1, got {args.repeat}")
    if args.nsweep:
        nsweep(args.block_dim, args.repeat)
    elif args.tiles:
        tiles(args.block_dim, args.repeat)
    else:
        grid(args.block_dim, args.repeat)


if __name__ == "__main__":
    main()
