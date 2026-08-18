#!/usr/bin/env python3
"""Run the VMI (ptodsl) parametric-N fast Walsh-Hadamard kernel.

ONE kernel, `full`, parametric on the transform width N in
{32,64,128,256,512,1024,2048}: in-place fp16 y = x @ H (unnormalized, natural-
order Sylvester Hadamard). make_full(N).compile() selects the matching per-N
.pto (see full_n_mlir.py / gen_had_full.py).

The correctness path (`--check`) is pyACL-only (torch-free), so the SAME code
runs both under cannsim (through scripts/run_sim.sh) and on a real A5 device.
The optional bandwidth path (`--bench`) needs torch + torch_npu and a device;
its import is guarded so `--check` parses and runs cleanly without torch.

Modes:
  --check   (default) pyACL correctness vs the numpy golden; prints rel_err +
            PASS/FAIL. Runs on cannsim AND device.
  --nsweep  --check over every supported N (device or cannsim).
  --bench   DEVICE-ONLY effective GM bandwidth next to a torch D2D copy floor.

Divisibility: the kernel streams grid=G row bands of CR=32KB/(N*2) rows through
a 4-buffer UB ring, so batch %% (G * 65536/N) == 0. The default batch is the
smallest valid one, G * (65536/N).
"""
from __future__ import annotations

import argparse
import ctypes
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from full_n_mlir import SUPPORTED_NS, make_full   # noqa: E402
from golden import ref_hadamard                    # noqa: E402  (x @ H)

TOLERANCE = 0.03            # fp16 accumulation over log2(N) butterfly stages
SANITY_MAX_GBS = 8000.0     # drop measurement glitches (real A5 HBM tops ~3 TB/s)
WORKING_SET_BYTES = 256 << 20   # 256 MiB: past the A5 L2 knee -> real HBM
DEVICE_MEM_CAP = 1 << 30    # ~1 GiB working-set ceiling for the timing pools


def default_batch(n: int, grid: int) -> int:
    """Smallest batch that splits evenly across the grid and fills the 4-buffer
    ring: G * 4*CR = G * 65536/N."""
    return grid * (65536 // n)


# ---------------------------------------------------------------------------
# Correctness self-check -- pyACL, torch-free, RUNS UNDER cannsim AND on device.
# Lifted from the vmi-demo benchmark's self_check_pyacl (acl.init/malloc/
# bytes_to_ptr H2D / LaunchHandle / synchronize / D2H), specialized to `full`.
# ---------------------------------------------------------------------------
def self_check(n: int, batch: int, grid: int, seed: int = 1234,
               verbose: bool = True) -> float:
    """Launch the full kernel once via pyACL, compare against x @ H, return rel_err."""
    import acl  # local: keep the module importable without a device at parse time

    if n not in SUPPORTED_NS:
        raise ValueError(f"N={n} unsupported (choose {SUPPORTED_NS})")
    if batch % default_batch(n, grid) != 0:
        raise ValueError(
            f"batch={batch} invalid for N={n} grid={grid}: need batch % "
            f"{default_batch(n, grid)} == 0 (G * 65536/N)")

    rng = np.random.default_rng(seed)
    x = np.ascontiguousarray(rng.standard_normal((batch, n)).astype(np.float16))
    ref = ref_hadamard(x)

    H2D, D2H, HUGE = 1, 2, 0
    r_init = acl.init()
    count, _rc = acl.rt.get_device_count()
    if os.environ.get("ASCEND_DEVICE") is not None:
        cands = [int(os.environ["ASCEND_DEVICE"])]
    else:
        cands = list(range(count or 1))  # task-submit may lock a non-zero device
    dev_id, r_setdev = None, None
    for _d in cands:
        r_setdev = acl.rt.set_device(_d)
        if r_setdev == 0:
            dev_id = _d
            break
    if dev_id is None:
        raise RuntimeError(
            f"no usable NPU: set_device failed on {cands} (last ret={r_setdev}); "
            f"set ASCEND_DEVICE to your task's device id")
    ctx, r_ctx = acl.rt.create_context(dev_id)
    stream, r_stream = acl.rt.create_stream()
    if verbose:
        print(f"[diag-acl] init={r_init} device={dev_id} set_device={r_setdev} "
              f"ctx_ret={r_ctx} stream_ret={r_stream} count={count}", flush=True)
    try:
        kn = make_full(n).compile()
        launch = kn[grid, stream]                  # ptodsl LaunchHandle (grid + stream)

        nbytes = x.nbytes
        xd, r_malloc = acl.rt.malloc(nbytes, HUGE)
        if verbose:
            print(f"[diag-acl] malloc ret={r_malloc} xd={xd}", flush=True)

        # acl.util.numpy_to_ptr is deprecated and silently no-ops H2D on CANN
        # 9.1.0; use bytes_to_ptr and keep the src bytes alive across the copy.
        _b2p = getattr(acl.util, "bytes_to_ptr", None)

        def _h2d(dev, arr):
            if _b2p is not None:
                buf = arr.tobytes()
                acl.rt.memcpy(dev, arr.nbytes, _b2p(buf), arr.nbytes, H2D)
            else:
                acl.rt.memcpy(dev, arr.nbytes, acl.util.numpy_to_ptr(arr),
                              arr.nbytes, H2D)

        def _d2h(dev, shape, dtype=np.float16):
            nb = int(np.prod(shape)) * np.dtype(dtype).itemsize
            if _b2p is not None:
                ob = bytes(nb)
                acl.rt.memcpy(_b2p(ob), nb, dev, nb, D2H)
                return np.frombuffer(ob, dtype=dtype).reshape(shape).copy()
            o = np.ascontiguousarray(np.zeros(shape, dtype))
            acl.rt.memcpy(acl.util.numpy_to_ptr(o), nb, dev, nb, D2H)
            return o

        _h2d(xd, x)
        launch(xd, batch, n, int(math.log2(n)))    # (x, batch, N, log2N), in-place
        acl.rt.synchronize_stream(stream)
        out = _d2h(xd, (batch, n))
        acl.rt.free(xd)

        outf = out.astype(np.float32)
        mx = float(np.abs(outf - ref).max())
        den = float(np.abs(ref).max()) or 1.0
        rel = mx / den
        ok = bool(np.isfinite(outf).all()) and rel < TOLERANCE
        print(f"[check] {'PASS' if ok else 'FAIL'}: N={n} batch={batch} grid={grid} "
              f"rel_err={rel:.6f} max_abs={mx:.4e}", flush=True)
        if not ok:
            xf = x.astype(np.float32)
            print(f"[diag] out[0,:8]={np.round(outf[0,:8],3)}", flush=True)
            print(f"[diag] ref[0,:8]={np.round(ref[0,:8],3)}", flush=True)
            print(f"[diag] in [0,:8]={np.round(xf[0,:8],3)}", flush=True)
            raise RuntimeError(f"self-check FAILED for N={n}: rel_err={rel:.6f}")
        return rel
    finally:
        acl.rt.destroy_stream(stream)
        acl.rt.destroy_context(ctx)
        acl.rt.reset_device(dev_id)
        acl.finalize()


# ---------------------------------------------------------------------------
# DEVICE-ONLY below: torch + torch_npu + a real A5 device. Never runs under
# cannsim; the import is guarded so --check works without torch.
# ---------------------------------------------------------------------------
def _select_torch_device():
    import torch
    if os.environ.get("ASCEND_DEVICE") is not None:
        cands = [int(os.environ["ASCEND_DEVICE"])]
    else:
        cands = list(range(torch.npu.device_count() or 1))
    last = None
    for d in cands:
        try:
            torch.npu.set_device(d)
            t = torch.zeros(1, dtype=torch.float16).npu()
            torch.npu.synchronize()
            del t
            print(f"[torch] using NPU device {d}", flush=True)
            return d
        except Exception as ex:  # noqa: BLE001
            last = ex
    raise RuntimeError(f"no usable NPU for torch among {cands} (last: {last})")


def _torch_stream():
    import torch
    resolved = getattr(torch.npu.current_stream(), "_as_parameter_", None)
    if resolved is None:
        raise RuntimeError("could not resolve the current NPU stream pointer")
    return resolved


def _direct_launcher(n: int, grid: int):
    """Low-overhead timed launch: build the .so once, resolve the launch symbol +
    argtypes once, then dispatch directly (grid + stream baked) so the timed window
    measures the kernel, not the per-call MLIR-Context arg marshaling."""
    from ptodsl._runtime.launch import _normalize_stream_ptr
    kn = make_full(n).compile()
    lh = kn[grid, _torch_stream()]
    lh._ensure_launch_fn()                  # build .so + resolve symbol/argtypes ONCE
    fn = lh._launch_fn
    grid_c = ctypes.c_uint32(grid)
    stream_c = _normalize_stream_ptr(lh._stream)

    def launch(ptr, batch, n_, log2n):
        fn(grid_c, stream_c, ctypes.c_void_p(ptr), batch, n_, log2n)

    return launch


def _hbm_pool_count(buffer_bytes: int, buffers_per_slot: int) -> int:
    target = max(2, WORKING_SET_BYTES // (buffers_per_slot * buffer_bytes))
    cap = max(2, DEVICE_MEM_CAP // (buffers_per_slot * buffer_bytes))
    return max(2, min(target, cap))


def torch_copy_gbs(batch: int, n: int, trials: int) -> float:
    """Copy floor: torch device-to-device copy = the DMA/HBM ceiling. DEVICE-ONLY.

    The pool spans WORKING_SET_BYTES of distinct src/dst pairs (past the L2 knee ->
    real HBM), each copied once per window between two torch-NPU events with a single
    trailing sync, median over max(trials,15) samples. bytes = 2*pool*batch*n*2."""
    import torch
    per = batch * n * 2
    pool = _hbm_pool_count(per, buffers_per_slot=2)
    data = 2 * pool * per
    src = [torch.randn(batch, n, dtype=torch.float16).npu() for _ in range(pool)]
    dst = [torch.empty_like(src[0]) for _ in range(pool)]

    def timed():
        torch.npu.synchronize()
        s = torch.npu.Event(enable_timing=True)
        e = torch.npu.Event(enable_timing=True)
        s.record()
        for i in range(pool):
            dst[i].copy_(src[i])
        e.record()
        torch.npu.synchronize()
        ms = s.elapsed_time(e)
        return data / 1e9 / (ms / 1e3)

    timed()  # warmup
    samples = sorted(g for g in (timed() for _ in range(max(trials, 15)))
                     if math.isfinite(g) and 0.0 < g < SANITY_MAX_GBS)
    del src, dst
    torch.npu.empty_cache()
    return samples[len(samples) // 2] if samples else float("nan")


def hadamard_gbs(n: int, batch: int, grid: int, trials: int) -> float:
    """Median effective GM bandwidth of the transform. DEVICE-ONLY.

    GM bytes/invocation = 4*batch*n (fp16 load + store). Timed with WALL-CLOCK +
    torch.npu.synchronize() (NOT torch Events): the direct launcher runs the kernel
    on ptodsl's OWN baked stream, while torch Events record on torch's current
    stream -- that mismatch made elapsed_time() bracket an empty region and report
    absurd GB/s. A blocking synchronize() waits for all streams, so (t1 - t0) is real
    end-to-end time. The transform is in-place and H@(H@x)=N*x, so each buffer is
    transformed AT MOST ONCE per trial (reset from x0 outside the timer) to avoid
    fp16 overflow to inf (which faults the AICORE)."""
    import torch
    launch = _direct_launcher(n, grid)
    data = 2 * batch * n * 2
    log2n = int(math.log2(n))
    reps = max(2, min(WORKING_SET_BYTES // (batch * n * 2),
                      DEVICE_MEM_CAP // (batch * n * 2),
                      max(1, 512 // grid)))
    x0 = torch.randn(batch, n, dtype=torch.float16).npu()
    pool = [torch.empty_like(x0) for _ in range(reps)]
    ptrs = [b.data_ptr() for b in pool]

    def timed():
        for b in pool:
            b.copy_(x0)              # reset -> bounded magnitude; OUTSIDE the timer
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for i in range(reps):
            launch(ptrs[i], batch, n, log2n)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        us = (t1 - t0) * 1e6 / reps
        return data / 1e9 / (us / 1e6)

    timed()  # warmup
    gs = sorted(g for g in (timed() for _ in range(max(trials, 15)))
                if math.isfinite(g) and 0.0 < g < SANITY_MAX_GBS)
    del x0, pool
    torch.npu.empty_cache()
    return gs[len(gs) // 2] if gs else float("nan")


# ---------------------------------------------------------------------------
# Modes.
# ---------------------------------------------------------------------------
def run_check(n: int, batch: int, grid: int) -> int:
    rel = self_check(n, batch, grid)
    return 0 if rel < TOLERANCE else 1


def run_nsweep(grid: int) -> int:
    rc = 0
    for n in SUPPORTED_NS:
        batch = default_batch(n, grid)
        try:
            rel = self_check(n, batch, grid, verbose=False)
            if rel >= TOLERANCE:
                rc = 1
        except Exception as ex:  # noqa: BLE001
            print(f"[check] FAIL: N={n} grid={grid} error={ex}", flush=True)
            rc = 1
    return rc


def run_bench(n: int, batch: int, grid: int, trials: int) -> int:
    _select_torch_device()
    cg = torch_copy_gbs(batch, n, trials)
    hg = hadamard_gbs(n, batch, grid, trials)
    frac = (hg / cg) if (hg and cg) else 0.0
    print(f"{'N':>5} {'batch':>8} {'grid':>5} {'had_GBs':>10} "
          f"{'copy_GBs':>10} {'frac':>8}")
    print(f"{n:>5} {batch:>8} {grid:>5} {hg:>10.1f} {cg:>10.1f} {frac:>8.4f}",
          flush=True)
    return 0


def parse_args(argv):
    ap = argparse.ArgumentParser(description="VMI fast Walsh-Hadamard (parametric N) runner")
    ap.add_argument("--n", type=int, default=256, choices=SUPPORTED_NS,
                    help="transform width N (default 256)")
    ap.add_argument("--batch", type=int, default=None,
                    help="batch rows (default = smallest valid = grid * 65536/N)")
    ap.add_argument("--grid", type=int, default=1,
                    help="launch grid (block_dim); the batch is split into G row bands")
    ap.add_argument("--check", action="store_true",
                    help="pyACL correctness self-check (default; torch-free, cannsim-safe)")
    ap.add_argument("--nsweep", action="store_true",
                    help="--check over every supported N")
    ap.add_argument("--bench", action="store_true",
                    help="DEVICE-ONLY effective GM bandwidth vs a torch D2D copy floor")
    ap.add_argument("--repeat", type=int, default=7, help="bench timing trials (median)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    grid = args.grid
    n = args.n
    batch = args.batch if args.batch is not None else default_batch(n, grid)

    if args.nsweep:
        sys.exit(run_nsweep(grid))
    if args.bench:
        sys.exit(run_bench(n, batch, grid, args.repeat))
    # default: correctness check
    sys.exit(run_check(n, batch, grid))


if __name__ == "__main__":
    main()
