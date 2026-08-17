"""Bandwidth benchmark for mxfp4_quant_a5: ours vs PTO TQuant and vs torch_npu.

Reproduces every figure in README.md. Run it through run_benchmark.sh.

ONE CALL PATH. `ours` is a bare ctypes launch with preallocated outputs -- the
fastest way to call it -- and everything is judged against that:

  tquant     PTO's own MXFP4 tile op, reached the same way: this source built
             twice with only the four compute passes swapped. Identical tiling,
             buffering and DMA, so the difference is COMPUTE.
  torch_npu  npu_dynamic_mx_quant. Its schema returns freshly allocated tensors
             and has no out= or _out overload, so there is no preallocated form
             to call. Part of its gap against ours is therefore that allocation
             rather than the kernel, which the README states plainly.
  d2d_copy   a torch device-to-device copy over the same rotating pool: the DMA
             floor for the shape, the yardstick fast_hadamard_a5 reports against.

Bandwidth counts every byte the operation must move: 2K read plus K/2 + K/32
written, i.e. 2.53125 bytes per element, one formula for every arm.

TIMING. BRACKETS brackets per shape; each fires LAUNCHES launches between two
synchronizes and divides the wall clock by LAUNCHES, identically for every arm.
These are steady-state throughput figures, not single-call latency. Contenders
are interleaved one bracket at a time with a ROTATING order: under a fixed order
the first arm in each bracket absorbs the previous one's cache eviction, which
alone was enough to make a preallocated arm read slower than an allocating one.
The reported ratio is the median paired per-bracket ratio with a percentile
bootstrap 95% interval, and a shape whose interval spans 1.0 is unresolved.

That interval covers variation WITHIN a process only. torch_npu has been seen to
select a different kernel from one process to the next, so run several processes
with different --tag values and compare their spread before believing a small
margin on the api pair.

Every contender is gated bit-exact against torch_npu before it is timed, so a
wrong kernel cannot report a fast number.

The vendor arm is whatever CANN is on ASCEND_HOME_PATH: torch_npu resolves
libopapi_nn.so from there, so its numbers move with the toolkit and rows are
comparable only within one version.

Emits build/pairs_<axis>_<tag>.csv, which is what the plotting scripts read.
"""

import argparse
import csv
import random
import statistics
import sys
import time
from pathlib import Path

import ctypes
import os
import subprocess

import torch
import torch_npu  # noqa: F401  (registers the npu backend)

from jit_util_mxfp4_a5 import MX_BLOCK, SUPPORTED_K, row_quantum

HERE = Path(__file__).resolve().parent
BUILDDIR = HERE / "build"

# The same widths as the fast_hadamard_a5 sweep (PR #221), keeping only those
# with an EVEN block count: torch_npu lays its scales out as (batch, K/64, 2), so
# an odd block count does not fit its layout. The kernel supports all 26 widths;
# the rest are covered by the tests.
HADAMARD_NS = [32, 64, 128, 256, 512, 1024, 2048]
K_LIST = [k for k in HADAMARD_NS if k in SUPPORTED_K and (k // MX_BLOCK) % 2 == 0]
K_SWEEP_BATCH = 65536  # fixed across the K sweep
BATCH_SWEEP_K = 4096
WORKING_SET_BYTES = 1024 * 1024 * 1024
POOL_MIN, POOL_MAX = 2, 64
LAUNCHES = 40
BRACKETS = 64
BOOTSTRAP = 2000
SEED = 20260811
WARMUP = 5
VENDOR_DST_TYPE = 296

BYTES_PER_ELEM = 2.0 + 0.5 + 1.0 / MX_BLOCK


def raw_launcher(so_path, k):
    """ctypes launcher with no wrapper: the path the TQuant arm also uses."""
    from jit_util_mxfp4_a5 import VECTOR_CORES, current_stream_ptr

    handle = ctypes.CDLL(str(so_path))
    launch = handle.call_mxfp4_quant
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

    def call(tensor, packed, scales):
        launch(
            VECTOR_CORES,
            current_stream_ptr(),
            ctypes.c_void_p(tensor.data_ptr()),
            ctypes.c_void_p(packed.data_ptr()),
            ctypes.c_void_p(scales.data_ptr()),
            tensor.shape[0],
            k,
        )

    return call


class TQuantUnavailable(RuntimeError):
    """PTO has no MXFP4 quantizer -- it arrived in 9.1.0, so 9.0.0 cannot build."""


def build_tquant(k):
    """Build the kernel with its four compute passes replaced by PTO TQuant."""
    home = os.environ["ASCEND_HOME_PATH"]
    out = BUILDDIR / "mxfp4_a5_tquant.so"
    obj = BUILDDIR / "mxfp4_a5_tquant.o"
    BUILDDIR.mkdir(parents=True, exist_ok=True)
    source = HERE / "mxfp4_quant_a5.cpp"
    flags = (
        f"-xcce --cce-aicore-arch=dav-c310-vec -DREGISTER_BASE -DMXFP4_TQUANT "
        f"-std=c++17 -O2 -fPIC -Wno-ignored-attributes -Wno-macro-redefined "
        f"-mllvm -cce-aicore-stack-size=0x8000 "
        f"-mllvm -cce-aicore-function-stack-size=0x8000 "
        f"-mllvm -cce-aicore-addr-transform "
        f"-mllvm -cce-aicore-dcci-insert-for-scalar=false -Xhost-start -Xhost-end "
        f"-I{home}/aarch64-linux/include -I{home}/include"
    ).split()
    # PTO 9.1.0 release inserted a `bool Exp2DStrided` template parameter that
    # 9.1.0-beta.3 lacks, and the tile types sit in a non-deduced position, so one
    # spelling cannot serve both. Try the beta.3 form, then the release form.
    errors = []
    for extra in ([], ["-DMXFP4_TQUANT_EXP2D"]):
        compile_step = subprocess.run(
            [f"{home}/bin/bisheng", *flags, *extra, "-c", str(source), "-o", str(obj)],
            capture_output=True,
            text=True,
            check=False,
        )
        if compile_step.returncode == 0:
            break
        errors.append(
            f"{' '.join(extra) or 'no extra flag'}: "
            f"{compile_step.stderr.strip()[-300:]}"
        )
    else:
        raise TQuantUnavailable("\n".join(errors))
    subprocess.run(
        [
            f"{home}/bin/bisheng",
            "-fPIC",
            "-shared",
            "--cce-fatobj-link",
            f"-Wl,-soname,{out.name}",
            str(obj),
            "-o",
            str(out),
        ],
        check=True,
    )
    handle = ctypes.CDLL(str(out))
    launch = handle.call_mxfp4_quant
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
    from jit_util_mxfp4_a5 import VECTOR_CORES, current_stream_ptr

    def call(tensor, packed, scales):
        launch(
            VECTOR_CORES,
            current_stream_ptr(),
            ctypes.c_void_p(tensor.data_ptr()),
            ctypes.c_void_p(packed.data_ptr()),
            ctypes.c_void_p(scales.data_ptr()),
            tensor.shape[0],
            k,
        )

    return call


def batch_for(k):
    """Fixed batch, rounded up to the quantum so no shape takes the pad path."""
    quantum = row_quantum(k)
    return -(-K_SWEEP_BATCH // quantum) * quantum


def make_pool(batch, k):
    per_buffer = batch * k * 2
    depth = max(POOL_MIN, min(POOL_MAX, WORKING_SET_BYTES // max(per_buffer, 1)))
    # generated on device in bf16 directly: a host randn of this size costs
    # minutes, and going via fp32 on device doubles peak memory for no benefit --
    # enough to fail a 538 MB working set on a device with 125 GB free
    pool = [
        torch.randn(batch, k, dtype=torch.bfloat16, device="npu") for _ in range(depth)
    ]
    torch.npu.synchronize()
    return pool


def interleaved_micros(contenders, pool):
    """Time every contender round-robin, one bracket each, keeping all samples.

    Bracket i of every contender sees the same machine, which paired_speedup
    requires. The order ROTATES per bracket: with a fixed order the first contender
    pays whatever the previous bracket's last one left in cache, which is enough
    to make a preallocated arm read slower than an allocating one.
    """
    for _, call in contenders:
        for i in range(WARMUP):
            call(pool[i % len(pool)])
    torch.npu.synchronize()
    samples = {name: [] for name, _ in contenders}
    for bracket in range(BRACKETS):
        turn = bracket % len(contenders)
        for name, call in contenders[turn:] + contenders[:turn]:
            torch.npu.synchronize()
            start = time.perf_counter()
            for i in range(LAUNCHES):
                call(pool[i % len(pool)])
            torch.npu.synchronize()
            samples[name].append((time.perf_counter() - start) * 1e6 / LAUNCHES)
    return samples


def paired_speedup(ours, theirs):
    """Median per-bracket ratio theirs/ours, with a percentile bootstrap 95% CI.

    Paired, because the brackets are interleaved: the ratio within a bracket
    cancels whatever drift both contenders saw. The interval is a bootstrap over
    those ratios, NOT their min-max range -- a range widens as you sample more,
    so using it as the test would make more evidence look like less. Resolved
    means the interval excludes 1.0.
    """
    pairs = [t / o for o, t in zip(ours, theirs) if o > 0 and t > 0]
    if len(pairs) < 3:
        return 0.0, 0.0, 0.0, False
    rng = random.Random(SEED)
    draws = sorted(
        statistics.median(rng.choices(pairs, k=len(pairs))) for _ in range(BOOTSTRAP)
    )
    return (
        statistics.median(pairs),
        draws[int(0.025 * BOOTSTRAP)],
        draws[int(0.975 * BOOTSTRAP) - 1],
        not (draws[int(0.025 * BOOTSTRAP)] <= 1.0 <= draws[int(0.975 * BOOTSTRAP) - 1]),
    )


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def gate(name, ours_out, other_out):
    """Bit-exactness before timing: a fast arm that computes nothing is the
    signature failure on this part."""
    (oq, os_), (tq, ts) = ours_out, other_out
    torch.npu.synchronize()
    packed = (oq == tq).float().mean().item()
    scale = (os_ == ts).float().mean().item()
    wrote = bool(oq.any().item())
    assert wrote, f"{name}: our arm wrote nothing"
    return packed, scale, wrote


COPY_BYTES_PER_ELEM = 4.0  # bf16 read + bf16 written


def copy_contender(pool):
    """A torch device-to-device copy, as the DMA floor for this shape.

    fast_hadamard_a5 judges its transform against exactly this: for a
    memory-bound op the honest yardstick is not a rival kernel but a plain copy
    of the same bytes. It moves 4 B/element against the quantizer's 2.53, so the
    two are compared as achieved bandwidth, each counting the bytes it moves.

    It reads the SAME rotating pool as the other arms and writes to a rotating
    destination. Copying one fixed buffer instead measured 5306 GB/s at K=256 --
    above the HBM ceiling, because both buffers fit in the 128 MB L2.
    """
    dsts = [torch.empty_like(pool[0]) for _ in pool]
    state = {"i": 0}

    def run(x):
        dst = dsts[state["i"] % len(dsts)]
        state["i"] += 1
        return dst.copy_(x)

    return ("d2d_copy", run)


def measure_shape(k, batch, want_tquant=True):
    """Every contender at one shape, all judged against ours on a raw launch.

    ours is a bare ctypes launch with preallocated outputs -- the fastest way to
    call it. Its rivals:

      tquant     PTO's own MXFP4 tile op, reached the same way: the same source
                 built twice with only the four compute passes swapped, outputs
                 preallocated. Identical tiling, buffering and DMA, so this
                 isolates COMPUTE.
      torch_npu  npu_dynamic_mx_quant, which ALLOCATES its two outputs. Its
                 schema has no out= and no _out overload, so there is no
                 preallocated form to call; the allocation is part of using it.
                 Some of the gap against ours is therefore that allocation and
                 not the kernel, and the README says so.
      d2d_copy   a torch device-to-device copy over the same rotating pool: the
                 DMA floor for the shape, as fast_hadamard_a5 reports.
    """
    from jit_util_mxfp4_a5 import compile_kernel

    pool = make_pool(batch, k)
    blocks = k // MX_BLOCK
    qa = torch.empty((batch, k // 2), dtype=torch.uint8, device="npu")
    sa = torch.empty((batch, blocks), dtype=torch.uint8, device="npu")
    qb = torch.empty((batch, k // 2), dtype=torch.uint8, device="npu")
    sb = torch.empty((batch, blocks), dtype=torch.uint8, device="npu")
    mine = raw_launcher(compile_kernel(verbose=False), k)

    vendor = getattr(torch_npu, "npu_dynamic_mx_quant", None)
    if vendor is None:
        raise RuntimeError("npu_dynamic_mx_quant missing: no baseline")

    contenders = [("ours_raw", lambda x: mine(x, qa, sa))]
    mine(pool[0], qa, sa)
    matches = {}

    if want_tquant:
        theirs = build_tquant(k)
        contenders.append(("tquant", lambda x: theirs(x, qb, sb)))
        theirs(pool[0], qb, sb)
        matches["tquant"] = gate("tquant", (qa, sa), (qb, sb))

    contenders.append(("torch_npu", lambda x: vendor(x, dst_type=VENDOR_DST_TYPE)))
    tq, ts = vendor(pool[0], dst_type=VENDOR_DST_TYPE)
    ts = ts.reshape(ts.shape[0], -1)[:, :blocks]
    matches["torch_npu"] = gate("torch_npu", (qa, sa), (tq, ts))

    contenders.append(copy_contender(pool))

    samples = interleaved_micros(contenders, pool)
    samples = interleaved_micros(contenders, pool)
    ours = contenders[0][0]
    rows = []
    for name, _ in contenders:
        taken = samples[name]
        micros = statistics.median(taken)
        # the copy moves 4 B/element where the quantizer moves 2.53, so each
        # counts the bytes it actually moves and the two meet as bandwidth
        per_elem = COPY_BYTES_PER_ELEM if name == "d2d_copy" else BYTES_PER_ELEM
        if name in (ours, "d2d_copy"):
            # the copy is a floor, not a rival: it moves 4 B/element against the
            # quantizer's 2.53, so a time ratio between them means nothing. The
            # two meet as bandwidth only.
            ratio, low, high, resolved = 1.0, 1.0, 1.0, 0
        else:
            ratio, low, high, res = paired_speedup(samples[ours], taken)
            resolved = int(res)
        match = matches.get(name, (1.0, 1.0))
        rows.append(
            {
                "pair": "raw",
                "k": k,
                "batch": batch,
                "contender": name,
                "micros": round(micros, 3),
                "p_lo": round(min(taken), 3),
                "p_hi": round(max(taken), 3),
                "spread_pct": round(100 * (max(taken) - min(taken)) / micros, 1),
                "gbs": round(batch * k * per_elem / (micros * 1e-6) / 1e9, 1),
                "packed_match": round(match[0], 6),
                "scale_match": round(match[1], 6),
                "speedup": round(ratio, 4),
                "speedup_lo": round(low, 4),
                "speedup_hi": round(high, 4),
                "resolved": resolved,
                "brackets_n": len(taken),
                "status": "ok",
            }
        )
    pool.clear()
    torch.npu.empty_cache()
    return rows


def tquant_builds():
    """Report whether the TQuant variant compiles, so the arm can be skipped."""
    try:
        build_tquant(K_LIST[0])
        return True
    except TQuantUnavailable as exc:
        print(f"skipping the raw pair: PTO here has no MXFP4 quantizer\n  {exc}")
        return False


def cell(got, name, width=8):
    """One bandwidth cell, or a dash when that contender did not run."""
    return f"{got[name]['gbs']:>{width}.0f}" if name in got else f"{'--':>{width}}"


def main():
    parser = argparse.ArgumentParser(
        description="ours on a raw launch against TQuant, torch_npu and a copy"
    )
    parser.add_argument(
        "--tag", default="1", help="suffix for the CSV; use one per process"
    )
    parser.add_argument("--axis", choices=["k", "batch"], default="k")
    parser.add_argument("--ks", default="", help="comma list overriding K_LIST")
    args = parser.parse_args()

    # PR 221 sweeps ROWS PER LAUNCH over this list at a fixed width. Only 4096
    # and 8192 of those values are legal K here (SUPPORTED_K stops at 14336), so
    # the larger ones are the batch axis, not widths.
    pr221_batches = (4096, 8192, 16384, 32768, 65536, 131072)
    widths = tuple(int(v) for v in args.ks.split(",")) if args.ks else K_LIST
    for w in widths:
        assert w in SUPPORTED_K, f"K={w} has no instantiation"
    shapes = (
        [(k, batch_for(k)) for k in widths]
        if args.axis == "k"
        else [(BATCH_SWEEP_K, b) for b in pr221_batches]
    )
    label = "K" if args.axis == "k" else "batch"
    want_tquant = tquant_builds()

    out = []
    header = f"{label:>7} {'ours':>8} {'TQuant':>8} {'torch_npu':>10} {'copy':>8}"
    print(f"\n=== bandwidth (GB/s), ours on a raw launch, by {label} ===")
    print(header)
    for width, batch in shapes:
        rows = measure_shape(width, batch, want_tquant)
        out += rows
        got = {r["contender"]: r for r in rows}
        key = width if args.axis == "k" else batch
        print(
            f"{key:>7} {cell(got, 'ours_raw')} {cell(got, 'tquant')} "
            f"{got['torch_npu']['gbs']:>10.0f} {cell(got, 'd2d_copy')}"
        )
    write_csv(BUILDDIR / f"pairs_{args.axis}_{args.tag}.csv", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
