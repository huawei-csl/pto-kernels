"""Three arms over an (M, K=N) grid: bf16 matmul, the vendor MXFP4 matmul, ours.

Method, and every part of it is load-bearing on a shared box:

* **L2 is evicted before every timed launch**, outside the timer, by copying a
  buffer twice the size of L2. That models a forward pass, where a layer's
  weights are touched once per token and the model dwarfs the 128 MB L2. A
  back-to-back loop instead leaves the weights resident and quietly favours
  whichever format is smaller.
* **Arms are interleaved inside one measurement window**, order rotated per
  rep. Measuring them in sequential blocks lets a drift in machine load land
  entirely on one arm; a 9% drift once faked a 4% difference that was really
  8%.
* **The wall clock over a synchronised launch**, not the NPU event timer, which
  has reported 82, 28, 7.6 and 24 us for one and the same launch.
* **Medians over enough reps to resolve**, with the observed spread printed
  next to every row. Believe a difference only once it is larger than that
  spread, and confirm it by running this more than once -- the vendor kernel is
  chosen per process, so a single process can settle a near-tie the wrong way.
* **A device-to-device copy reference per run.** If it collapses, another
  tenant is on the device and the ratios in that run are not comparable.

Both MXFP4 arms are timed on **pre-quantized operands**, so what is compared is
matmul against matmul. Quantization is a separate kernel and is not in either
number.
"""

import argparse
import csv
import gc
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa
import torch_npu  # noqa

from jit_util_mxfp4_matmul_a5 import (  # noqa
    MX_BLOCK,
    compile_kernel,
    load_matmul,
    tile_plan,
)

L2_BYTES = 128 << 20  # A5 L2; the eviction buffer is twice this in bf16 halves
WARMUP = 20
TARGET_WINDOW_US = 200_000.0  # per arm, per shape
CUBE_PEAK_TFS = 1692.0  # MXFP4 cube ceiling, for the reps estimate only
FIELDS = [
    "m",
    "m_run",
    "kn",
    "bf16_us",
    "vendor_us",
    "ours_us",
    "ours_vs_bf16",
    "vendor_vs_bf16",
    "ours_vs_vendor",
    "bf16_tfs",
    "vendor_tfs",
    "ours_tfs",
    "padded",
    "reps",
    "spread_pct",
    "copy_gbs",
]

MS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
KNS = [512, 1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384]


def host(array):
    """Operands are generated on the host and moved, never with device RNG."""
    return torch.from_numpy(np.ascontiguousarray(array)).npu()


def copy_reference(source, destination) -> float:
    """GB/s of a plain device-to-device copy, as this run's contention check."""
    for _ in range(5):
        destination.copy_(source)
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(20):
        destination.copy_(source)
    torch.npu.synchronize()
    seconds = (time.perf_counter() - start) / 20
    return 2 * source.numel() * source.element_size() / seconds / 1e9


def timed_medians(arms, reps, source, destination):
    """Interleaved, order-rotated, L2 cold before each timed launch."""
    for arm in arms.values():
        for _ in range(WARMUP):
            arm()
    torch.npu.synchronize()
    observed = {name: [] for name in arms}
    names = list(arms)
    for rep in range(reps):
        offset = rep % len(names)
        for name in names[offset:] + names[:offset]:
            destination.copy_(source)  # evict, untimed
            torch.npu.synchronize()
            start = time.perf_counter()
            arms[name]()
            torch.npu.synchronize()
            observed[name].append((time.perf_counter() - start) * 1e6)
    return {
        name: (
            statistics.median(samples),
            100 * (max(samples) - min(samples)) / statistics.median(samples),
        )
        for name, samples in observed.items()
    }


def vendor_arm(activations, weights):
    """torch_npu's MXFP4 matmul, with the operands quantized up front.

    The scales are (rows, K/64, 2) pairs of E8M0 and any 2D scale is refused;
    ``group_sizes=[1, 1, 32]`` is mandatory; and the weight and its scale must
    share transpose state.
    """
    a_packed, a_scale = torch_npu.npu_dynamic_mx_quant(activations)
    b_packed, b_scale = torch_npu.npu_dynamic_mx_quant(weights.t().contiguous())
    fp4, e8m0 = torch.float4_e2m1fn_x2, torch.float8_e8m0fnu
    view = (
        a_packed.view(fp4),
        b_packed.view(fp4).t(),
        b_scale.view(e8m0).transpose(0, 1),
        a_scale.view(e8m0),
    )

    def run():
        return torch_npu.npu_quant_matmul(
            view[0],
            view[1],
            view[2],
            pertoken_scale=view[3],
            output_dtype=torch.bfloat16,
            group_sizes=[1, 1, 32],
        )

    return run


def our_arm(matmul, m_run, k, n, rng):
    """Ours, on pre-packed nibbles, with the shape bound once.

    ``matmul.prepare`` validates and derives the tile plan up front and hands
    back a launcher that only launches. Timing the validating entry point
    instead charges this arm three ctypes round-trips per call, which at
    M=16 K=N=1024 read as 1.30x against the vendor where the kernel itself is
    2.3x -- the wrapper, not the kernel.
    """
    packed = (
        host(rng.integers(0, 255, (m_run, k // 2), dtype=np.uint8)),
        host(rng.integers(120, 136, (m_run, k // MX_BLOCK)).astype(np.uint8)),
        host(rng.integers(0, 255, (n, k // 2), dtype=np.uint8)),
        host(rng.integers(120, 136, (n, k // MX_BLOCK)).astype(np.uint8)),
    )
    launch, out = matmul.prepare(*packed, m=m_run)
    torch.npu.synchronize()
    return launch, packed, out


def measure(matmul, m, m_run, kn, source, destination, max_reps):
    """One grid point, three arms. Returns the CSV row.

    The operands live in this frame and nowhere else, so they are released when
    it returns -- rather than by a ``del`` at the end of a loop body, which the
    arm closures above would then be holding stale names for.
    """
    rng = np.random.default_rng(137)
    activations = host(
        np.asarray(rng.normal(0, 1, (m, kn)), dtype=np.float32)
    ).bfloat16()
    weights = host(
        np.asarray(rng.normal(0, 0.05, (kn, kn)), dtype=np.float32)
    ).bfloat16()
    bf16_out = torch.empty((m, kn), dtype=torch.bfloat16, device="npu")
    ours, _packed, _our_out = our_arm(matmul, m_run, kn, kn, rng)

    flop = 2.0 * m * kn * kn
    floor_us = max(6.0, flop / (CUBE_PEAK_TFS * 1e12) * 1e6)
    reps = int(min(max_reps, max(5, round(TARGET_WINDOW_US / floor_us))))
    result = timed_medians(
        {
            "bf16": lambda: torch.matmul(activations, weights, out=bf16_out),
            "vendor": vendor_arm(activations, weights),
            "ours": ours,
        },
        reps,
        source,
        destination,
    )
    bf16_us, bf16_spread = result["bf16"]
    vendor_us, vendor_spread = result["vendor"]
    ours_us, ours_spread = result["ours"]
    return {
        "m": m,
        "m_run": m_run,
        "kn": kn,
        "bf16_us": round(bf16_us, 3),
        "vendor_us": round(vendor_us, 3),
        "ours_us": round(ours_us, 3),
        "ours_vs_bf16": round(bf16_us / ours_us, 3),
        "vendor_vs_bf16": round(bf16_us / vendor_us, 3),
        "ours_vs_vendor": round(vendor_us / ours_us, 3),
        "bf16_tfs": round(flop / bf16_us / 1e6),
        "vendor_tfs": round(flop / vendor_us / 1e6),
        "ours_tfs": round(flop / ours_us / 1e6),
        "padded": int(m_run != m),
        "reps": reps,
        "spread_pct": round(max(bf16_spread, vendor_spread, ours_spread), 1),
    }


def already_done(path):
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            rows = {(int(r["m"]), int(r["kn"])) for r in csv.DictReader(handle)}
        if rows:
            print(f"  resuming: {len(rows)} rows already present")
        return rows, True
    except FileNotFoundError:
        return set(), False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="mxfp4_matmul_a5.csv")
    parser.add_argument("--kn", type=int, nargs="*", default=KNS)
    parser.add_argument("--m", type=int, nargs="*", default=MS)
    parser.add_argument("--max-reps", type=int, default=100)
    args = parser.parse_args()

    torch.npu.set_device(0)
    source = torch.ones(L2_BYTES, dtype=torch.bfloat16, device="npu")
    destination = torch.empty_like(source)
    copy_gbs = copy_reference(source, destination)
    print(f"\n  device-to-device copy reference: {copy_gbs:.0f} GB/s")
    print(f"  {torch.npu.get_device_name(0)}\n")

    done, had_rows = already_done(args.out)
    with open(args.out, "a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        if not had_rows:
            writer.writeheader()
            output.flush()

        header = (
            f"  {'M':>6} {'run':>6} {'K=N':>6} {'bf16':>9} {'vendor':>9} "
            f"{'ours':>9} {'o/v':>6} {'o/bf16':>7} {'reps':>5} {'spread':>7}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))

        added = 0
        for kn in args.kn:
            so = compile_kernel(kn, kn, verbose=False)
            matmul = load_matmul(so, kn, kn)
            plan = tile_plan(so)
            for m in args.m:
                if (m, kn) in done:
                    continue
                m_run = plan(m)[0]
                try:
                    row = measure(
                        matmul, m, m_run, kn, source, destination, args.max_reps
                    )
                except (RuntimeError, ValueError) as error:
                    first = str(error).splitlines()[0][:60]
                    print(f"  {m:>6} {m_run:>6} {kn:>6}  skipped: {first}", flush=True)
                else:
                    print(
                        f"  {m:>6} {m_run:>6} {kn:>6} {row['bf16_us']:>9.2f} "
                        f"{row['vendor_us']:>9.2f} {row['ours_us']:>9.2f} "
                        f"{row['ours_vs_vendor']:>6.2f} {row['ours_vs_bf16']:>7.2f} "
                        f"{row['reps']:>5} {row['spread_pct']:>6.1f}%",
                        flush=True,
                    )
                    writer.writerow({**row, "copy_gbs": round(copy_gbs)})
                    output.flush()
                    added += 1
                # The operands are held only by measure()'s frame, so returning
                # drops them; at M=32768 K=N=16384 that matters within one sweep.
                gc.collect()
                torch.npu.empty_cache()
    total = len(done) + added
    print(f"\n  +{added}, {total} of {len(args.m) * len(args.kn)} in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
