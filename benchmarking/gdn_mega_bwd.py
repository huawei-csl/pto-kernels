"""Fast GDN backward checks/timing; --extensive writes three evaluation PNGs.

Build first: cmake --build build --target pto_kernels_ops -j 2
"""

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "python"), str(ROOT / "tests")]
os.environ.setdefault("OMP_NUM_THREADS", "4")
SUPPORTED_HEADS = (8, 12, 16, 24, 32, 48, 64)
GRADIENTS = ("dq", "dk", "dv", "dg", "dbeta", "dh0")
# label, sequence lengths, value heads, QK heads, input mode, dtype, saved caches
DEV_CASES = (
    ("Cached tail", [129], 8, 2, "both", "float16", True),
    ("Cached identical keys", [129], 8, 2, "replace", "float16", True),
    ("BF16 packed/empty", [0, 1, 17, 0], 8, 2, "both", "bfloat16", False),
    ("FP32 zero beta", [129], 8, 2, "zero_beta", "float32", False),
)
AUDIT_CASES = tuple(
    (label, [length], heads, keys, mode, "float16", True)
    for label, length, heads, keys, mode in (
        ("Random · 128", 128, 8, 2, "both"),
        ("Tail · 129", 129, 8, 2, "both"),
        ("Zero beta · 129", 129, 8, 2, "zero_beta"),
        ("Identical keys · 257", 257, 8, 2, "replace"),
        ("Weak decay · 513", 513, 8, 2, "weak"),
        ("Near-dependent + weak · 257", 257, 8, 2, "correlated_weak"),
        ("Strong decay · 256", 256, 8, 2, "strong_decay"),
        ("Small beta · 384 · H64/64", 384, 64, 64, "tiny_beta"),
        ("Correlated · 512 · H64/16", 512, 64, 16, "correlated"),
        ("Grouped · 8192", 8192, 8, 2, "both"),
    )
)


@lru_cache(None)
def runtime():
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(os.environ.get("NPU_DEVICE", "npu:0"))
    torch.npu.config.allow_internal_format = False
    return torch


def check(case, cache_dtype, flashqla=False):
    torch = runtime()
    import test_gdn_mega_bwd as tests

    label, lengths, heads, keys, mode, dtype, cached = case
    row = dict(
        label=label,
        lengths=lengths,
        heads=heads,
        key_heads=keys,
        mode=mode,
        dtype=dtype,
        cached=cached,
        status="passed",
        gradients={},
    )
    records = []
    try:
        tests.test_backward(
            lengths,
            heads,
            keys,
            mode,
            getattr(torch, dtype),
            f"npu:{torch.npu.current_device()}",
            "both" if cached else None,
            getattr(torch, cache_dtype),
            records,
            flashqla_bound=flashqla,
        )
    except AssertionError as error:
        row.update(status="failed", failure=str(error))
    finally:
        tests._saved_references.clear()  # No oracle reuse between these cases.
    row["gradients"] = {r["gradient"]: r for r in records}
    print(
        f"{label}: {row['status']}"
        + (f"\n{row['failure']}" if "failure" in row else ""),
        flush=True,
    )
    return row


def measure(shape, cache_dtype, repeats, profile=None):
    torch = runtime()
    from pto_kernels import pto_gdn_mega_bwd as backward
    from test_gdn_mega import _prepare_gdn_mega, pto_gdn_mega

    length, heads, keys = shape
    torch.manual_seed(42)
    device = f"npu:{torch.npu.current_device()}"
    q = torch.randn(1, length, keys, 128, device=device, dtype=torch.float16) * 0.1
    k = torch.nn.functional.normalize(torch.randn_like(q), dim=-1)
    v = torch.randn(1, length, heads, 128, device=device, dtype=q.dtype)
    g = -torch.rand(1, length, heads, device=device) * 0.1
    beta, dout = torch.rand_like(g).half(), torch.randn_like(v)
    cu = torch.tensor([0, length], device=device, dtype=torch.int32)
    launch = _prepare_gdn_mega(q, k, v, g, beta, cu)
    cache = torch.empty(
        (length + 127) // 128,
        heads,
        128,
        128,
        device=device,
        dtype=getattr(torch, cache_dtype),
    )
    gates = tuple(launch[name] for name in ("g_t", "g_sum", "beta_t"))
    calls = {
        "forward": lambda: pto_gdn_mega(**launch, inverse_cache=cache),
        "backward": lambda: backward(
            q, k, v, g, beta, dout, inverse_cache=cache, gate_cache=gates
        ),
    }
    samples = {name: [] for name in calls}
    # Forward always precedes backward, producing the exact buffers it consumes.
    # Two warmups exclude compilation; output destruction is outside timing.
    for trial in range(2 + repeats):
        for name, call in calls.items():
            torch.npu.synchronize()
            start = time.perf_counter()
            result = call()
            torch.npu.synchronize()
            if trial >= 2:
                samples[name].append((time.perf_counter() - start) * 1000)
            del result
    if profile:
        import torch_npu

        profile = profile.resolve()
        profile.mkdir(parents=True, exist_ok=True)
        previous = Path.cwd()
        try:
            os.chdir(profile)  # Keep CANN's intermediate profiler files here too.
            with torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ]
            ) as trace:
                calls["backward"]()
                torch.npu.synchronize()
            trace.export_chrome_trace("trace.json")
        finally:
            os.chdir(previous)
    return dict(
        shape=list(shape),
        samples_ms=samples,
        median_ms={name: statistics.median(xs) for name, xs in samples.items()},
    )


def fla_shape(item):
    s = item["shape"]
    if (s["batch_size"], s["query_key_head_dimension"], s["value_head_dimension"]) != (
        1,
        128,
        128,
    ):
        raise ValueError(f"Unsupported FLA shape: {s}")
    for name in ("forward", "backward"):
        value = item["latency"][name]["median_ms"]
        if not 0 < value < float("inf"):
            raise ValueError(f"Invalid FLA {name} timing: {value}")
    return (s["sequence_length_tokens"], s["value_heads"], s["query_key_heads"])


def supported(shape):
    length, heads, keys = shape
    return length > 0 and heads in SUPPORTED_HEADS and keys > 0 and heads % keys == 0


def provenance():
    sources = [
        p
        for pattern in (
            "csrc/kernel/gdn_mega/*.*",
            "csrc/kernel/gdn_mega_bwd/*.*",
            "csrc/host/torch_gdn_mega*.h",
            "csrc/host/pybind11.cpp",
            "CMakeLists.txt",
            "python/pto_kernels/*.py",
            "tests/*gdn*py",
            "benchmarking/*gdn*py",
        )
        for p in ROOT.glob(pattern)
    ]
    git = ["git", "-c", f"safe.directory={ROOT}"]
    return dict(
        timestamp=datetime.now(timezone.utc).isoformat(),
        commit=subprocess.check_output(
            [*git, "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        dirty=subprocess.check_output(
            [*git, "status", "--porcelain"], cwd=ROOT, text=True
        ),
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(sources)
            if p.is_file()
        },
        forward_sha256=hashlib.sha256(
            (ROOT / "build/lib/libno_workspace_kernel.so").read_bytes()
        ).hexdigest(),
        timing="Median synchronized wall ms; two warmups; backward includes allocation/enqueue, forward uses preallocated buffers. Compilation, input generation and TP communication excluded.",
        inputs="FP16 q/k/v/beta/dout, FP32 g; seed42; q=.1*randn, normalized k, v/dout=randn, beta=uniform, g=-.1*uniform; h0/dht omitted; D=C=128.",
        cache="Forward saves inverse + gate/beta buffers; backward consumes those exact buffers.",
        bound="Per gradient: max|kernel-golden| <= 2*max|reordered low-precision Torch-golden| + 1e-5; FP32 golden for FP16/BF16, FP64 for FP32.",
        fla_caveat="Exact shapes; different dtype, chunk size, inputs, state outputs and allocation/autograd overhead. FLA/PTO is a reported latency ratio, not a controlled same-workload comparison.",
    )


def save(report, destination):
    temporary = destination / "results.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(destination / "results.json")


def extensive(args):
    fla = json.loads(args.fla.read_text())
    fla_shapes = [fla_shape(item) for item in fla["results"]]
    if not fla_shapes or len(set(fla_shapes)) != len(fla_shapes):
        raise ValueError("FLA results must contain nonempty, unique shapes")
    grid = [
        dict(heads=h, tp=tp, shape=[length, h // tp, 16 // tp])
        for h in (64, 48, 32, 16)
        for tp in (1, 2, 4, 8)
        for length in (256, 1024, 4096, 16384, 32768)
    ]
    shapes = {tuple(row["shape"]) for row in grid} | set(fla_shapes)
    groups = {}
    for shape in sorted(shapes):
        if supported(shape):
            groups.setdefault(shape[1:], []).append(shape)
    report = dict(
        status="incomplete",
        cache_dtype=args.cache_dtype,
        metadata=provenance(),
        fla=fla,
        fla_sha256=hashlib.sha256(args.fla.read_bytes()).hexdigest(),
        grid=grid,
        measurements=[],
        precision=[],
        flashqla_bound=args.flashqla_bound,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    save(report, args.output)
    # Separate processes release large NPU allocations between head groups.
    # Each distinct local shape is timed once, even if several TP cells use it.
    for (heads, keys), group in groups.items():
        print(
            f"Timing Hv/Hqk={heads}/{keys}, lengths={[s[0] for s in group]}", flush=True
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--cache-dtype",
            args.cache_dtype,
            "--repeats",
            str(args.repeats),
        ]
        for shape in group:
            command += ["--shape", *map(str, shape)]
        result = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
        log = args.output / f"h{heads}_k{keys}.log"
        log.write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"Timing failed; see {log}")
        rows = [
            json.loads(line)
            for line in result.stdout.splitlines()
            if line.startswith('{"shape":')
        ]
        if [tuple(r["shape"]) for r in rows] != group:
            raise RuntimeError(f"Incomplete timing output; see {log}")
        report["measurements"].extend(rows)
        save(report, args.output)
    for case in AUDIT_CASES:
        report["precision"].append(check(case, args.cache_dtype, args.flashqla_bound))
        save(report, args.output)
    report["status"] = (
        "failed"
        if any(r["status"] != "passed" for r in report["precision"])
        else "passed"
    )
    save(report, args.output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--extensive",
        action="store_true",
        help="Full TP/FLA timing sweep and numerical audit",
    )
    modes.add_argument(
        "--plot-only",
        action="store_true",
        help="Redraw saved results.json; no Torch/NPU or FLA file needed",
    )
    modes.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--shape",
        type=int,
        nargs=3,
        action="append",
        metavar=("TOKENS", "HV", "HK"),
        help="Development timing shape; repeatable",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        help="Timed samples per direction (development: 3, extensive: 5)",
    )
    parser.add_argument(
        "--cache-dtype", choices=("float16", "bfloat16", "float32"), default="float16"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("benchmark_data/gdn_mega_bwd/eval")
    )
    candidates = [
        Path(base) / "benchmarks/fla-gdn-9e82a80/sweep/gdn_ascend_benchmark.json"
        for base in ("/mounted_home", "/home/fskogh")
    ]
    parser.add_argument(
        "--fla", type=Path, default=next((p for p in candidates if p.exists()), None)
    )
    parser.add_argument(
        "--flashqla-bound",
        action="store_true",
        help="Also check relative L2 <= 2%% against FP64; include it in precision.png",
    )
    parser.add_argument(
        "--bench-only", action="store_true", help="Skip development correctness checks"
    )
    parser.add_argument(
        "--profile",
        type=Path,
        help="Save a warmed backward trace for the last development shape",
    )
    args = parser.parse_args()
    args.repeats = (
        args.repeats if args.repeats is not None else (5 if args.extensive else 3)
    )
    if args.repeats < 1 or any(not supported(s) for s in args.shape or []):
        parser.error(
            "Use positive repeats and supported shapes (Hv=8/12/16/24/32/48/64, Hv divisible by Hqk)"
        )
    if (args.extensive or args.plot_only) and (
        args.shape or args.bench_only or args.profile
    ):
        parser.error(
            "--shape, --bench-only and --profile apply only to development runs"
        )
    if args.extensive and (args.fla is None or not args.fla.is_file()):
        parser.error("--extensive requires --fla PATH to saved FLA timings")
    args.output = args.output.resolve()
    if args.extensive or args.plot_only:
        from benchmarking.gdn_mega_bwd_plot import plot

        report = (
            json.loads((args.output / "results.json").read_text())
            if args.plot_only
            else extensive(args)
        )
        plot(report, args.output)
        print(f"Saved heatmap.png, precision.png, speedup.png in {args.output}")
        return int(report["status"] != "passed")
    if not args.worker and not args.bench_only:
        for case in DEV_CASES:
            if check(case, args.cache_dtype, args.flashqla_bound)["status"] != "passed":
                return 1
    shapes = args.shape or [(2048, 16, 2)]
    for i, shape in enumerate(shapes):
        row = measure(
            shape,
            args.cache_dtype,
            args.repeats,
            args.profile if i == len(shapes) - 1 else None,
        )
        if args.worker:
            print(json.dumps(row), flush=True)
        else:
            fwd, bwd = (row["median_ms"][name] for name in ("forward", "backward"))
            print(
                f"{shape}: cached forward {fwd:.3f} ms; backward {bwd:.3f} ms ({bwd/fwd:.2f}x); inverse={args.cache_dtype}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
