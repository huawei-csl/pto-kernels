"""Build, verify, benchmark and plot GDN mega. Run inside the CANN environment.

Raw measurements go to ignored benchmark_data/. Only Markdown and two PNGs
are published to docs/gdn_mega. Use --help for commands and settings.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
HEADS = (8, 12, 16, 24, 32, 48, 64)
DEFAULT_PAIRS = [
    (8, 2),
    (12, 3),
    (16, 4),
    (24, 6),
    (24, 8),
    (32, 8),
    (48, 12),
    (48, 16),
    (64, 16),
]
LENGTHS = [128, 256, 1024, 4096, 8192, 16384, 32768]
sys.path.insert(0, str(ROOT / "tests"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def error_metrics(*args, **kwargs):
    from test_gdn_mega import error_metrics as measure

    return measure(*args, **kwargs)


def parse_head_configs(values):
    try:
        pairs = (
            [tuple(map(int, v.split("/"))) for v in values]
            if values
            else DEFAULT_PAIRS.copy()
        )
        if any(
            len(p) != 2 or p[0] not in HEADS or p[1] <= 0 or p[0] % p[1] for p in pairs
        ):
            raise ValueError()
        if len(set(pairs)) != len(pairs):
            raise ValueError()
        return pairs
    except (ValueError, ZeroDivisionError):
        raise ValueError(
            "Use distinct H/Hg pairs with supported H and positive Hg dividing H"
        ) from None


def numerical_failure(row, variant):
    return (
        row[variant]["status"] == "failed"
        or row.get(f"{variant}_accuracy_status") == "failed"
    )


def library_env(directory):
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = (
        f"{Path(directory).resolve()}:{environment.get('LD_LIBRARY_PATH', '')}"
    )
    environment["PYTHONPATH"] = f"{ROOT / 'python'}:{environment.get('PYTHONPATH', '')}"
    environment.setdefault("NPU_DEVICE", "npu:0")
    environment.setdefault(
        "GDN_REFERENCE_CACHE", str(ROOT / "benchmark_data/gdn_mega/reference_cache")
    )
    environment["GDN_EXPECTED_LIBRARY_SHA256"] = sha256(
        Path(directory) / "libno_workspace_kernel.so"
    )
    return environment


FLAGS = [
    "-DMEMORY_BASE",
    "-Dno_workspace_kernel_EXPORTS",
    "-I/usr/local/Ascend/driver/kernel/inc",
    "-O0",
    "-g",
    "-fPIC",
    "-D_FORTIFY_SOURCE=2",
    "-O2",
    "-std=c++17",
    "-Wno-macro-redefined",
    "-Wno-ignored-attributes",
    "-fstack-protector-strong",
    "--npu-arch=dav-2201",
    "--asc-aicore-lang",
]


QUICK_CASES = (
    "ordinary_seed2027",
    "weak_seed119",
    "packed_empty",
    "packed_empty_zero_state",
    "correlated_beta1",
    "nearly_correlated_beta1",
    "fla_positive_strong_tail75",
    "critical_nearly_correlated_3011_128.0",
    "critical_tiny_gate_-1e-07_4097",
)


def build(source, destination, heads):
    destination.mkdir(parents=True, exist_ok=True)
    sources = sorted(p for p in source.iterdir() if p.suffix in (".hpp", ".cpp"))
    hashes = {p.name: sha256(p) for p in sources}
    for path in sources:
        shutil.copyfile(path, destination / path.name)
    source_text = (source / "kernel_gdn_mega.cpp").read_text()
    signature = re.search(
        r'extern "C" void pto_launch_gdn_mega\([\s\S]*?\)\s*{', source_text
    )
    if signature is None:
        raise ValueError("Cannot identify the production launch signature")
    declaration = signature.group()[:-1].strip()
    arguments = [
        p.strip().split()[-1].lstrip("*")
        for p in declaration.split("(", 1)[1][:-1].split(",")
    ]
    dispatch = re.compile(r"case\s+(\d+):\s*LAUNCH_MEGA\(\1\);\s*break;")
    calls = dispatch.findall(source_text)
    if tuple(map(int, calls)) != HEADS:
        raise ValueError("Expected the complete production head dispatch")

    def compile_head(head):
        text = dispatch.sub(
            lambda m: m.group() if int(m[1]) == head else "",
            source_text,
        )
        text = text.replace(
            "void pto_launch_gdn_mega(", f"void pto_launch_gdn_mega_{head}("
        )
        path = destination / f"head_{head}.cpp"
        path.write_text(text)
        with (destination / f"head_{head}.log").open("w") as log:
            subprocess.run(
                ["bisheng", *FLAGS, "-c", str(path), "-o", str(path.with_suffix(".o"))],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(heads)) as executor:
        list(executor.map(compile_head, heads))
    shim = "#include <cstdint>\n" + "\n".join(
        declaration.replace(
            "void pto_launch_gdn_mega(", f"void pto_launch_gdn_mega_{h}("
        )
        + ";"
        for h in heads
    )
    shim += "\n" + declaration + "{\n switch(num_heads){\n"
    shim += "\n".join(
        f' case {h}: pto_launch_gdn_mega_{h}({", ".join(arguments)}); break;'
        for h in heads
    )
    shim += "\n }\n}\n"
    (destination / "launch.cpp").write_text(shim)
    subprocess.run(
        [
            "bisheng",
            "-fPIC",
            "-O2",
            "-c",
            str(destination / "launch.cpp"),
            "-o",
            str(destination / "launch.o"),
        ],
        check=True,
    )
    library = destination / "libno_workspace_kernel.so"
    subprocess.run(
        [
            "bisheng",
            "-shared",
            "-s",
            "-Wl,-z,relro",
            "-Wl,-z,now",
            "-Wl,-soname,libno_workspace_kernel.so",
            "-o",
            str(library),
            str(destination / "launch.o"),
            *(str(destination / f"head_{h}.o") for h in heads),
        ],
        check=True,
    )
    return {
        "source_sha256": hashes,
        "heads": heads,
        "library_sha256": sha256(library),
        "build_seconds": time.perf_counter() - started,
    }


class AccuracyFailure(AssertionError):
    def __init__(self, metrics):
        self.metrics = metrics
        super().__init__("Pointwise/RMSE acceptance failed; see numerical_errors")


def check_outputs(
    launch, expected, expected_state, previous=None, *, allow_failed=False
):
    """Check raw output against FP64 raw goldens; reuse only unchanged results."""
    import torch

    # Keep independent snapshots: even CPU test inputs may be mutated in place.
    raw = launch["out"].cpu().clone()
    state = launch["final_state"].cpu().view_as(expected_state).clone()
    # Mutation counters prevent reusing metrics after in-place oracle changes.
    # PyTorch exposes this counter only through Tensor._version.
    versions = (
        expected._version,  # pylint: disable=protected-access
        expected_state._version,  # pylint: disable=protected-access
    )
    if (
        previous is not None
        and previous[3] is expected
        and previous[4] is expected_state
        and previous[5:] == versions
        and all(
            a.shape == b.shape
            and a.dtype == b.dtype
            and torch.equal(a.view(torch.int16), b.view(torch.int16))
            for a, b in zip((raw, state), previous[:2])
        )
    ):
        # Every output/state bit was checked; unchanged tensors have the same
        # elementwise bounds and maximum error against this fixed oracle.
        if not allow_failed and not all(m["passed"] for m in previous[2].values()):
            raise AccuracyFailure(previous[2])
        return previous
    metrics = {
        "raw_output": error_metrics(raw, expected),
        "final_state": error_metrics(state, expected_state),
    }
    if not allow_failed and not all(value["passed"] for value in metrics.values()):
        raise AccuracyFailure(metrics)
    return (
        raw,
        state,
        metrics,
        expected,
        expected_state,
        *versions,
    )


def time_launch(launch, *, trials=15, inner=20):
    """Median device time for warmed, allocation-free graph replays."""
    import torch
    from pto_kernels import pto_gdn_mega

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(inner):
            pto_gdn_mega(**launch)
    graph.replay()
    torch.npu.synchronize()
    times = []
    for _ in range(trials):
        start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / inner)
    return {
        "median_us": statistics.median(times),
        "min_us": min(times),
        "max_us": max(times),
        "trials": trials,
        "calls_per_trial": inner,
        "trial_us": times,
    }


def benchmark_case(args, length):
    """Check and time one shape; release its graph/tensors on return."""
    import torch
    from pto_kernels import pto_gdn_mega
    from test_gdn_mega import (
        _make_inputs,
        _prepare_gdn_mega,
        cached_recurrent_reference,
        tensor_fingerprint,
        POLICY,
    )

    reference_start = time.perf_counter()
    inputs = _make_inputs(length, args.heads, args.key_heads)
    inputs[3].mul_(args.gate_scale)
    initial = (
        torch.randn(1, args.heads, 128, 128, dtype=torch.float16) * 0.1
        if args.initial_state
        else None
    )
    expected, expected_state = cached_recurrent_reference(*inputs, initial)
    cu = torch.tensor([0, length], dtype=torch.int32)
    reference_seconds = time.perf_counter() - reference_start
    check_seconds = 0.0
    launch = _prepare_gdn_mega(
        *(t.to(args.device) for t in inputs),
        cu.to(args.device),
        initial.to(args.device) if initial is not None else None,
    )
    checked = None
    warmup_errors = []
    for _ in range(1 if args.profile_once else 5):
        pto_gdn_mega(**launch)
        torch.npu.synchronize()
        check_start = time.perf_counter()
        checked = check_outputs(
            launch, expected, expected_state, checked, allow_failed=args.time_failed
        )
        warmup_errors.append(checked[2])
        check_seconds += time.perf_counter() - check_start
    metrics = checked[2]
    if args.profile_once:
        print(
            json.dumps(
                {
                    "length": length,
                    "numerical_errors": metrics,
                    "validation_policy": POLICY,
                }
            ),
            flush=True,
        )
        return None
    # Capture launches so Python dispatch cannot leave gaps between kernels.
    timing = time_launch(launch, trials=args.trials, inner=args.inner)
    record = {
        "length": length,
        "heads": args.heads,
        "key_heads": args.key_heads,
        "initial_state": args.initial_state,
        "gate_scale": args.gate_scale,
        **timing,
        "numerical_errors": metrics,
    }
    record["input_sha256"] = tensor_fingerprint([*inputs, initial, cu])
    record["reference_sha256"] = tensor_fingerprint([expected, expected_state])
    record["validation_policy"] = POLICY
    # Verify the timed graph also leaves correct outputs, outside its timing.
    check_start = time.perf_counter()
    record["after_replay_errors"] = check_outputs(
        launch, expected, expected_state, checked, allow_failed=args.time_failed
    )[2]
    passed = all(
        m["passed"]
        for phase in [*warmup_errors, record["after_replay_errors"]]
        for m in phase.values()
    )
    record["status"] = "passed" if passed else "failed"
    record["warmup_errors"] = warmup_errors
    if not passed:
        record["timing_warning"] = (
            "Runtime measured despite numerical failure; not an accuracy-equivalent result"
        )
    check_seconds += time.perf_counter() - check_start
    record["validation_seconds"] = {
        "inputs_and_fp64_reference": reference_seconds,
        "all_output_checks": check_seconds,
    }
    record["validation_reuse"] = (
        "Statistics reused only after byte equality of every output/state element"
    )
    return record


def summarize(repeats, name):
    metrics = [r[name] for r in repeats]
    finite = all(m["finite"] for m in metrics)
    return {
        "finite": finite,
        "passed": all(m["passed"] for m in metrics),
        "elements": metrics[0]["elements"],
        "max_abs": max(m["max_abs"] for m in metrics) if finite else None,
        "max_tolerance_ratio": (
            max(m["max_tolerance_ratio"] for m in metrics) if finite else None
        ),
        "pointwise_violations_max": (
            max(m["pointwise_violations"] for m in metrics) if finite else None
        ),
        "relative_rmse_max": (
            max(m["relative_rmse"] for m in metrics) if finite else None
        ),
        "actual_nonfinite_max": max(m.get("actual_nonfinite", 0) for m in metrics),
    }


def worker(spec_path):
    import torch
    import torch_npu
    from test_gdn_mega import recurrent_reference

    spec = json.loads(spec_path.read_text())
    args = argparse.Namespace(**spec["settings"])
    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = False
    libraries = {
        line.split()[-1]
        for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
        if line.split()[-1].endswith("/libno_workspace_kernel.so")
    }
    if len(libraries) != 1 or sha256(libraries.pop()) != spec["library_sha256"]:
        raise RuntimeError("Loaded library differs from requested library")
    props = torch.npu.get_device_properties(args.device)
    environment = {
        "device_name": props.name,
        "cube_cores": props.cube_core_num,
        "vector_cores": props.vector_core_num,
        "torch_version": str(torch.__version__),
        "torch_npu_version": str(torch_npu.__version__),
    }
    records = []
    for args.heads, args.key_heads in spec["head_configs"]:
        for length in spec["lengths"]:
            record = benchmark_case(args, length)
            if record is None:  # One checked launch for msprof, no timing graph.
                continue
            record.update(
                kernel_library_sha256=spec["library_sha256"],
                environment=environment,
                oracle_sha256=hashlib.sha256(
                    inspect.getsource(recurrent_reference).encode()
                ).hexdigest(),
            )
            records.append(record)
            Path(spec["output"]).write_text(
                json.dumps(records, indent=2) + "\n", encoding="utf-8"
            )
            print(
                f"H{args.heads}/Hg{args.key_heads} T{length}: {record['median_us']:.2f} us; {record['status']}",
                flush=True,
            )
    return int(any(r["status"] != "passed" for r in records))


def run_measurements(args):
    """Load each library in a separate process and alternate order by head pair."""
    pairs = parse_head_configs(args.head_configs)
    lengths = args.lengths or (LENGTHS if args.command == "heatmap" else [256, 4096])
    if not lengths or min(lengths) <= 0 or len(set(lengths)) != len(lengths):
        raise ValueError("Lengths must be distinct positive integers")
    if args.trials < 1 or args.inner < 1:
        raise ValueError("Trials and calls per trial must be positive")
    if args.command == "bench" and not args.head_configs:
        pairs = [(16, 4)]
    libraries = {"current": args.library.resolve()}
    if args.command == "heatmap":
        libraries["baseline"] = args.baseline_lib.resolve()
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "head_configs": pairs,
        "lengths": lengths,
        "library_sha256": {
            v: sha256(p / "libno_workspace_kernel.so") for v, p in libraries.items()
        },
        "working_tree_source_sha256": {
            p.name: sha256(p)
            for p in (ROOT / "csrc/kernel/gdn_mega").iterdir()
            if p.suffix in (".cpp", ".hpp")
        },
        "working_tree_commit": subprocess.check_output(
            ["git", "-c", f"safe.directory={ROOT}", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip(),
        "seed": 0,
        "initial_state": args.initial_state,
        "gate_scale": args.gate_scale,
        "records": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gdn_mega_") as temporary:
        directory = Path(temporary)
        for index, pair in enumerate(pairs):
            results = {}
            for variant in list(libraries) if index % 2 else list(libraries)[::-1]:
                path = directory / f"{variant}.json"
                spec = {
                    "head_configs": [pair],
                    "lengths": lengths,
                    "output": str(path),
                    "library_sha256": report["library_sha256"][variant],
                    "settings": {
                        "device": args.device,
                        "initial_state": args.initial_state,
                        "gate_scale": args.gate_scale,
                        "trials": args.trials,
                        "inner": args.inner,
                        "profile_once": False,
                        "time_failed": True,
                    },
                }
                manifest = directory / "worker.json"
                manifest.write_text(json.dumps(spec))
                result = subprocess.run(
                    [sys.executable, __file__, "_worker", str(manifest)],
                    cwd=ROOT,
                    env=library_env(libraries[variant]),
                    check=False,
                )
                if result.returncode not in (0, 1) or not path.exists():
                    raise RuntimeError(
                        f"{variant} worker failed with exit code {result.returncode}"
                    )
                rows = json.loads(path.read_text())
                if len(rows) != len(lengths) or any(
                    r["kernel_library_sha256"] != spec["library_sha256"] for r in rows
                ):
                    raise RuntimeError("Incomplete results or mismatched library")
                results[variant] = {r["length"]: r for r in rows}
            for length in lengths:
                current = results["current"][length]
                row = {
                    "heads": pair[0],
                    "key_heads": pair[1],
                    "length": length,
                    "current": current,
                    "current_ms": current["median_us"] / 1000,
                    "status": current["status"],
                }
                if "baseline" in results:
                    baseline = results["baseline"][length]
                    for key in (
                        "input_sha256",
                        "reference_sha256",
                        "validation_policy",
                        "environment",
                        "oracle_sha256",
                    ):
                        if current[key] != baseline[key]:
                            raise RuntimeError(f"Mismatched baseline/current {key}")
                    row.update(
                        baseline=baseline,
                        baseline_ms=baseline["median_us"] / 1000,
                        speedup=baseline["median_us"] / current["median_us"],
                    )
                report["records"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.command == "heatmap":
        render(report, args.report_dir)
    return int(any(r["current"]["status"] != "passed" for r in report["records"]))


def validate_report(data):
    pairs = [tuple(p) for p in data["head_configs"]]
    lengths = data["lengths"]
    cells = {(r["heads"], r["key_heads"], r["length"]): r for r in data["records"]}
    if len(cells) != len(data["records"]) or set(cells) != {
        (h, k, n) for h, k in pairs for n in lengths
    }:
        raise ValueError("Heatmaps require one complete matched grid")
    for row in cells.values():
        for variant in ("baseline", "current"):
            record = row[variant]
            if record["kernel_library_sha256"] != data["library_sha256"][variant]:
                raise ValueError("Mismatched library fingerprint")
            if record["status"] not in ("passed", "failed") or record["median_us"] <= 0:
                raise ValueError("Missing timing or accuracy status")
            phases = record.get("warmup_errors", [record["numerical_errors"]]) + [
                record["after_replay_errors"]
            ]
            passed = all(m["passed"] for phase in phases for m in phase.values())
            if passed != (record["status"] == "passed"):
                raise ValueError("Accuracy status hides a failed checked launch")
        for field in (
            "validation_policy",
            "environment",
            "input_sha256",
            "reference_sha256",
            "oracle_sha256",
        ):
            # Older saved grids predate explicit input/reference hashes. Their
            # recorded source identities remain visible; new workers require all.
            if row["baseline"].get(field) != row["current"].get(field):
                raise ValueError(f"Mismatched baseline/current {field}")
    return pairs, lengths, cells


def render(data, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, TwoSlopeNorm
    import numpy as np

    pairs, lengths, cells = validate_report(data)
    benchmark_link = os.path.relpath(ROOT / "benchmarking/gdn_mega.py", directory)
    ranges_link = os.path.relpath(ROOT / "docs/gdn_mega/input_ranges.md", directory)
    figures = directory / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    values = np.array([[cells[h, k, n]["speedup"] for n in lengths] for h, k in pairs])
    fig, axis = plt.subplots(
        figsize=(max(9, 2 * len(lengths)), max(4, len(pairs) * 1.1)),
        layout="constrained",
    )
    mesh = axis.imshow(
        values,
        cmap="RdYlGn",
        norm=TwoSlopeNorm(
            vmin=min(0.5, float(values.min())),
            vcenter=1,
            vmax=max(1.01, float(values.max())),
        ),
        aspect="auto",
    )
    for i, (h, k) in enumerate(pairs):
        for j, n in enumerate(lengths):
            row = cells[h, k, n]

            current_mark = "*" if numerical_failure(row, "current") else ""
            baseline_mark = "*" if numerical_failure(row, "baseline") else ""
            label = (
                f"{row['speedup']:.2f}×\n"
                f"ours: {row['current_ms']*1000:,.1f} µs{current_mark}\n"
                f"base: {row['baseline_ms']*1000:,.1f} µs{baseline_mark}"
            )
            axis.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=9,
                color="white" if mesh.norm(values[i, j]) > 0.8 else "#17202a",
            )

    def labels(axis):
        axis.set_xticks(range(len(lengths)), [str(n) for n in lengths])
        axis.set_yticks(range(len(pairs)), [f"{h}/{k}" for h, k in pairs])
        axis.set_xlabel("Tokens")
        axis.set_ylabel("Value / key heads")

    labels(axis)
    axis.set_title(
        f"GDN mega: baseline / current runtime — {data['created_utc'][:10]}\n"
        "* Only the runtime of an implementation failing numerical checks is marked"
    )
    fig.colorbar(mesh, ax=axis, label="Runtime ratio (higher is faster)")
    fig.savefig(figures / "heatmap.png", dpi=160)
    plt.close(fig)
    metrics = {}
    for key, row in cells.items():
        for variant in ("baseline", "current"):
            r = row[variant]
            phases = r.get("warmup_errors", [r["numerical_errors"]]) + [
                r["after_replay_errors"]
            ]
            metrics[key, variant] = summarize(phases, "raw_output")
    errors = np.array(
        [
            [
                [
                    (
                        metrics[(h, k, n), v]["max_abs"]
                        if metrics[(h, k, n), v]["finite"]
                        else np.nan
                    )
                    for n in lengths
                ]
                for h, k in pairs
            ]
            for v in ("baseline", "current")
        ]
    )
    positive = errors[np.isfinite(errors) & (errors > 0)]
    lo = float(positive.min()) if positive.size else 1e-8
    hi = max(float(positive.max()), lo * 1.01) if positive.size else 1.0
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(max(12, 3 * len(lengths)), max(4, len(pairs) * 0.7)),
        layout="constrained",
    )
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#d9dde3")
    for i, (axis, v) in enumerate(zip(axes, ("baseline", "current"))):
        mesh = axis.imshow(
            np.maximum(errors[i], lo), norm=LogNorm(lo, hi), cmap=cmap, aspect="auto"
        )
        labels(axis)
        axis.set_title(f"{v}: {data['library_sha256'][v][:10]}")
        for r in range(len(pairs)):
            for c in range(len(lengths)):
                value = errors[i, r, c]
                label = f"{value:.2e}" if np.isfinite(value) else "NaN/Inf"
                axis.text(
                    c,
                    r,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=(
                        "white"
                        if not np.isfinite(value) or mesh.norm(max(value, lo)) < 0.7
                        else "#17202a"
                    ),
                )
    fig.suptitle(
        f"Maximum absolute raw-output error against CPU FP64 — {data['created_utc'][:10]} (all checked launches)"
    )
    fig.colorbar(mesh, ax=axes, label="max |out − reference|; shared log scale")
    fig.savefig(figures / "error_heatmap_max_abs.png", dpi=160)
    plt.close(fig)
    policy = data["records"][0]["current"]["validation_policy"]
    scope = (
        f"Measured {data['created_utc'][:10]}; "
        f"baseline library `{data['library_sha256']['baseline'][:12]}`, "
        f"current `{data['library_sha256']['current'][:12]}`.\n"
    )
    scope += f"Nonzero initial state: {data['initial_state']}; seed 0; gate scale {data['gate_scale']}; D=C=128.\n"
    scope += "Independent CPU FP64 token recurrence on actual rounded inputs. Every output/state element must satisfy\n"
    scope += (
        f"`abs(error) <= {policy['atol']:g} + {policy['rtol']:g}*abs(reference)`, "
        f"plus finiteness and relative RMSE < {policy['max_relative_rmse']:g}.\n"
    )
    current_passes = sum(r["current"]["status"] == "passed" for r in cells.values())
    baseline_passes = sum(r["baseline"]["status"] == "passed" for r in cells.values())
    scope += (
        f"Current passes {current_passes}/{len(cells)} shapes; "
        f"baseline passes {baseline_passes}/{len(cells)}.\n"
    )
    table = (
        "| H/Hg | "
        + " | ".join(map(str, lengths))
        + " |\n|---|"
        + "---:|" * len(lengths)
        + "\n"
    )
    for h, k in pairs:
        table += (
            f"| {h}/{k} | "
            + " | ".join(
                f"{cells[h,k,n]['current_ms']*1000:.1f}"
                + ("*" if numerical_failure(cells[h, k, n], "current") else "")
                for n in lengths
            )
            + " |\n"
        )
    trials = data["records"][0]["current"]["trials"]
    calls = data["records"][0]["current"]["calls_per_trial"]
    latency = f"""# GDN speed heatmap

{scope}
![Latency](figures/heatmap.png)

Each cell shows runtime ratio and total median µs for both kernels. Asterisks
mark only failed implementations; those ratios are not accuracy-equivalent
speedups.

## Benchmark procedure (identical for both libraries)

1. For each shape, generate identical seed-0 inputs and the CPU FP64 reference.
   Copy inputs to the NPU and allocate outputs, masks, intermediates and all
   workspace **once**, using the same caller allocation helper for both libraries.
2. Run five warmup calls, checking raw output and final state after each call.
3. Capture an NPU graph containing **{calls} consecutive full kernel calls**.
   Replay that graph once without timing, then synchronize.
4. Measure **{trials} graph replays** with NPU start/end events. Divide each graph
   duration by {calls}; report the median of these {trials} per-call averages.
   This is {trials*calls} timed calls, not individually timed Python-loop calls.
5. Check output and final state after the final timed replay, outside timing.

All calls reuse the same device addresses and input values. Each starts from
that shape's original supplied state (or zero when initial state is disabled);
final state is a separate output and is not fed into the next call. Scratch and
outputs are reused without external clearing between calls; initialization
required by the caller contract occurs before warmup.

The timed region covers the complete device kernel and graph execution overhead,
including the kernel's internal transfers and synchronization. It excludes
allocation/initialization, host-device input copies, CPU reference computation,
validation, graph capture and caller-side output scaling. Graph replay amortizes
submission overhead and avoids per-kernel Python dispatch gaps. No cache flush
is performed: these are warmed, repeated-buffer timings, not end-to-end request
latencies or guaranteed cold-cache timings.

Current and baseline run serially in separate processes. Their order alternates
between head configurations. Loaded-library hashes are verified, and matched
workers must agree on input/reference hashes, environment and accuracy policy.
Both libraries are freshly timed, including the original when it fails checks.

## Current median total microseconds

{table}
## Scheduling tradeoff

The removed H8-specific schedule (`365b105`) used sixteen state-column workers
and eight preparation helpers. Matched H8/Hg2 measurements at 4K–32K showed
**22–27% lower latency (1.29–1.38× throughput)** than the uniform schedule.
It was removed for simplicity, saving 153 source lines; shared factored and
asynchronous joins remain. This is a historical comparison, not a guarantee
for arbitrary candidate libraries supplied to this script.

Use the [benchmark tool]({benchmark_link}) (`python benchmarking/gdn_mega.py --help`)
for reproduction. Raw run files are intentionally excluded from the documentation tree.
"""
    (directory / "heatmap.md").write_text(latency)
    text = f"""# GDN absolute-error heatmap

{scope}
![Maximum absolute output error](figures/error_heatmap_max_abs.png)

Maxima cover every raw-output element across checked warmups and after timing.
Final state is checked too, but is not plotted. This is the same ordinary-input
grid as the speed heatmap, not the adversarial regression suite.

## Accuracy regimes

The regular suite includes 178 fixed cases: ordinary Gaussian and FLA-style
positive inputs; beta endpoints and interior values; zero, tiny and strong
gates; correlated/nearly identical keys; large values and initial states;
packed/empty sequences and contexts through 32K. Each case uses five poisoned
scratch launches and checks inputs and output guards. The production uniform
kernel at `f381ae8` passed these cases under the fixed policy above. An
experimental candidate's ordinary-input grid does not certify its full suite.

These are measured regimes, not a guarantee for arbitrary inputs. Primary use
assumes normalized Q/K, beta in [0,1] and nonpositive log gates. A few exploratory
tests go beyond those assumptions. Strong correlation, large state/value
magnitudes and nearly lossless long recurrences are the most sensitive cases.

Under the older atol=1e-5, 23 of 160 historical stress cases failed. The largest
required absolute floor at rtol=1e-3 was 8.293e-5; the user approved atol=1e-4
and moved those cases into regular tests. This was a tolerance-policy change,
not a numerical fix. Failures were often small absolute errors near zero;
maximum error alone can also be dominated by unavoidable FP16 output rounding.
No clipping or case-specific thresholds are used.

Use the [benchmark tool]({benchmark_link}) (`python benchmarking/gdn_mega.py test`)
for validation. See [input ranges]({ranges_link}) for the captured workload.
"""
    (directory / "error_heatmap.md").write_text(text)
    print(f"Saved two PNGs and heatmap Markdown to {directory}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(
        dest="command", required=True, metavar="{build,test,bench,heatmap,render}"
    )
    p = commands.add_parser(
        "build", help="Fast isolated compilation; does not install or run the kernel"
    )
    p.add_argument("--source", type=Path, default=ROOT / "csrc/kernel/gdn_mega")
    p.add_argument("--output", type=Path, default=ROOT / "build/gdn_candidate")
    p.add_argument("--quick", action="store_true", help="Compile H8/H16 only")
    p = commands.add_parser(
        "test", help="Run the full suite, or nine selected numerical cases"
    )
    p.add_argument("--library", type=Path, default=ROOT / "build/lib")
    p.add_argument("--quick", action="store_true")
    for name in ("bench", "heatmap"):
        p = commands.add_parser(name)
        p.add_argument("--library", type=Path, default=ROOT / "build/lib")
        p.add_argument("--head-configs", nargs="+", metavar="H/Hg")
        p.add_argument("--lengths", type=int, nargs="+")
        p.add_argument(
            "--initial-state", action=argparse.BooleanOptionalAction, default=True
        )
        p.add_argument("--gate-scale", type=float, default=1.0)
        p.add_argument("--trials", type=int, default=15)
        p.add_argument("--inner", type=int, default=20)
        p.add_argument("--device", default=os.environ.get("NPU_DEVICE", "npu:0"))
        p.add_argument(
            "--output", type=Path, default=ROOT / f"benchmark_data/gdn_mega/{name}.json"
        )
        if name == "heatmap":
            p.add_argument("--baseline-lib", type=Path, required=True)
            p.add_argument("--report-dir", type=Path, default=ROOT / "docs/gdn_mega")
    p = commands.add_parser(
        "render", help="Regenerate the two reports from saved heatmap JSON"
    )
    p.add_argument("input", type=Path)
    p.add_argument("--report-dir", type=Path, default=ROOT / "docs/gdn_mega")
    p = commands.add_parser("_worker")
    p.add_argument("manifest", type=Path)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.command == "build":
        source, output = args.source.resolve(), args.output.resolve()
        if output == source or source in output.parents:
            raise ValueError("Build outside the source directory")
        report = build(source, output, (8, 16) if args.quick else HEADS)
        (output / "build.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"Built {len(report['heads'])} specializations in {report['build_seconds']:.2f}s: {output}"
        )
    elif args.command == "test":
        selected = (
            [
                f"tests/test_gdn_mega.py::test_independent_recurrence_and_memory[{case}]"
                for case in QUICK_CASES
            ]
            if args.quick
            else ["tests/test_gdn_mega.py"]
        )
        return subprocess.call(
            [sys.executable, "-m", "pytest", *selected, "-q"],
            cwd=ROOT,
            env=library_env(args.library),
        )
    elif args.command == "_worker":
        return worker(args.manifest)
    elif args.command == "render":
        render(json.loads(args.input.read_text()), args.report_dir)
    else:
        return run_measurements(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
