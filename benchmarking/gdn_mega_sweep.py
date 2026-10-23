"""Compare two compiled GDN libraries over sequence length and head count.

Runs gdn_mega.py in isolated processes so each implementation is actually loaded.
Every cell checks output and final state; allocation/reference time is excluded.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_HEADS = [8, 12, 16, 24, 32, 48, 64]


def save_report(report, output):
    for row in report["records"]:
        row.setdefault("status", "passed")
    output.write_text(json.dumps(report, indent=2) + "\n")
    with output.with_suffix(".csv").open("w", newline="") as f:
        fields = ["heads", "key_heads", "length", "status", "baseline_ms", "current_ms", "speedup"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(report["records"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-lib", type=Path, required=True, help="Directory containing original libno_workspace_kernel.so")
    parser.add_argument("--current-lib", type=Path, default=ROOT / "build/lib")
    parser.add_argument("--heads", type=int, nargs="+", default=SUPPORTED_HEADS)
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 256, 1024, 4096, 8192, 16384, 32768])
    parser.add_argument("--group-size", type=int, default=4, help="Value heads per key head; Hg = H / group_size")
    parser.add_argument("--gate-scale", type=float, default=1.0)
    parser.add_argument("--initial-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default=os.environ.get("NPU_DEVICE", "npu:0"))
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark_data/gdn_mega/sweep.json")
    parser.add_argument("--resume", action="store_true", help="Reuse completed head rows with identical settings and library hashes")
    args = parser.parse_args()
    if (args.group_size <= 0 or any(h not in SUPPORTED_HEADS or h % args.group_size for h in args.heads)
            or any(n <= 0 for n in args.lengths) or len(set(args.heads)) != len(args.heads)
            or len(set(args.lengths)) != len(args.lengths)):
        parser.error("Use distinct supported heads divisible by group-size and distinct positive lengths")
    libraries = {"baseline": args.baseline_lib.resolve(), "current": args.current_lib.resolve()}
    fingerprints = {name: hashlib.sha256((path / "libno_workspace_kernel.so").read_bytes()).hexdigest()
                    for name, path in libraries.items()}
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(["git", "-c", f"safe.directory={ROOT}", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "heads": args.heads, "lengths": args.lengths, "group_size": args.group_size,
        "gate_scale": args.gate_scale, "initial_state": args.initial_state, "device": args.device,
        "chunk_size": 128, "head_dim": 128, "seed": 0, "warmups": 5, "trials": 15, "calls_per_trial": 20,
        "timing": "Median device-event time per graph-replayed call, warm cache; allocation and checks excluded",
        "library_sha256": fingerprints, "records": [],
    }
    if args.resume and output.exists():
        previous = json.loads(output.read_text())
        for key in ["heads", "lengths", "group_size", "gate_scale", "initial_state", "device",
                    "chunk_size", "head_dim", "seed", "warmups", "trials", "calls_per_trial", "library_sha256"]:
            if previous[key] != report[key]:
                parser.error(f"Cannot resume: {key} differs")
        report = previous
    save_report(report, output)
    with tempfile.TemporaryDirectory(prefix="gdn_mega_sweep_") as temporary:
        runs_dir = Path(temporary)
        for index, heads in enumerate(args.heads):
            if {row["length"] for row in report["records"] if row["heads"] == heads} == set(args.lengths):
                continue
            runs = {}
            # Alternate order across head counts to avoid always measuring one variant first.
            for variant in (["baseline", "current"] if index % 2 == 0 else ["current", "baseline"]):
                path = runs_dir / f"{variant}_h{heads}.json"
                command = [sys.executable, str(ROOT / "benchmarking/gdn_mega.py"),
                           "--heads", str(heads), "--key-heads", str(heads // args.group_size),
                           "--lengths", *map(str, args.lengths), "--gate-scale", str(args.gate_scale),
                           "--device", args.device, "--output", str(path)]
                if args.initial_state:
                    command.append("--initial-state")
                env = dict(os.environ)
                env["LD_LIBRARY_PATH"] = str(libraries[variant]) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
                env["PYTHONPATH"] = str(ROOT / "python") + os.pathsep + env.get("PYTHONPATH", "")
                print(f"\n{variant}: H={heads}, Hg={heads // args.group_size}", flush=True)
                process = subprocess.run(command, cwd=ROOT, env=env, timeout=900)
                if process.returncode not in (0, 1) or not path.exists():
                    raise RuntimeError(f"Worker failed before completing: {variant}, H={heads}")
                runs[variant] = {row["length"]: row for row in json.loads(path.read_text())}
                report["environment"] = next(iter(runs[variant].values()))["environment"]
                if set(runs[variant]) != set(args.lengths):
                    raise RuntimeError(f"Incomplete run: {path}")
                failures = any(row.get("status") == "failed" for row in runs[variant].values())
                if process.returncode != int(failures):
                    raise RuntimeError(f"Unexpected worker failure: {variant}, H={heads}")
                if any(row["kernel_library_sha256"] != fingerprints[variant] for row in runs[variant].values()):
                    raise RuntimeError(f"Dynamic loader selected the wrong library: {variant}")
            for length in args.lengths:
                before, after = runs["baseline"][length], runs["current"][length]
                passed = all(row.get("status", "passed") == "passed" for row in (before, after))
                report["records"].append({
                    "heads": heads, "key_heads": heads // args.group_size, "length": length,
                    "status": "passed" if passed else "failed",
                    "baseline_ms": before["median_us"] / 1000 if "median_us" in before else None,
                    "current_ms": after["median_us"] / 1000 if "median_us" in after else None,
                    "speedup": before["median_us"] / after["median_us"] if passed else None,
                    "baseline": before, "current": after,
                })
            save_report(report, output)
    save_report(report, output)
    print(f"\nSaved {len(report['records'])} comparison results to {output}", flush=True)


if __name__ == "__main__":
    main()
