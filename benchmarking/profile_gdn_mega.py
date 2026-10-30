"""Collect one msprof hardware/simulator profile or render a traceviz timeline.

Run inside CANN; source set_env.sh first. Outputs stay under benchmark_data/.
No instrumented kernel, generated source, or profile.md is required.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from gdn_mega import ROOT, library_env, parse_head_configs, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["hardware", "simulator", "render"])
    parser.add_argument("--library", type=Path, default=ROOT / "build/lib")
    parser.add_argument("--head-config", default="16/4", metavar="H/Hg")
    parser.add_argument(
        "--length", type=int, help="Default: 4096 hardware, 512 simulator"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "benchmark_data/gdn_mega"
        / datetime.now(timezone.utc).strftime("profile_%Y%m%d_%H%M%S"),
    )
    parser.add_argument(
        "--input", type=Path, help="Simulator OPPROF directory for rendering"
    )
    parser.add_argument("--traceviz", type=Path, default=ROOT.parent / "traceviz-mcp")
    parser.add_argument(
        "--units", nargs="+", default=["core0.*"], help="Traceviz core patterns"
    )
    parser.add_argument("--window", nargs=2, type=float, metavar=("START_NS", "END_NS"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "render":
        if args.input is None or not args.input.exists():
            parser.error(
                "render needs --input pointing to a simulator OPPROF directory"
            )
        if not (args.traceviz / "traceviz").is_dir():
            parser.error("--traceviz must point to the traceviz-mcp checkout")
        environment = dict(
            os.environ,
            PYTHONPATH=f"{args.traceviz.resolve()}:{os.environ.get('PYTHONPATH','')}",
        )
        command = [
            sys.executable,
            "-m",
            "traceviz.cli",
            str(args.input.resolve()),
            "--units",
            *args.units,
        ]
        if args.window:
            command += ["--window", *map(str, args.window)]
        with (args.output / "summary.txt").open("w") as output:
            subprocess.run(
                [*command, "--summary"], env=environment, stdout=output, check=True
            )
        subprocess.run(
            [*command, "--out", str((args.output / "timeline.png").resolve())],
            env=environment,
            check=True,
        )
        print(f"Saved {args.output/'summary.txt'} and timeline.png")
        return
    length = (
        args.length
        if args.length is not None
        else (512 if args.mode == "simulator" else 4096)
    )
    if length <= 0:
        parser.error("--length must be positive")
    pair = parse_head_configs([args.head_config])[0]
    environment = library_env(args.library)
    if args.mode == "simulator":
        home = environment.get("ASCEND_HOME_PATH")
        if not home:
            parser.error("Source CANN set_env.sh before using the simulator")
        environment["LD_LIBRARY_PATH"] = (
            f"{home}/tools/simulator/Ascend910B2/lib:{environment['LD_LIBRARY_PATH']}"
        )
    manifest = args.output / "launch.json"
    manifest.write_text(
        json.dumps(
            {
                "head_configs": [pair],
                "lengths": [length],
                "output": str((args.output / "accuracy.json").resolve()),
                "library_sha256": sha256(args.library / "libno_workspace_kernel.so"),
                "settings": {
                    "device": environment["NPU_DEVICE"],
                    "initial_state": True,
                    "gate_scale": 1.0,
                    "trials": 1,
                    "inner": 1,
                    "profile_once": True,
                    "time_failed": False,
                },
            },
            indent=2,
        )
        + "\n"
    )
    command = ["msprof", "op"] + (["simulator"] if args.mode == "simulator" else [])
    command += [
        "--kernel-name=*gdn_mega_specialized*",
        "--aic-metrics="
        + ("PipeUtilization" if args.mode == "simulator" else "Roofline"),
        "--output=" + str(args.output.resolve()),
        sys.executable,
        str(ROOT / "benchmarking/gdn_mega.py"),
        "_worker",
        str(manifest.resolve()),
    ]
    with (args.output / "profile.log").open("w") as output:
        subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )
    checks = [
        json.loads(line)
        for line in (args.output / "profile.log").read_text().splitlines()
        if line.startswith("{") and '"numerical_errors"' in line
    ]
    if not checks or not all(
        m["passed"] for check in checks for m in check["numerical_errors"].values()
    ):
        raise RuntimeError(
            "Profiler application did not finish a passing FP64 output/state check; inspect profile.log"
        )
    (args.output / "accuracy.json").write_text(
        json.dumps(
            {
                "library_sha256": sha256(args.library / "libno_workspace_kernel.so"),
                "checks": checks,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"Profile collected at {args.output}; checked-launch results are in profile.log"
    )


if __name__ == "__main__":
    main()
