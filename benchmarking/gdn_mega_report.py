"""Generate cached GDN timing, saved-FLA comparison and numerical heatmaps."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))


def provenance():
    git = ["git", "-c", f"safe.directory={ROOT}"]  # Container sees host-owned files.
    sources = sorted(
        {
            *ROOT.glob("csrc/kernel/gdn_mega/*.*"),
            *ROOT.glob("csrc/kernel/gdn_mega_bwd/*.*"),
            *ROOT.glob("tests/*gdn*py"),
            Path(__file__),
        }
    )
    return dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=subprocess.check_output(
            [*git, "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        git_status=subprocess.check_output(
            [*git, "status", "--porcelain"], cwd=ROOT, text=True
        ),
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
            if p.is_file()
        },
        forward_library_sha256=hashlib.sha256(
            (ROOT / "build/lib/libno_workspace_kernel.so").read_bytes()
        ).hexdigest(),
    )


def main():
    fla_name = "benchmarks/fla-gdn-9e82a80/sweep/gdn_ascend_benchmark.json"
    fla_paths = [Path("/mounted_home") / fla_name, Path("/home/fskogh") / fla_name]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmark_data/gdn_mega_bwd/report")
    )
    parser.add_argument(
        "--fla",
        type=Path,
        default=next((p for p in fla_paths if p.exists()), fla_paths[-1]),
    )
    parser.add_argument(
        "--cache-dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--flashqla-bound",
        action="store_true",
        help="Also check backward L2 relative error <= 2% against an FP64 recurrence",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Small smoke report: 8 TP shapes, 2 FLA shapes, 2 numerical cases",
    )
    parser.add_argument(
        "--only",
        choices=("timings", "fla", "errors"),
        help="Generate just one report section",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate PNG from saved JSON; no Torch or NPU work",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    sections = (args.only,) if args.only else ("timings", "fla", "errors")
    if "fla" in sections and not args.fla.is_file():
        parser.error(
            "Saved FLA JSON missing; provide --fla PATH or use --only timings/errors"
        )
    args.output = args.output.resolve()
    for section in sections:
        destination = args.output / section
        destination.mkdir(parents=True, exist_ok=True)
        opts = SimpleNamespace(
            **{**vars(args), "output": destination, "wall_only": True}
        )
        metadata = None
        if not args.plot_only:
            metadata = provenance()
            metadata["options"] = {
                k: str(v) if isinstance(v, Path) else v for k, v in vars(opts).items()
            }
            metadata["status"] = "incomplete"
            (destination / "metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n"
            )
        if section == "timings":
            import benchmark_gdn_mega_bwd_sweep as sweep

            if args.plot_only:
                sweep.plot(
                    [
                        json.loads(line)
                        for line in (destination / "results.jsonl")
                        .read_text()
                        .splitlines()
                    ],
                    destination,
                )
            else:
                if args.quick:
                    opts.global_heads, opts.tps, opts.lengths = (
                        (64, 16),
                        (1, 2),
                        (256, 1024),
                    )
                sweep.run(opts)
        elif section == "fla":
            import benchmark_gdn_mega_bwd_fla as fla

            command = [
                "--fla",
                str(args.fla),
                "--output",
                str(destination),
                "--repeats",
                str(args.repeats),
                "--cache-dtype",
                args.cache_dtype,
            ]
            # Replot the original quick/full selection, independent of current defaults.
            quick = args.quick
            if args.plot_only:
                quick = json.loads((destination / "metadata.json").read_text())[
                    "options"
                ]["quick"]
            fla.main(
                command
                + (["--quick"] if quick else [])
                + (["--plot-only"] if args.plot_only else [])
            )
        else:
            import benchmark_gdn_mega_bwd_errors as errors

            errors.run(opts)
        if metadata is not None:
            metadata["status"] = "complete"
            (destination / "metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n"
            )
        print(f"{section}: {destination}", flush=True)


if __name__ == "__main__":
    main()
