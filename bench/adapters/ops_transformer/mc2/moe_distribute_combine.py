from __future__ import annotations

import json
from pathlib import Path

from pto_kernels.bench.adapter_utils import describe_baseline
from pto_kernels.ops.mc2.moe_distribute_combine.runtime import (
    VARIANT,
    VARIANTS,
    baseline_blocker,
    run_distributed_baseline_benchmark,
)


def describe(repo_root, spec):
    summary = describe_baseline(repo_root, "mc2", "moe_distribute_combine", spec.inventory_ref)
    summary["runtime_entrypoint"] = "torch_npu.npu_moe_distribute_combine"
    summary["seed_variant"] = {"default": VARIANT.as_dict(), "variants": [variant.as_dict() for variant in VARIANTS]}
    return summary


def compile_kernel(repo_root, spec, artifacts_dir):
    del repo_root, spec, artifacts_dir
    report = baseline_blocker(device_index=0)
    environment = report.get("environment", {})
    if report["status"] == "ready" and environment.get("symbol_available"):
        return {
            "status": "runtime_builtin_distributed",
            "entrypoint": "torch_npu.npu_moe_distribute_combine",
            "note": "Baseline execution needs a real 8-rank HCCL launch; the current local harness still fast-fails by default.",
            "environment": environment,
        }
    return report


def benchmark(repo_root, spec, artifacts_dir):
    del repo_root
    variant_reports = []
    for variant in VARIANTS:
        variant_reports.append(
            run_distributed_baseline_benchmark(
                variant=variant,
                artifacts_dir=Path(artifacts_dir) / variant.label,
                warmup=spec.bench.warmup,
                repeat=spec.bench.repeat,
            )
        )
    first_blocked = next((item for item in variant_reports if item.get("status") != "ok"), None)
    if first_blocked is not None:
        report = {
            "status": "blocked",
            "variants": [variant.as_dict() for variant in VARIANTS],
            "entrypoint": "torch_npu.npu_moe_distribute_combine",
            "reason": first_blocked.get("reason", "Distributed MC2 baseline launch failed."),
            "variant_reports": variant_reports,
        }
    else:
        report = {
            "status": "ok",
            "variants": [item["variant"] for item in variant_reports],
        }
    report_path = Path(artifacts_dir) / "ops_transformer_moe_distribute_combine_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
