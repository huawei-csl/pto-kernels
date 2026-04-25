from __future__ import annotations

import json
from pathlib import Path

from pto_kernels.bench.adapter_utils import describe_baseline
from pto_kernels.ops.mc2.matmul_all_reduce.runtime import (
    VARIANT,
    VARIANTS,
    baseline_blocker,
    run_distributed_baseline_benchmark,
)


def describe(repo_root, spec):
    summary = describe_baseline(repo_root, "mc2", "matmul_all_reduce", spec.inventory_ref)
    summary["runtime_entrypoint"] = "torch_npu.npu_mm_all_reduce_base"
    summary["seed_variant"] = {"default": VARIANT.as_dict(), "variants": [variant.as_dict() for variant in VARIANTS]}
    return summary


def compile_kernel(repo_root, spec, artifacts_dir):
    del repo_root, spec, artifacts_dir
    report = baseline_blocker(device_index=0)
    if report["status"] == "ready":
        return {
            "status": "runtime_builtin_distributed",
            "entrypoint": "torch_npu.npu_mm_all_reduce_base",
            "environment": report["environment"],
        }
    return report


def benchmark(repo_root, spec, artifacts_dir):
    del repo_root
    try:
        variant_reports = []
        for variant in VARIANTS:
            variant_report = run_distributed_baseline_benchmark(
                variant=variant,
                artifacts_dir=Path(artifacts_dir) / variant.label,
                warmup=spec.bench.warmup,
                repeat=spec.bench.repeat,
            )
            variant_reports.append(variant_report)
    except Exception as exc:  # pragma: no cover - exercised on NPU hosts
        report = baseline_blocker(device_index=int(spec.device.get("id", 0)))
        report["reason"] = f"Distributed matmul_all_reduce baseline failed: {exc}"
    else:
        if any(item.get("status") != "ok" for item in variant_reports):
            first_blocked = next(item for item in variant_reports if item.get("status") != "ok")
            report = {
                "status": "blocked",
                "variants": [variant.as_dict() for variant in VARIANTS],
                "entrypoint": "torch_npu.npu_mm_all_reduce_base",
                "reason": first_blocked.get("reason", "Distributed matmul_all_reduce baseline failed."),
                "variant_reports": variant_reports,
            }
        else:
            max_abs_diff = max(float(item["correctness"]["max_abs_diff"]) for item in variant_reports)
            report = {
                "status": "ok",
                "variants": [item["variant"] for item in variant_reports],
                "entrypoint": "torch_npu.npu_mm_all_reduce_base",
                "shape_summaries": [item.get("shape_summary") for item in variant_reports],
                "timings_ms": {
                    "median": max(item["timings_ms"]["median"] for item in variant_reports),
                    "min": min(item["timings_ms"]["min"] for item in variant_reports),
                    "max": max(item["timings_ms"]["max"] for item in variant_reports),
                },
                "correctness": {
                    "max_abs_diff": max_abs_diff,
                    "atol": spec.correctness.atol,
                    "rtol": spec.correctness.rtol,
                    "passes": bool(max_abs_diff <= spec.correctness.atol),
                },
                "variant_reports": variant_reports,
                "reference_contract": "all_reduce(sum_i(x1_local_i @ x2))",
            }
    report_path = Path(artifacts_dir) / "ops_transformer_matmul_all_reduce_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
