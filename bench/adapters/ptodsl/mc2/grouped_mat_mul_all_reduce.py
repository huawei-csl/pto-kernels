from __future__ import annotations

import json
import os
from pathlib import Path

from pto_kernels.bench.adapter_utils import compile_pto_kernel, describe_pto, load_module, temporary_env
from pto_kernels.ops.mc2.grouped_mat_mul_all_reduce.runtime import (
    VARIANTS,
    run_distributed_pto_benchmark,
)


KERNEL = "python/pto_kernels/ops/mc2/grouped_mat_mul_all_reduce/kernel.py"
META = "python/pto_kernels/ops/mc2/grouped_mat_mul_all_reduce/meta.py"


def describe(repo_root, spec):
    return describe_pto(repo_root, KERNEL, META)


def compile_kernel(repo_root, spec, artifacts_dir):
    return compile_pto_kernel(repo_root, KERNEL, artifacts_dir)


def _variant_env(variant) -> dict[str, str]:
    return {
        "PTO_MC2_GMM_AR_WORLD_SIZE": str(variant.expected_world_size),
        "PTO_MC2_GMM_AR_M": str(variant.m),
        "PTO_MC2_GMM_AR_K_LOCAL": str(variant.k_local),
        "PTO_MC2_GMM_AR_N": str(variant.n),
        "PTO_MC2_GMM_AR_BASE_M": os.environ.get("PTO_MC2_GMM_AR_BASE_M", "32"),
        "PTO_MC2_GMM_AR_BASE_N": os.environ.get("PTO_MC2_GMM_AR_BASE_N", "32"),
        "PTO_MC2_GMM_AR_BASE_K": os.environ.get("PTO_MC2_GMM_AR_BASE_K", "64"),
        "PTO_MC2_GMM_AR_BLOCK_DIM": os.environ.get("PTO_MC2_GMM_AR_BLOCK_DIM", "4"),
    }


def benchmark(repo_root, spec, artifacts_dir):
    try:
        variant_reports = []
        artifact_paths: list[str] = []
        kernel_file = repo_root / KERNEL
        for variant in VARIANTS:
            with temporary_env(_variant_env(variant)):
                module = load_module(Path(kernel_file))
                builder = getattr(module, "build_jit_wrapper", None)
                if not callable(builder):
                    return {
                        "status": "blocked",
                        "reason": "kernel module does not expose build_jit_wrapper(output_dir)",
                    }

                wrapper = builder(output_dir=Path(artifacts_dir) / variant.label / "compile_probe")
                build = getattr(wrapper, "_build", None)
                if callable(build):
                    build()
                artifact_paths.extend([str(path) for path in getattr(wrapper, "_artifact_paths", lambda: ())()])

                variant_report = run_distributed_pto_benchmark(
                    variant=variant,
                    artifacts_dir=Path(artifacts_dir) / variant.label,
                    warmup=spec.bench.warmup,
                    repeat=spec.bench.repeat,
                )
                if variant_report.get("status") == "ok":
                    max_abs_diff = float(variant_report["correctness"]["max_abs_diff"])
                    variant_report["correctness"].update(
                        {
                            "atol": spec.correctness.atol,
                            "rtol": spec.correctness.rtol,
                            "passes": bool(max_abs_diff <= spec.correctness.atol),
                        }
                    )
                variant_reports.append(variant_report)
    except Exception as exc:  # pragma: no cover - exercised on NPU hosts
        report = {
            "status": "blocked",
            "variants": [variant.as_dict() for variant in VARIANTS],
            "reason": f"PTO compile failed: {exc}",
        }
        report_path = Path(artifacts_dir) / "ptodsl_grouped_mat_mul_all_reduce_benchmark.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        report["report_path"] = str(report_path)
        return report

    if any(item.get("status") != "ok" for item in variant_reports):
        first_blocked = next(item for item in variant_reports if item.get("status") != "ok")
        report = {
            "status": "blocked",
            "variants": [variant.as_dict() for variant in VARIANTS],
            "reason": first_blocked.get("reason", "Distributed PTO grouped_mat_mul_all_reduce launch failed."),
            "variant_reports": variant_reports,
            "artifact_paths": artifact_paths,
        }
    else:
        max_abs_diff = max(float(item["correctness"]["max_abs_diff"]) for item in variant_reports)
        report = {
            "status": "ok",
            "variants": [item["variant"] for item in variant_reports],
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
            "artifact_paths": artifact_paths,
            "reference_contract": "pto_local_grouped_matmul_then_all_reduce",
        }

    report_path = Path(artifacts_dir) / "ptodsl_grouped_mat_mul_all_reduce_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
