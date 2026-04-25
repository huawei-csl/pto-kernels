from __future__ import annotations

import json
import os
from pathlib import Path

from pto_kernels.bench.adapter_utils import describe_pto, load_module, temporary_env
from pto_kernels.ops.mc2.matmul_all_reduce_add_rms_norm.runtime import (
    VARIANTS,
    run_distributed_pto_benchmark,
)


KERNEL = "python/pto_kernels/ops/mc2/matmul_all_reduce_add_rms_norm/kernel.py"
META = "python/pto_kernels/ops/mc2/matmul_all_reduce_add_rms_norm/meta.py"
MM_KERNEL = "python/pto_kernels/ops/mc2/matmul_all_reduce/kernel.py"


def describe(repo_root, spec):
    return describe_pto(repo_root, KERNEL, META)


def _variant_env(variant) -> dict[str, str]:
    return {
        "PTO_MC2_MM_AR_WORLD_SIZE": str(variant.expected_world_size),
        "PTO_MC2_MM_AR_M": str(variant.m),
        "PTO_MC2_MM_AR_K": str(variant.k),
        "PTO_MC2_MM_AR_N": str(variant.n),
        "PTO_MC2_MM_AR_BASE_M": os.environ.get("PTO_MC2_MM_AR_BASE_M", "32"),
        "PTO_MC2_MM_AR_BASE_N": os.environ.get("PTO_MC2_MM_AR_BASE_N", "32"),
        "PTO_MC2_MM_AR_BASE_K": os.environ.get("PTO_MC2_MM_AR_BASE_K", "64"),
        "PTO_MC2_MM_AR_BLOCK_DIM": os.environ.get("PTO_MC2_MM_AR_BLOCK_DIM", "4"),
        "PTO_MC2_MM_ARN_WORLD_SIZE": str(variant.expected_world_size),
        "PTO_MC2_MM_ARN_M": str(variant.m),
        "PTO_MC2_MM_ARN_N": str(variant.n),
        "PTO_MC2_MM_ARN_BLOCK_DIM": os.environ.get("PTO_MC2_MM_ARN_BLOCK_DIM", "8"),
    }


def compile_kernel(repo_root, spec, artifacts_dir):
    del spec
    try:
        variant = VARIANTS[0]
        artifact_paths: list[str] = []
        library_paths: list[str | None] = []
        with temporary_env(_variant_env(variant)):
            for kernel_path, build_name in (
                (MM_KERNEL, "build_jit_wrapper"),
                (KERNEL, "build_add_rms_norm_jit_wrapper"),
            ):
                module = load_module(repo_root / kernel_path)
                builder = getattr(module, build_name, None)
                if not callable(builder):
                    return {
                        "status": "blocked",
                        "reason": f"{kernel_path} does not expose {build_name}(output_dir)",
                    }
                wrapper = builder(output_dir=Path(artifacts_dir) / Path(kernel_path).stem)
                build = getattr(wrapper, "_build", None)
                if callable(build):
                    build()
                artifact_paths.extend([str(path) for path in getattr(wrapper, "_artifact_paths", lambda: ())()])
                library_paths.append(getattr(wrapper, "library_path", None))
    except Exception as exc:  # pragma: no cover - exercised on NPU hosts
        return {"status": "blocked", "reason": f"PTO compile failed: {exc}"}
    return {
        "status": "ready",
        "kernel_path": str(repo_root / KERNEL),
        "output_dir": str(artifacts_dir),
        "artifact_paths": artifact_paths,
        "library_paths": library_paths,
    }


def benchmark(repo_root, spec, artifacts_dir):
    try:
        variant_reports = []
        artifact_paths: list[str] = []
        for variant in VARIANTS:
            with temporary_env(_variant_env(variant)):
                for kernel_path, build_name, suffix in (
                    (MM_KERNEL, "build_jit_wrapper", "mm_compile_probe"),
                    (KERNEL, "build_add_rms_norm_jit_wrapper", "arn_compile_probe"),
                ):
                    module = load_module(repo_root / kernel_path)
                    builder = getattr(module, build_name, None)
                    if not callable(builder):
                        return {
                            "status": "blocked",
                            "reason": f"{kernel_path} does not expose {build_name}(output_dir)",
                        }
                    wrapper = builder(output_dir=Path(artifacts_dir) / variant.label / suffix)
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
        report_path = Path(artifacts_dir) / "ptodsl_matmul_all_reduce_add_rms_norm_benchmark.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        report["report_path"] = str(report_path)
        return report

    if any(item.get("status") != "ok" for item in variant_reports):
        first_blocked = next(item for item in variant_reports if item.get("status") != "ok")
        report = {
            "status": "blocked",
            "variants": [variant.as_dict() for variant in VARIANTS],
            "reason": first_blocked.get("reason", "Distributed PTO matmul_all_reduce_add_rms_norm launch failed."),
            "variant_reports": variant_reports,
            "artifact_paths": artifact_paths,
        }
    else:
        y_diff = max(float(item["correctness"]["y_max_abs_diff"]) for item in variant_reports)
        norm_diff = max(float(item["correctness"]["norm_max_abs_diff"]) for item in variant_reports)
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
                "y_max_abs_diff": y_diff,
                "norm_max_abs_diff": norm_diff,
                "max_abs_diff": max_abs_diff,
                "atol": spec.correctness.atol,
                "rtol": spec.correctness.rtol,
                "passes": bool(max_abs_diff <= spec.correctness.atol),
            },
            "variant_reports": variant_reports,
            "artifact_paths": artifact_paths,
            "reference_contract": "pto_local_matmul_then_all_reduce_then_add_rms_norm",
        }

    report_path = Path(artifacts_dir) / "ptodsl_matmul_all_reduce_add_rms_norm_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
