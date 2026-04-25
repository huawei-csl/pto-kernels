from __future__ import annotations

import json
import os
from pathlib import Path

from pto_kernels.bench.adapter_utils import compile_pto_kernel, describe_pto, load_module, temporary_env
from pto_kernels.ops.mc2.moe_distribute_combine.runtime import (
    VARIANTS,
    run_distributed_pto_benchmark,
)


KERNEL = "python/pto_kernels/ops/mc2/moe_distribute_combine/kernel.py"
META = "python/pto_kernels/ops/mc2/moe_distribute_combine/meta.py"


def describe(repo_root, spec):
    return describe_pto(repo_root, KERNEL, META)


def compile_kernel(repo_root, spec, artifacts_dir):
    return compile_pto_kernel(repo_root, KERNEL, artifacts_dir)


def _variant_env(variant) -> dict[str, str]:
    return {
        "PTO_MC2_MOE_COMBINE_TOKENS": str(variant.tokens),
        "PTO_MC2_MOE_COMBINE_HIDDEN": str(variant.hidden_size),
        "PTO_MC2_MOE_COMBINE_WORLD_SIZE": str(variant.expected_world_size),
        "PTO_MC2_MOE_COMBINE_BLOCK_DIM": os.environ.get("PTO_MC2_MOE_COMBINE_BLOCK_DIM", "4"),
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

                variant_reports.append(
                    run_distributed_pto_benchmark(
                        variant=variant,
                        artifacts_dir=Path(artifacts_dir) / variant.label,
                        warmup=spec.bench.warmup,
                        repeat=spec.bench.repeat,
                    )
                )
    except Exception as exc:  # pragma: no cover - exercised on NPU hosts
        report = {
            "status": "blocked",
            "variants": [variant.as_dict() for variant in VARIANTS],
            "reason": f"PTO compile failed: {exc}",
        }
        report_path = Path(artifacts_dir) / "ptodsl_moe_distribute_combine_benchmark.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        report["report_path"] = str(report_path)
        return report

    first_blocked = next((item for item in variant_reports if item.get("status") != "ok"), None)
    if first_blocked is not None:
        report = {
            "status": "blocked",
            "variants": [variant.as_dict() for variant in VARIANTS],
            "reason": first_blocked.get("reason", "Distributed PTO moe_distribute_combine launch failed."),
            "variant_reports": variant_reports,
            "artifact_paths": artifact_paths,
        }
    else:
        report = {
            "status": "ok",
            "variants": [item["variant"] for item in variant_reports],
            "artifact_paths": artifact_paths,
        }

    report_path = Path(artifacts_dir) / "ptodsl_moe_distribute_combine_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
