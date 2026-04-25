from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu

from pto_kernels.bench.adapter_utils import describe_baseline
from pto_kernels.ops.gmm.grouped_matmul.runtime import (
    VARIANT,
    VARIANTS,
    make_dense_single_weight_inputs,
    run_torch_npu_grouped_matmul,
)


def describe(repo_root, spec):
    summary = describe_baseline(repo_root, "gmm", "grouped_matmul", spec.inventory_ref)
    summary["runtime_entrypoint"] = "torch_npu.npu_grouped_matmul"
    summary["seed_variant"] = {"default": VARIANT.as_dict(), "variants": [variant.as_dict() for variant in VARIANTS]}
    summary["upstream_build_status"] = (
        "fast_kernel_launch_example is currently blocked on this host because bisheng "
        "fails while compiling torch header dependencies in the example extension."
    )
    return summary


def compile_kernel(repo_root, spec, artifacts_dir):
    if not hasattr(torch_npu, "npu_grouped_matmul"):
        return {
            "status": "blocked",
            "reason": "torch_npu does not expose npu_grouped_matmul on this environment.",
        }
    return {
        "status": "runtime_builtin",
        "entrypoint": "torch_npu.npu_grouped_matmul",
        "note": (
            "Baseline execution currently relies on the runtime-installed op. "
            "Building the standalone ops-transformer example is tracked as a separate blocker."
        ),
    }


def benchmark(repo_root, spec, artifacts_dir):
    try:
        variant_reports = []
        for variant in VARIANTS:
            inputs = make_dense_single_weight_inputs(variant, device_index=int(spec.device.get("id", 0)))
            reference = inputs["baseline_reference"]
            for _ in range(spec.bench.warmup):
                run_torch_npu_grouped_matmul(inputs)
            torch.npu.synchronize()

            timings_ms = []
            output = None
            for _ in range(spec.bench.repeat):
                torch.npu.synchronize()
                start = time.perf_counter()
                output = run_torch_npu_grouped_matmul(inputs)
                torch.npu.synchronize()
                timings_ms.append((time.perf_counter() - start) * 1000.0)

            if output is None:
                raise RuntimeError(
                    f"Baseline benchmark did not produce an output tensor for {variant.label}."
                )

            baseline_tensor = output[0] if isinstance(output, (list, tuple)) else output
            max_abs_diff = (baseline_tensor.float().cpu() - reference).abs().max().item()
            variant_reports.append(
                {
                    "variant": variant.as_dict(),
                    "shape_summary": variant.shape_summary,
                    "timings_ms": {
                        "median": statistics.median(timings_ms),
                        "min": min(timings_ms),
                        "max": max(timings_ms),
                    },
                    "correctness": {
                        "max_abs_diff": max_abs_diff,
                    },
                    "output_type": str(type(output)),
                }
            )
    except Exception as exc:  # pragma: no cover - exercised on NPU hosts
        report = {
            "status": "blocked",
            "variants": [variant.as_dict() for variant in VARIANTS],
            "entrypoint": "torch_npu.npu_grouped_matmul",
            "reason": str(exc),
            "blocking_gap": "ops-transformer-grouped-matmul-v5-shape-contract",
        }
        report_path = Path(artifacts_dir) / "ops_transformer_grouped_matmul_benchmark.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        report["report_path"] = str(report_path)
        return report

    max_abs_diff = max(item["correctness"]["max_abs_diff"] for item in variant_reports)
    report = {
        "status": "ok",
        "variants": [item["variant"] for item in variant_reports],
        "entrypoint": "torch_npu.npu_grouped_matmul",
        "shape_summaries": [item["shape_summary"] for item in variant_reports],
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
        "reference_contract": "bf16_rounded_matmul",
    }
    report_path = Path(artifacts_dir) / "ops_transformer_grouped_matmul_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
