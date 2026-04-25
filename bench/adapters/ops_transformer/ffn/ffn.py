from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu

from pto_kernels.bench.adapter_utils import describe_baseline
from pto_kernels.ops.ffn.ffn.runtime import (
    VARIANT,
    VARIANTS,
    make_dense_relu_inputs,
    run_torch_npu_ffn,
)


def describe(repo_root, spec):
    summary = describe_baseline(repo_root, "ffn", "ffn", spec.inventory_ref)
    summary["runtime_entrypoint"] = "torch_npu.npu_ffn"
    summary["seed_variant"] = {"default": VARIANT.as_dict(), "variants": [variant.as_dict() for variant in VARIANTS]}
    return summary


def compile_kernel(repo_root, spec, artifacts_dir):
    if not hasattr(torch_npu, "npu_ffn"):
        return {
            "status": "blocked",
            "reason": "torch_npu does not expose npu_ffn on this environment.",
        }
    return {
        "status": "runtime_builtin",
        "entrypoint": "torch_npu.npu_ffn",
        "note": (
            "Baseline execution relies on the installed custom ops runtime package. "
            "The seed variant is constrained to dense float16 FFN with relu activation and no bias."
        ),
    }


def benchmark(repo_root, spec, artifacts_dir):
    try:
        variant_reports = []
        for variant in VARIANTS:
            inputs = make_dense_relu_inputs(variant, device_index=int(spec.device.get("id", 0)))
            reference = inputs["reference"]
            for _ in range(spec.bench.warmup):
                run_torch_npu_ffn(inputs)
            torch.npu.synchronize()

            timings_ms = []
            output = None
            for _ in range(spec.bench.repeat):
                torch.npu.synchronize()
                start = time.perf_counter()
                output = run_torch_npu_ffn(inputs)
                torch.npu.synchronize()
                timings_ms.append((time.perf_counter() - start) * 1000.0)

            if output is None:
                raise RuntimeError(f"Baseline benchmark did not produce an output tensor for {variant.label}.")

            max_abs_diff = (output.float().cpu() - reference).abs().max().item()
            variant_reports.append(
                {
                    "variant": variant.as_dict(),
                    "shape_summary": variant.shape_summary,
                    "timings_ms": {
                        "median": statistics.median(timings_ms),
                        "min": min(timings_ms),
                        "max": max(timings_ms),
                    },
                    "correctness": {"max_abs_diff": max_abs_diff},
                }
            )
    except Exception as exc:  # pragma: no cover - exercised on NPU hosts
        report = {
            "status": "blocked",
            "variants": [variant.as_dict() for variant in VARIANTS],
            "entrypoint": "torch_npu.npu_ffn",
            "reason": str(exc),
        }
        report_path = Path(artifacts_dir) / "ops_transformer_ffn_benchmark.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        report["report_path"] = str(report_path)
        return report

    max_abs_diff = max(item["correctness"]["max_abs_diff"] for item in variant_reports)
    report = {
        "status": "ok",
        "variants": [item["variant"] for item in variant_reports],
        "entrypoint": "torch_npu.npu_ffn",
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
        "reference_contract": "fp16_dense_relu_ffn_torch_ops",
    }
    report_path = Path(artifacts_dir) / "ops_transformer_ffn_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
