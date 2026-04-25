from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu

from pto_kernels.bench.adapter_utils import describe_baseline
from pto_kernels.ops.moe.moe_token_permute.runtime import (
    VARIANT,
    VARIANTS,
    make_top1_permutation_inputs,
    run_torch_npu_moe_token_permute,
)


def describe(repo_root, spec):
    summary = describe_baseline(repo_root, "moe", "moe_token_permute", spec.inventory_ref)
    summary["runtime_entrypoint"] = "torch_npu.npu_moe_token_permute"
    summary["seed_variant"] = {"default": VARIANT.as_dict(), "variants": [variant.as_dict() for variant in VARIANTS]}
    return summary


def compile_kernel(repo_root, spec, artifacts_dir):
    if not hasattr(torch_npu, "npu_moe_token_permute"):
        return {
            "status": "blocked",
            "reason": "torch_npu does not expose npu_moe_token_permute on this environment.",
        }
    return {
        "status": "runtime_builtin",
        "entrypoint": "torch_npu.npu_moe_token_permute",
        "note": (
            "Baseline execution relies on the installed custom ops runtime package. "
            "The seed variant is constrained to top-1 permutation with 1D int32 indices and padded_mode=false."
        ),
    }


def benchmark(repo_root, spec, artifacts_dir):
    try:
        variant_reports = []
        for variant in VARIANTS:
            inputs = make_top1_permutation_inputs(variant, device_index=int(spec.device.get("id", 0)))
            for _ in range(spec.bench.warmup):
                run_torch_npu_moe_token_permute(inputs)
            torch.npu.synchronize()

            timings_ms = []
            output = None
            for _ in range(spec.bench.repeat):
                torch.npu.synchronize()
                start = time.perf_counter()
                output = run_torch_npu_moe_token_permute(inputs)
                torch.npu.synchronize()
                timings_ms.append((time.perf_counter() - start) * 1000.0)

            if output is None:
                raise RuntimeError(f"Baseline benchmark did not produce output tensors for {variant.label}.")

            permuted_tokens, sorted_indices = output
            token_diff = (permuted_tokens.float().cpu() - inputs["reference_tokens"]).abs().max().item()
            index_diff = (
                sorted_indices.to(torch.int32).cpu() - inputs["reference_sorted_indices"]
            ).abs().max().item()
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
                        "token_max_abs_diff": token_diff,
                        "sorted_index_max_abs_diff": index_diff,
                        "max_abs_diff": max(token_diff, float(index_diff)),
                    },
                }
            )
    except Exception as exc:  # pragma: no cover - exercised on NPU hosts
        report = {
            "status": "blocked",
            "variants": [variant.as_dict() for variant in VARIANTS],
            "entrypoint": "torch_npu.npu_moe_token_permute",
            "reason": str(exc),
        }
        report_path = Path(artifacts_dir) / "ops_transformer_moe_token_permute_benchmark.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        report["report_path"] = str(report_path)
        return report

    max_abs_diff = max(item["correctness"]["max_abs_diff"] for item in variant_reports)
    report = {
        "status": "ok",
        "variants": [item["variant"] for item in variant_reports],
        "entrypoint": "torch_npu.npu_moe_token_permute",
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
            "passes": bool(
                all(
                    item["correctness"]["token_max_abs_diff"] <= spec.correctness.atol
                    and item["correctness"]["sorted_index_max_abs_diff"] == 0
                    for item in variant_reports
                )
            ),
        },
        "variant_reports": variant_reports,
        "reference_contract": "top1_argsort_permute",
    }
    report_path = Path(artifacts_dir) / "ops_transformer_moe_token_permute_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
