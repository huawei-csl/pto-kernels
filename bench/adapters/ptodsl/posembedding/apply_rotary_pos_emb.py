from __future__ import annotations

import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch

from pto_kernels.bench.adapter_utils import compile_pto_kernel, describe_pto
from pto_kernels.ops.posembedding.apply_rotary_pos_emb.runtime import (
    VARIANTS,
    make_inputs,
    reset_inplace_inputs,
    run_pto_variant,
)


KERNEL = "python/pto_kernels/ops/posembedding/apply_rotary_pos_emb/kernel.py"
META = "python/pto_kernels/ops/posembedding/apply_rotary_pos_emb/meta.py"


def describe(repo_root, spec):
    return describe_pto(repo_root, KERNEL, META)


def compile_kernel(repo_root, spec, artifacts_dir):
    return compile_pto_kernel(repo_root, KERNEL, artifacts_dir)


def benchmark(repo_root, spec, artifacts_dir):
    kernel_file = repo_root / KERNEL
    spec_obj = importlib.util.spec_from_file_location("pto_apply_rotary_pos_emb_kernel", kernel_file)
    if spec_obj is None or spec_obj.loader is None:
        return {"status": "blocked", "reason": f"Unable to import {kernel_file}"}

    module = importlib.util.module_from_spec(spec_obj)
    spec_obj.loader.exec_module(module)
    wrapper = module.build_jit_wrapper(output_dir=artifacts_dir)
    build = getattr(wrapper, "_build", None)
    try:
        if callable(build):
            build()
    except Exception as exc:  # pragma: no cover - exercised on NPU hosts
        report = {
            "status": "blocked",
            "variants": [variant.as_dict() for variant in VARIANTS],
            "reason": f"PTO compile failed: {exc}",
        }
        report_path = Path(artifacts_dir) / "ptodsl_apply_rotary_pos_emb_benchmark.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        report["report_path"] = str(report_path)
        return report

    try:
        variant_reports = []
        for variant in VARIANTS:
            inputs = make_inputs(variant, device_index=int(spec.device.get("id", 0)))
            for _ in range(spec.bench.warmup):
                reset_inplace_inputs(inputs)
                run_pto_variant(wrapper, inputs)
            torch.npu.synchronize()

            timings_ms = []
            output = None
            for _ in range(spec.bench.repeat):
                reset_inplace_inputs(inputs)
                torch.npu.synchronize()
                start = time.perf_counter()
                output = run_pto_variant(wrapper, inputs)
                torch.npu.synchronize()
                timings_ms.append((time.perf_counter() - start) * 1000.0)

            if output is None:
                raise RuntimeError(f"PTO benchmark did not produce an output tensor for {variant.layout}.")

            query_out, key_out = output
            query_diff = (query_out.float().cpu() - inputs["reference_query"]).abs().max().item()
            key_diff = (key_out.float().cpu() - inputs["reference_key"]).abs().max().item()
            variant_reports.append(
                {
                    "variant": variant.as_dict(),
                    "shape_summary": inputs["shape_summary"],
                    "timings_ms": {
                        "median": statistics.median(timings_ms),
                        "min": min(timings_ms),
                        "max": max(timings_ms),
                    },
                    "correctness": {
                        "query_max_abs_diff": query_diff,
                        "key_max_abs_diff": key_diff,
                        "max_abs_diff": max(query_diff, key_diff),
                    },
                }
            )
    except Exception as exc:  # pragma: no cover - exercised on NPU hosts
        report = {
            "status": "blocked",
            "variants": [variant.as_dict() for variant in VARIANTS],
            "reason": f"PTO execution failed: {exc}",
            "artifact_paths": [str(path) for path in getattr(wrapper, "_artifact_paths", lambda: ())()],
        }
        report_path = Path(artifacts_dir) / "ptodsl_apply_rotary_pos_emb_benchmark.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        report["report_path"] = str(report_path)
        return report

    max_abs_diff = max(item["correctness"]["max_abs_diff"] for item in variant_reports)
    report = {
        "status": "ok",
        "variants": [item["variant"] for item in variant_reports],
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
        "artifact_paths": [str(path) for path in getattr(wrapper, "_artifact_paths", lambda: ())()],
        "reference_contract": "fp16_half_rope_tnd_and_bsnd",
    }
    report_path = Path(artifacts_dir) / "ptodsl_apply_rotary_pos_emb_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
