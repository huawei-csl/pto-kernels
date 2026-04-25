from __future__ import annotations

import importlib.util
import json
import os
import statistics
import time
from pathlib import Path

import torch

from pto_kernels.bench.adapter_utils import compile_pto_kernel, describe_pto, temporary_env
from pto_kernels.ops.gmm.grouped_matmul.runtime import (
    VARIANT,
    VARIANTS,
    make_dense_single_weight_inputs,
    run_pto_dense_variant,
)


KERNEL = "python/pto_kernels/ops/gmm/grouped_matmul/kernel.py"
META = "python/pto_kernels/ops/gmm/grouped_matmul/meta.py"


def describe(repo_root, spec):
    return describe_pto(repo_root, KERNEL, META)


def compile_kernel(repo_root, spec, artifacts_dir):
    return compile_pto_kernel(repo_root, KERNEL, artifacts_dir)


def _variant_env(variant) -> dict[str, str]:
    return {
        "PTO_GROUPED_MATMUL_M": str(variant.m),
        "PTO_GROUPED_MATMUL_K": str(variant.k),
        "PTO_GROUPED_MATMUL_N": str(variant.n),
        "PTO_GROUPED_MATMUL_BASE_M": os.environ.get("PTO_GROUPED_MATMUL_BASE_M", "16"),
        "PTO_GROUPED_MATMUL_BASE_N": os.environ.get("PTO_GROUPED_MATMUL_BASE_N", "64"),
        "PTO_GROUPED_MATMUL_BASE_K": os.environ.get("PTO_GROUPED_MATMUL_BASE_K", "64"),
        "PTO_GROUPED_MATMUL_BLOCK_DIM": os.environ.get("PTO_GROUPED_MATMUL_BLOCK_DIM", "16"),
    }


def benchmark(repo_root, spec, artifacts_dir):
    try:
        variant_reports = []
        artifact_paths: list[str] = []
        for variant in VARIANTS:
            variant_dir = Path(artifacts_dir) / variant.label
            kernel_file = repo_root / KERNEL
            spec_obj = importlib.util.spec_from_file_location(
                f"pto_grouped_matmul_kernel_{variant.label}",
                kernel_file,
            )
            if spec_obj is None or spec_obj.loader is None:
                raise ImportError(f"Unable to import {kernel_file}")

            with temporary_env(_variant_env(variant)):
                module = importlib.util.module_from_spec(spec_obj)
                spec_obj.loader.exec_module(module)
                wrapper = module.build_jit_wrapper(output_dir=variant_dir)
                build = getattr(wrapper, "_build", None)
                if callable(build):
                    build()
                set_block_dim = getattr(wrapper, "set_block_dim", None)
                if callable(set_block_dim):
                    base_m = int(os.environ.get("PTO_GROUPED_MATMUL_BASE_M", "16"))
                    base_n = int(os.environ.get("PTO_GROUPED_MATMUL_BASE_N", "64"))
                    total_tiles = (variant.m // base_m) * (variant.n // base_n)
                    requested_block_dim = int(os.environ.get("PTO_GROUPED_MATMUL_BLOCK_DIM", "16"))
                    set_block_dim(min(requested_block_dim, total_tiles))

                inputs = make_dense_single_weight_inputs(variant, device_index=int(spec.device.get("id", 0)))
                reference = inputs["baseline_reference"]

                for _ in range(spec.bench.warmup):
                    run_pto_dense_variant(wrapper, inputs)
                torch.npu.synchronize()

                timings_ms = []
                output = None
                for _ in range(spec.bench.repeat):
                    torch.npu.synchronize()
                    start = time.perf_counter()
                    output = run_pto_dense_variant(wrapper, inputs)
                    torch.npu.synchronize()
                    timings_ms.append((time.perf_counter() - start) * 1000.0)

                if output is None:
                    raise RuntimeError(f"PTO benchmark did not produce an output tensor for {variant.label}.")

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
                        "correctness": {
                            "max_abs_diff": max_abs_diff,
                        },
                    }
                )
                artifact_paths.extend(
                    [str(path) for path in getattr(wrapper, "_artifact_paths", lambda: ())()]
                )
    except Exception as exc:  # pragma: no cover - exercised on NPU hosts
        report = {
            "status": "blocked",
            "variants": [variant.as_dict() for variant in VARIANTS],
            "reason": f"PTO execution failed: {exc}",
        }
        report_path = Path(artifacts_dir) / "ptodsl_grouped_matmul_benchmark.json"
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
        "artifact_paths": artifact_paths,
    }
    report_path = Path(artifacts_dir) / "ptodsl_grouped_matmul_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
