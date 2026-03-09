import argparse
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from jit_util_matmul import jit_compile as jit_compile_custom
from jit_util_original_pto import jit_compile as jit_compile_original

DEVICE = os.environ.get("NPU_DEVICE", "npu:0")
DTYPE = torch.float16

N_REPEAT = 20
N_WARMUP = 5
N_ALLOC = N_REPEAT + N_WARMUP

# Custom-kernel swizzle runtime params (used by jit_util_matmul.py wrapper).
# Direction: 0=Zn, 1=Nz. Count <= 0 disables swizzle.
CUSTOM_SWIZZLE_DIRECTION = 1
CUSTOM_SWIZZLE_COUNT = 3
DEFAULT_SWIZZLE_CONFIGS = [(CUSTOM_SWIZZLE_DIRECTION, CUSTOM_SWIZZLE_COUNT)]

M_LIST = [128 * i for i in range(1, 33)]  # 128, 256, ..., 4096

# B is [N, K], output is [M, N]
SHAPES_NK = [
    (16384, 16384),
]

DEFAULT_CSV_REL_PATH = Path("outputs") / "csv" / "matmul_timing.csv"


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark torch/custom/original matmul kernels and save the results to "
            "a CSV file."
        )
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(DEFAULT_CSV_REL_PATH),
        help=f"Output CSV path (default: {DEFAULT_CSV_REL_PATH})",
    )
    parser.add_argument(
        "--swizzle",
        action="append",
        default=[],
        help=(
            "Custom swizzle config as 'direction,count' (or 'direction:count'). "
            "Repeat this argument to compare multiple swizzles, e.g. "
            "--swizzle 0,0 --swizzle 0,5 --swizzle 1,12"
        ),
    )
    parser.add_argument(
        "--with-torch",
        action="store_true",
        help="Include torch baseline timing/throughput benchmarking.",
    )
    parser.add_argument(
        "--with-original",
        action="store_true",
        help="Include the original PTO backend in the benchmark run.",
    )
    return parser.parse_args()


def _parse_swizzle_pair(raw: str) -> tuple[int, int]:
    text = raw.strip().replace(":", ",")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(
            f"Invalid --swizzle value '{raw}'. Expected 'direction,count' (for example: 0,5)."
        )

    direction = int(parts[0])
    count = int(parts[1])
    if direction not in (0, 1):
        raise ValueError(
            f"Invalid swizzle direction {direction}. Supported values are 0 or 1."
        )
    return direction, count


def _build_swizzle_configs(raw_values: list[str]) -> list[tuple[int, int]]:
    if not raw_values:
        return list(DEFAULT_SWIZZLE_CONFIGS)

    configs = []
    seen = set()
    for raw in raw_values:
        cfg = _parse_swizzle_pair(raw)
        if cfg in seen:
            continue
        configs.append(cfg)
        seen.add(cfg)

    if not configs:
        raise ValueError("No valid swizzle configs were provided.")
    return configs


def _time_backend(func, a_list, b_list):
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for a, b in zip(
        a_list[N_WARMUP : N_WARMUP + N_REPEAT], b_list[N_WARMUP : N_WARMUP + N_REPEAT]
    ):
        func(a, b)
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) / N_REPEAT * 1e3


def _bench_backend(func, a_list, b_list, c_ref):
    c = None
    for a, b in zip(a_list[:N_WARMUP], b_list[:N_WARMUP]):
        c = func(a, b)
    if c is None:
        raise RuntimeError("backend returned no output")

    mean_diff = float(torch.mean(torch.abs(c - c_ref)).cpu())
    abs_error = float(torch.max(torch.abs(c - c_ref)).cpu())
    dur_us = _time_backend(func, a_list, b_list)
    return {
        "time_us": dur_us,
        "mean_diff": mean_diff,
        "abs_error": abs_error,
        "out_elem_bytes": int(c.element_size()),
    }


def bench_one_shape(
    custom_backend,
    original_backend,
    swizzle_configs,
    m,
    n,
    k,
    original_enabled,
    torch_enabled,
):
    print(f"\n=== (M, N, K) = {m}, {n}, {k} ===")

    a_list = [torch.randn(m, k, dtype=DTYPE, device=DEVICE) for _ in range(N_ALLOC)]
    b_list = [torch.randn(n, k, dtype=DTYPE, device=DEVICE) for _ in range(N_ALLOC)]

    ref_a = a_list[N_WARMUP - 1]
    ref_b = b_list[N_WARMUP - 1]
    c_ref = F.linear(ref_a, ref_b)

    flops = 2.0 * m * n * k
    torch_total_bytes = (m * k + n * k) * 2 + m * n * int(c_ref.element_size())

    if torch_enabled:
        for a, b in zip(a_list[:N_WARMUP], b_list[:N_WARMUP]):
            F.linear(a, b)
        dur_ref_us = _time_backend(F.linear, a_list, b_list)
        torch_tflops = flops / dur_ref_us / 1e6
        torch_bandwidth = torch_total_bytes * 1e6 / dur_ref_us / (1024**3)
        torch_error = ""
        print(f"torch duration: {dur_ref_us:.3f} us")
        print(f"torch TFLOPS: {torch_tflops:.3f}")
    else:
        dur_ref_us = float("nan")
        torch_tflops = float("nan")
        torch_bandwidth = float("nan")
        torch_error = "disabled"

    base_record = {
        "M": m,
        "N": n,
        "K": k,
        "torch_time_us": dur_ref_us,
        "torch_tflops": torch_tflops,
        "torch_bandwidth_gbs": torch_bandwidth,
        "torch_error": torch_error,
    }

    original_stats = None
    original_error = ""
    if original_backend is not None:
        try:
            original_stats = _bench_backend(original_backend, a_list, b_list, c_ref)
            print(f"original duration: {original_stats['time_us']:.3f} us")
            print(f"original TFLOPS: {flops / original_stats['time_us'] / 1e6:.3f}")
            print(f"original mean diff: {original_stats['mean_diff']}")
        except Exception as exc:
            original_error = str(exc)
            print(f"original unavailable: {original_error}")
    else:
        if original_enabled:
            original_error = "backend not compiled"
            print(f"original unavailable: {original_error}")
        else:
            original_error = "disabled"

    records = []
    for direction, count in swizzle_configs:
        swizzle_label = f"d{direction}_c{count}"

        def _custom_with_swizzle(a, b, _fn=custom_backend, _d=direction, _c=count):
            return _fn(a, b, swizzle_direction=_d, swizzle_count=_c)

        record = {
            **base_record,
            "custom_swizzle_direction": direction,
            "custom_swizzle_count": count,
            "custom_swizzle": swizzle_label,
        }

        try:
            custom_stats = _bench_backend(_custom_with_swizzle, a_list, b_list, c_ref)
            custom_tflops = flops / custom_stats["time_us"] / 1e6
            custom_total_bytes = (m * k + n * k) * 2 + m * n * custom_stats[
                "out_elem_bytes"
            ]
            custom_bw = custom_total_bytes * 1e6 / custom_stats["time_us"] / (1024**3)
            record["custom_time_us"] = custom_stats["time_us"]
            record["custom_tflops"] = custom_tflops
            record["custom_bandwidth_gbs"] = custom_bw
            record["custom_mean_diff"] = custom_stats["mean_diff"]
            record["custom_abs_error"] = custom_stats["abs_error"]
            record["custom_speedup_vs_torch"] = dur_ref_us / custom_stats["time_us"]
            record["custom_error"] = ""
            print(
                f"custom({swizzle_label}) duration: {custom_stats['time_us']:.3f} us, "
                f"TFLOPS: {custom_tflops:.3f}, mean diff: {custom_stats['mean_diff']}"
            )
        except Exception as exc:
            record["custom_time_us"] = float("nan")
            record["custom_tflops"] = float("nan")
            record["custom_bandwidth_gbs"] = float("nan")
            record["custom_mean_diff"] = float("nan")
            record["custom_abs_error"] = float("nan")
            record["custom_speedup_vs_torch"] = float("nan")
            record["custom_error"] = str(exc)
            print(f"custom({swizzle_label}) unavailable: {record['custom_error']}")

        if original_stats is not None:
            original_tflops = flops / original_stats["time_us"] / 1e6
            original_total_bytes = (m * k + n * k) * 2 + m * n * original_stats[
                "out_elem_bytes"
            ]
            original_bw = (
                original_total_bytes * 1e6 / original_stats["time_us"] / (1024**3)
            )
            record["original_time_us"] = original_stats["time_us"]
            record["original_tflops"] = original_tflops
            record["original_bandwidth_gbs"] = original_bw
            record["original_mean_diff"] = original_stats["mean_diff"]
            record["original_abs_error"] = original_stats["abs_error"]
            record["original_speedup_vs_torch"] = dur_ref_us / original_stats["time_us"]
            record["original_error"] = ""
        else:
            record["original_time_us"] = float("nan")
            record["original_tflops"] = float("nan")
            record["original_bandwidth_gbs"] = float("nan")
            record["original_mean_diff"] = float("nan")
            record["original_abs_error"] = float("nan")
            record["original_speedup_vs_torch"] = float("nan")
            record["original_error"] = original_error

        records.append(record)

    return records


def main():
    args = _parse_args()
    swizzle_configs = _build_swizzle_configs(args.swizzle)
    include_torch = args.with_torch

    include_original = args.with_original
    original_reason = (
        "enabled by --with-original" if include_original else "disabled by default"
    )

    torch.npu.set_device(DEVICE)
    base = Path(__file__).resolve().parent

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = base / csv_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        "Custom swizzle configs: "
        + ", ".join(f"(direction={d}, count={c})" for d, c in swizzle_configs)
    )
    if include_torch:
        print("Torch baseline: enabled")
    print(
        f"Original PTO backend: {'enabled' if include_original else 'disabled'} ({original_reason})"
    )

    custom_backend = None
    original_backend = None

    custom_path = base / "matmul_custom_pto.cpp"
    try:
        print(f"Compiling {custom_path.name} for backend 'custom' ...")
        custom_backend = jit_compile_custom(str(custom_path), verbose=True)
    except Exception as exc:
        print(f"[WARN] backend 'custom' unavailable: {exc}")

    if include_original:
        original_path = base / "matmul_original_pto.cpp"
        try:
            print(f"Compiling {original_path.name} for backend 'original' ...")
            original_backend = jit_compile_original(str(original_path), verbose=True)
        except Exception as exc:
            print(f"[WARN] backend 'original' unavailable: {exc}")
    else:
        print("Skipping compile for backend 'original'.")

    if custom_backend is None:
        raise RuntimeError("Custom PTO backend could not be compiled.")

    records = []
    for n, k in SHAPES_NK:
        for m in M_LIST:
            records.extend(
                bench_one_shape(
                    custom_backend,
                    original_backend,
                    swizzle_configs,
                    m,
                    n,
                    k,
                    include_original,
                    include_torch,
                )
            )

    df = pd.DataFrame.from_records(records)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved benchmark CSV: {csv_path}")


if __name__ == "__main__":
    main()
