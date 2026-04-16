import argparse
import os
from pathlib import Path

import pandas as pd
import torch

from jit_util_histogram import jit_compile

DEVICE = os.environ.get("NPU_DEVICE", "npu:1")
DTYPE = torch.float32

N_REPEAT = 20
N_WARMUP = 5
N_ALLOC = N_REPEAT + N_WARMUP

TILE_SIZES = [512, 1024, 2048, 4096, 8192]
BINS_LIST = [8, 32, 64, 128, 192, 256]
MIN_VAL = 0.0
MAX_VAL = 255.0

DEFAULT_CSV_REL_PATH = Path("outputs") / "csv" / "histogram_timing.csv"


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark torch and implementation histogram kernels and save the results to "
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
        "--implementation",
        type=int,
        nargs="*",
        choices=[1, 2],
        help="Select the implementation steps (1: naive, 2: double buffering). If not provided, benchmarks all.",
    )
    parser.add_argument(
        "--with-torch",
        action="store_true",
        help="Include torch baseline timing/throughput benchmarking.",
    )
    return parser.parse_args()


IMPLEMENTATIONS = {
    1: "step1_naive_histogram",
    2: "step2_double_buffering",
}


def _bench_backend(
    name, func, a_list, z_list, c_ref, processed_elements, total_bytes, bins
):
    c = None
    for a, z in zip(a_list[:N_WARMUP], z_list[:N_WARMUP]):
        res = func(a, z, bins, MIN_VAL, MAX_VAL)
        c = res if res is not None else z

    mean_diff = float(torch.mean(torch.abs(c.float() - c_ref.float())).cpu())
    abs_error = float(torch.max(torch.abs(c.float() - c_ref.float())).cpu())

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for a, z in zip(
        a_list[N_WARMUP : N_WARMUP + N_REPEAT], z_list[N_WARMUP : N_WARMUP + N_REPEAT]
    ):
        func(a, z, bins, MIN_VAL, MAX_VAL)
    end.record()
    torch.npu.synchronize()
    dur_us = start.elapsed_time(end) / N_REPEAT * 1e3

    gmelem_s = processed_elements / dur_us / 1e3
    bw_gbs = total_bytes * 1e6 / dur_us / (1024**3)

    print(
        f"{name} duration: {dur_us:.3f} us, GElem/s: {gmelem_s:.3f}, mean diff: {mean_diff}"
    )

    return {
        f"{name}_time_us": dur_us,
        f"{name}_gmelem_s": gmelem_s,
        f"{name}_bandwidth_gbs": bw_gbs,
        f"{name}_mean_diff": mean_diff,
        f"{name}_abs_error": abs_error,
        f"{name}_error": "",
    }


def bench_n_elems(funcs_to_bench, num_elements, bins, tile_size):
    print(f"\n=== N = {num_elements:_} | bins = {bins} | tile_size = {tile_size} ===")

    a_list = [
        torch.rand(num_elements, dtype=DTYPE, device=DEVICE) * (MAX_VAL - MIN_VAL)
        + MIN_VAL
        for _ in range(N_ALLOC)
    ]
    z_list = [
        torch.zeros(bins, device=DEVICE, dtype=torch.int32) for _ in range(N_ALLOC)
    ]

    ref_a = a_list[N_WARMUP - 1]
    c_ref = torch.histc(ref_a, bins=bins, min=MIN_VAL, max=MAX_VAL).to(torch.int32)

    processed_elements = num_elements
    total_bytes = num_elements * int(ref_a.element_size()) + bins * int(
        c_ref.element_size()
    )

    record = {
        "N": num_elements,
        "bins": bins,
        "tile_size": tile_size,
    }

    for name, func in funcs_to_bench.items():
        if func is None:
            record.update(
                {
                    f"{name}_time_us": float("nan"),
                    f"{name}_gmelem_s": float("nan"),
                    f"{name}_bandwidth_gbs": float("nan"),
                    f"{name}_mean_diff": float("nan"),
                    f"{name}_abs_error": float("nan"),
                    f"{name}_error": "backend not compiled/enabled",
                }
            )
            continue

        try:
            stats = _bench_backend(
                name, func, a_list, z_list, c_ref, processed_elements, total_bytes, bins
            )
            record.update(stats)
        except Exception as exc:
            print(f"{name} unavailable: {exc}")
            record.update(
                {
                    f"{name}_time_us": float("nan"),
                    f"{name}_gmelem_s": float("nan"),
                    f"{name}_bandwidth_gbs": float("nan"),
                    f"{name}_mean_diff": float("nan"),
                    f"{name}_abs_error": float("nan"),
                    f"{name}_error": str(exc),
                }
            )

    return [record]


def main():
    args = _parse_args()
    include_torch = args.with_torch
    impls_to_run = args.implementation if args.implementation else [1, 2]

    torch.npu.set_device(DEVICE)
    base = Path(__file__).resolve().parent

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = base / csv_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Implementations selected: {[IMPLEMENTATIONS[i] for i in impls_to_run]}")
    if include_torch:
        print("Torch baseline: enabled")

    vector_cores = torch.npu.get_device_properties().vector_core_num
    base_elements = max(TILE_SIZES) * vector_cores
    # multiplier = max(1, (1024 * 1024) // base_elements)
    multiplier = 1
    n_elements_list = [base_elements * multiplier * i for i in range(1, 33)]

    records = []
    for tile_size in TILE_SIZES:
        funcs_to_bench = {}

        if include_torch:

            def torch_hist_bench(x, _, bins, min_val, max_val):
                return torch.histc(x, bins=bins, min=min_val, max=max_val).to(
                    torch.int32
                )

            funcs_to_bench["torch"] = torch_hist_bench

        for i in impls_to_run:
            impl_dir = IMPLEMENTATIONS[i]
            custom_path = base / impl_dir / "kernel_histogram.cpp"
            name = f"step{i}"
            try:
                print(
                    f"Compiling {impl_dir}/kernel_histogram.cpp for backend '{name}' (tile_size={tile_size}) ..."
                )
                funcs_to_bench[name] = jit_compile(
                    str(custom_path), tile_size=tile_size
                )
            except Exception as exc:
                print(f"[WARN] backend '{name}' unavailable: {exc}")
                funcs_to_bench[name] = None

        for bins in BINS_LIST:
            for n in n_elements_list:
                records.extend(bench_n_elems(funcs_to_bench, n, bins, tile_size))

    df = pd.DataFrame.from_records(records)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved benchmark CSV: {csv_path}")


if __name__ == "__main__":
    main()
