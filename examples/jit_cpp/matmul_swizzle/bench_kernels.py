import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
import torch_npu  # noqa

from jit_util_matmul import jit_compile as jit_compile_custom
from jit_util_original_pto import jit_compile as jit_compile_original

DEVICE = os.environ.get("NPU_DEVICE", "npu:0")
DTYPE = torch.float16

N_REPEAT = 20
N_WARMUP = 5
N_ALLOC = N_REPEAT + N_WARMUP

M_LIST = [128 * i for i in range(1, 33)]  # 128, 256, ..., 4096

# B is [N, K], output is [M, N]
SHAPES_NK = [
    (4096, 4096),
]

PLOT_N = 4096
PLOT_K = 4096

BACKEND_STYLE = {
    "torch": {"color": "#111111", "marker": "x", "linestyle": "--"},
    "custom": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "original": {"color": "#ff7f0e", "marker": "s", "linestyle": "-."},
}


def _shape_label(n: int, k: int) -> str:
    return f"N={int(n)},K={int(k)}"


def _shape_colors(df: pd.DataFrame):
    shape_keys = list(df.groupby(["N", "K"], sort=False).groups.keys())
    cmap_name = "tab10" if len(shape_keys) <= 10 else "tab20"
    cmap = plt.get_cmap(cmap_name, max(1, len(shape_keys)))
    return {key: cmap(i) for i, key in enumerate(shape_keys)}


def _style(name: str) -> dict:
    return BACKEND_STYLE.get(
        name, {"color": "#2ca02c", "marker": "^", "linestyle": "-"}
    )


def _time_backend(func, a_list, b_list):
    torch.npu.synchronize()
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


def bench_one_shape(backends, m, n, k):
    print(f"\n === (M, N, K) = {m}, {n}, {k} ===")

    a_list = [torch.randn(m, k, dtype=DTYPE, device=DEVICE) for _ in range(N_ALLOC)]
    b_list = [torch.randn(n, k, dtype=DTYPE, device=DEVICE) for _ in range(N_ALLOC)]

    ref_a = a_list[N_WARMUP - 1]
    ref_b = b_list[N_WARMUP - 1]
    c_ref = F.linear(ref_a, ref_b)

    backend_stats = {}
    backend_errors = {}
    for name, func in backends.items():
        try:
            c = None
            for a, b in zip(a_list[:N_WARMUP], b_list[:N_WARMUP]):
                c = func(a, b)
            if c is None:
                raise RuntimeError("backend returned no output")
            mean_diff = float(torch.mean(torch.abs(c - c_ref)).cpu())
            abs_error = float(torch.max(torch.abs(c - c_ref)).cpu())
            dur_us = _time_backend(func, a_list, b_list)
            backend_stats[name] = {
                "time_us": dur_us,
                "mean_diff": mean_diff,
                "abs_error": abs_error,
                "out_elem_bytes": int(c.element_size()),
            }
        except Exception as exc:
            backend_errors[name] = str(exc)

    for a, b in zip(a_list[:N_WARMUP], b_list[:N_WARMUP]):
        F.linear(a, b)
    dur_ref_us = _time_backend(F.linear, a_list, b_list)

    flops = 2.0 * m * n * k
    torch_total_bytes = (m * k + n * k) * 2 + m * n * int(c_ref.element_size())

    record = {
        "M": m,
        "N": n,
        "K": k,
        "torch_time_us": dur_ref_us,
        "torch_tflops": flops / dur_ref_us / 1e6,
        "torch_bandwidth_gbs": torch_total_bytes * 1e6 / dur_ref_us / (1024**3),
    }

    print(f"torch duration: {dur_ref_us:.3f} us")
    print(f"torch TFLOPS: {record['torch_tflops']:.3f}")

    for name in backends:
        if name in backend_stats:
            s = backend_stats[name]
            tflops = flops / s["time_us"] / 1e6
            total_bytes_backend = (m * k + n * k) * 2 + m * n * s["out_elem_bytes"]
            bw = total_bytes_backend * 1e6 / s["time_us"] / (1024**3)
            record[f"{name}_time_us"] = s["time_us"]
            record[f"{name}_tflops"] = tflops
            record[f"{name}_bandwidth_gbs"] = bw
            record[f"{name}_mean_diff"] = s["mean_diff"]
            record[f"{name}_abs_error"] = s["abs_error"]
            record[f"{name}_speedup_vs_torch"] = dur_ref_us / s["time_us"]
            record[f"{name}_error"] = ""
            print(f"{name} duration: {s['time_us']:.3f} us")
            print(f"{name} TFLOPS: {tflops:.3f}")
            print(f"{name} mean diff: {s['mean_diff']}")
        else:
            record[f"{name}_time_us"] = float("nan")
            record[f"{name}_tflops"] = float("nan")
            record[f"{name}_bandwidth_gbs"] = float("nan")
            record[f"{name}_mean_diff"] = float("nan")
            record[f"{name}_abs_error"] = float("nan")
            record[f"{name}_speedup_vs_torch"] = float("nan")
            record[f"{name}_error"] = backend_errors.get(name, "unknown error")
            print(f"{name} unavailable: {record[f'{name}_error']}")

    return record


def plot_runtime(df: pd.DataFrame, out_dir: Path, backends: list[str]) -> Path:
    plt.figure(figsize=(10, 5))
    for (n, k), group in df.groupby(["N", "K"], sort=False):
        g = group.sort_values("M")
        label = _shape_label(n, k)
        torch_style = _style("torch")
        plt.plot(
            g["M"],
            g["torch_time_us"],
            marker=torch_style["marker"],
            linestyle=torch_style["linestyle"],
            color=torch_style["color"],
            label=f"torch ({label})",
        )
        for b in backends:
            c = f"{b}_time_us"
            if c not in g.columns:
                continue
            style = _style(b)
            plt.plot(
                g["M"],
                g[c],
                marker=style["marker"],
                linestyle=style["linestyle"],
                color=style["color"],
                label=f"{b} ({label})",
            )
    plt.xlabel("M")
    plt.ylabel("Runtime (us)")
    plt.title("Runtime vs M")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_path = out_dir / "duration.png"
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def plot_tflops(df: pd.DataFrame, out_dir: Path, backends: list[str]) -> Path:
    plt.figure(figsize=(10, 5))
    for (n, k), group in df.groupby(["N", "K"], sort=False):
        g = group.sort_values("M")
        label = _shape_label(n, k)
        torch_style = _style("torch")
        plt.plot(
            g["M"],
            g["torch_tflops"],
            marker=torch_style["marker"],
            linestyle=torch_style["linestyle"],
            color=torch_style["color"],
            label=f"torch ({label})",
        )
        for b in backends:
            c = f"{b}_tflops"
            if c not in g.columns:
                continue
            style = _style(b)
            plt.plot(
                g["M"],
                g[c],
                marker=style["marker"],
                linestyle=style["linestyle"],
                color=style["color"],
                label=f"{b} ({label})",
            )
    plt.xlabel("M")
    plt.ylabel("TFLOPS")
    plt.title("TFLOPS vs M")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_path = out_dir / "flops.png"
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def plot_error(df: pd.DataFrame, out_dir: Path, backends: list[str]) -> Path:
    plt.figure(figsize=(10, 5))
    for (n, k), group in df.groupby(["N", "K"], sort=False):
        g = group.sort_values("M")
        label = _shape_label(n, k)
        for b in backends:
            c = f"{b}_mean_diff"
            if c not in g.columns:
                continue
            style = _style(b)
            plt.plot(
                g["M"],
                g[c],
                marker=style["marker"],
                linestyle=style["linestyle"],
                color=style["color"],
                label=f"{b} ({label})",
            )
    plt.xlabel("M")
    plt.ylabel("Mean Abs Error")
    plt.title("Error vs M")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_path = out_dir / "error.png"
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def main():
    torch.npu.set_device(DEVICE)
    base = Path(__file__).resolve().parent

    backend_builders = {
        "custom": (jit_compile_custom, base / "matmul_custom_pto.cpp"),
        "original": (jit_compile_original, base / "matmul_original_pto.cpp"),
    }

    backends = {}
    for name, (builder, path) in backend_builders.items():
        try:
            print(f"Compiling {path.name} for backend '{name}' ...")
            backends[name] = builder(str(path), verbose=True)
        except Exception as exc:
            print(f"[WARN] backend '{name}' unavailable: {exc}")

    if not backends:
        raise RuntimeError("No custom PTO backends could be compiled.")

    records = []
    for n, k in SHAPES_NK:
        for m in M_LIST:
            records.append(bench_one_shape(backends, m, n, k))

    csv_dir = base / "outputs" / "csv"
    plot_dir = base / "outputs" / "plots"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    csv_path = csv_dir / "matmul_timing.csv"
    df = pd.DataFrame.from_records(records)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved benchmark CSV: {csv_path}")

    backend_names = [b for b in ("custom", "original") if f"{b}_time_us" in df.columns]
    plot_df = df[(df["N"] == PLOT_N) & (df["K"] == PLOT_K)]
    if plot_df.empty:
        raise RuntimeError(f"No rows found for plot filter N={PLOT_N}, K={PLOT_K}")

    runtime_path = plot_runtime(plot_df, plot_dir, backend_names)
    flops_path = plot_tflops(plot_df, plot_dir, backend_names)
    error_path = plot_error(plot_df, plot_dir, backend_names)

    print(f"Saved runtime plot (N={PLOT_N}, K={PLOT_K}): {runtime_path}")
    print(f"Saved flops plot (N={PLOT_N}, K={PLOT_K}):   {flops_path}")
    print(f"Saved error plot (N={PLOT_N}, K={PLOT_K}):   {error_path}")


if __name__ == "__main__":
    main()
