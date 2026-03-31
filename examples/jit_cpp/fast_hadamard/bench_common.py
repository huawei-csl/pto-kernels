import math
import statistics
from pathlib import Path

import torch

from jit_util_common import DEFAULT_DEVICE, chmod_output_path, normalize_npu_device

BENCH_BATCHES = [1, 5, 8, 10, 16, 20, 32, 40, 64, 128, 256, 512, 1024]
BENCH_HIDDEN_DIMS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]

DTYPE = torch.float16
DEFAULT_SCALE = 9.0
DEFAULT_CSV_DIR = Path("outputs") / "csv"
DEFAULT_BUFFER_POOL = 8
DEFAULT_TRIALS = 3
POOL_RANDN_FP16 = "randn_fp16"
POOL_EMPTY_FP16 = "empty_fp16"
POOL_EMPTY_INT8 = "empty_int8"
HADAMARD_POOL_KINDS = {"x": POOL_RANDN_FP16}
QUANTIZE_POOL_KINDS = {"x": POOL_RANDN_FP16, "y": POOL_EMPTY_INT8}
FUSED_HADAMARD_QUANT_POOL_KINDS = {
    "fused_x": POOL_RANDN_FP16,
    "fused_y": POOL_EMPTY_INT8,
    "separate_x": POOL_RANDN_FP16,
    "separate_y": POOL_EMPTY_INT8,
    "torch_x": POOL_RANDN_FP16,
    "torch_work": POOL_EMPTY_FP16,
}


def add_common_benchmark_args(
    parser,
    *,
    default_warmup: int,
    default_repeats: int,
    default_trials: int = DEFAULT_TRIALS,
    include_scale: bool = False,
    default_scale: float = DEFAULT_SCALE,
):
    parser.add_argument(
        "--npu",
        type=normalize_npu_device,
        default=DEFAULT_DEVICE,
        help="NPU device (examples: 0, npu:0, '0', 'npu:0').",
    )
    if include_scale:
        parser.add_argument(
            "--scale",
            type=float,
            default=default_scale,
            help=f"Quantization scale multiplier (default: {default_scale}).",
        )
    parser.add_argument(
        "--warmup",
        type=int,
        default=default_warmup,
        help=f"Warmup iterations per shape (default: {default_warmup}).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=default_repeats,
        help=f"Timed iterations per shape (default: {default_repeats}).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=default_trials,
        help=(
            "Independent benchmark trials per shape; the canonical CSV duration uses "
            f"the trial median (default: {default_trials})."
        ),
    )
    parser.add_argument(
        "--batches",
        type=int,
        nargs="*",
        default=None,
        help="Optional batch-size subset. Defaults to the full benchmark grid.",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="*",
        default=None,
        help="Optional hidden-dimension subset. Defaults to the full benchmark grid.",
    )
    parser.add_argument(
        "--csv-dir",
        type=str,
        default=str(DEFAULT_CSV_DIR),
        help=f"Output CSV directory (default: {DEFAULT_CSV_DIR}).",
    )
    return parser


def validate_benchmark_args(args) -> None:
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.repeats <= 0:
        raise ValueError("--repeats must be > 0")
    if args.trials <= 0:
        raise ValueError("--trials must be > 0")
    if args.batches is not None and any(batch <= 0 for batch in args.batches):
        raise ValueError("--batches values must be > 0")
    if args.hidden_dims is not None and any(dim <= 0 for dim in args.hidden_dims):
        raise ValueError("--hidden-dims values must be > 0")


def benchmark_batches(args):
    return args.batches if args.batches else BENCH_BATCHES


def benchmark_hidden_dims(args):
    return args.hidden_dims if args.hidden_dims else BENCH_HIDDEN_DIMS


def resolve_dir_arg(base: Path, path_arg: str) -> Path:
    path = Path(path_arg)
    if not path.is_absolute():
        path = base / path
    return path


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    chmod_output_path(output_dir)


def allocation_count(
    warmup: int,
    repeats: int,
    buffer_pool: int = DEFAULT_BUFFER_POOL,
) -> int:
    return max(1, min(buffer_pool, warmup + repeats))


def pool_item(items, index):
    return items[index % len(items)]


def make_buffer_pool(
    warmup: int,
    repeats: int,
    factory,
    buffer_pool: int = DEFAULT_BUFFER_POOL,
):
    allocated = allocation_count(warmup, repeats, buffer_pool=buffer_pool)
    return [factory() for _ in range(allocated)]


def make_scale_tensor(scale: float, device: str, dtype=DTYPE):
    return torch.tensor([scale], device=device, dtype=dtype)


def torch_npu_quantize(x, scale_tensor):
    import torch_npu  # noqa

    return torch_npu.npu_quantize(x, scale_tensor, None, torch.qint8, -1, False)


def hadamard_torch_stagewise(x, work):
    n = x.shape[-1]
    n_half = n // 2
    log2_n = int(math.log2(n))
    src = x
    dst = work

    for _ in range(log2_n):
        pairs = src.reshape(x.shape[0], n_half, 2)
        torch.add(pairs[..., 0], pairs[..., 1], out=dst[:, :n_half])
        torch.sub(pairs[..., 0], pairs[..., 1], out=dst[:, n_half:])
        src, dst = dst, src
    return src


def _shape_pool_factory(kind: str, batch: int, n: int, device: str, dtype):
    if kind == POOL_RANDN_FP16:
        return lambda: torch.randn(batch, n, device=device, dtype=dtype)
    if kind == POOL_EMPTY_FP16:
        return lambda: torch.empty(batch, n, device=device, dtype=dtype)
    if kind == POOL_EMPTY_INT8:
        return lambda: torch.empty(batch, n, device=device, dtype=torch.int8)
    raise ValueError(f"Unsupported pool kind: {kind}")


def make_shape_pools(
    batch: int,
    n: int,
    warmup: int,
    repeats: int,
    *,
    device: str,
    pool_kinds,
    dtype=DTYPE,
    buffer_pool: int = DEFAULT_BUFFER_POOL,
):
    return {
        name: make_buffer_pool(
            warmup,
            repeats,
            _shape_pool_factory(kind, batch, n, device, dtype),
            buffer_pool=buffer_pool,
        )
        for name, kind in pool_kinds.items()
    }


def tensor_batch(x) -> int:
    return x.shape[0]


def tensor_n(x) -> int:
    return x.shape[1]


def tensor_log2_n(x) -> int:
    return int(math.log2(tensor_n(x)))


def run_hadamard_iteration(hadamard_func, x, *, block_dim=None):
    hadamard_func(
        x,
        tensor_batch(x),
        tensor_n(x),
        tensor_log2_n(x),
        block_dim=block_dim,
    )


def run_copy_iteration(copy_func, x, y, *, block_dim=None):
    copy_func(
        x,
        y,
        tensor_batch(x),
        tensor_n(x),
        block_dim=block_dim,
    )


def run_quantize_iteration(quantize_func, x, y, scale, *, block_dim=None):
    quantize_func(x, y, scale, block_dim=block_dim)


def run_fused_hadamard_quant_iteration(fused_func, x, y, scale, *, block_dim=None):
    fused_func(
        x,
        y,
        tensor_batch(x),
        tensor_n(x),
        tensor_log2_n(x),
        scale,
        block_dim=block_dim,
    )


def run_separate_hadamard_quant_iteration(
    hadamard_func,
    quantize_func,
    x,
    y,
    scale,
    *,
    block_dim=None,
):
    run_hadamard_iteration(hadamard_func, x, block_dim=block_dim)
    run_quantize_iteration(quantize_func, x, y, scale, block_dim=block_dim)


def benchmark_npu_us(warmup: int, repeats: int, iteration_fn) -> float:
    for i in range(warmup):
        iteration_fn(i)
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for i in range(repeats):
        iteration_fn(warmup + i)
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) / repeats * 1e3


def summarize_trial_samples(samples):
    if not samples:
        raise ValueError("expected at least one benchmark sample")

    mean_us = statistics.fmean(samples)
    median_us = statistics.median(samples)
    min_us = min(samples)
    max_us = max(samples)
    std_us = statistics.stdev(samples) if len(samples) > 1 else 0.0
    cv_pct = (std_us / mean_us * 100.0) if mean_us > 0 else 0.0
    return {
        "samples_us": tuple(samples),
        "count": len(samples),
        "mean_us": mean_us,
        "median_us": median_us,
        "min_us": min_us,
        "max_us": max_us,
        "std_us": std_us,
        "cv_pct": cv_pct,
    }


def benchmark_trials_us(trials: int, benchmark_once):
    return summarize_trial_samples([benchmark_once() for _ in range(trials)])


def benchmark_hadamard_us(
    hadamard_func, x_list, *, block_dim, warmup, repeats
) -> float:
    return benchmark_npu_us(
        warmup,
        repeats,
        lambda i: run_hadamard_iteration(
            hadamard_func,
            pool_item(x_list, i),
            block_dim=block_dim,
        ),
    )


def benchmark_copy_us(copy_func, x_list, y_list, *, block_dim, warmup, repeats) -> float:
    return benchmark_npu_us(
        warmup,
        repeats,
        lambda i: run_copy_iteration(
            copy_func,
            pool_item(x_list, i),
            pool_item(y_list, i),
            block_dim=block_dim,
        ),
    )


def benchmark_quantize_us(
    quantize_func,
    x_list,
    y_list,
    scale,
    *,
    block_dim,
    warmup,
    repeats,
) -> float:
    return benchmark_npu_us(
        warmup,
        repeats,
        lambda i: run_quantize_iteration(
            quantize_func,
            pool_item(x_list, i),
            pool_item(y_list, i),
            scale,
            block_dim=block_dim,
        ),
    )


def benchmark_torch_quantize_us(
    x_list,
    scale_tensor,
    *,
    warmup,
    repeats,
    torch_quantize_fn,
) -> float:
    return benchmark_npu_us(
        warmup,
        repeats,
        lambda i: torch_quantize_fn(pool_item(x_list, i), scale_tensor),
    )


def benchmark_torch_clone_us(x_list, *, warmup, repeats) -> float:
    outputs = [None] * len(x_list)
    return benchmark_npu_us(
        warmup,
        repeats,
        lambda i: outputs.__setitem__(
            i % len(outputs),
            pool_item(x_list, i).clone(),
        ),
    )


def benchmark_fused_hadamard_quant_us(
    fused_func,
    x_list,
    y_list,
    scale,
    *,
    block_dim,
    warmup,
    repeats,
) -> float:
    return benchmark_npu_us(
        warmup,
        repeats,
        lambda i: run_fused_hadamard_quant_iteration(
            fused_func,
            pool_item(x_list, i),
            pool_item(y_list, i),
            scale,
            block_dim=block_dim,
        ),
    )


def benchmark_separate_hadamard_quant_us(
    hadamard_func,
    quantize_func,
    x_list,
    y_list,
    scale,
    *,
    block_dim,
    warmup,
    repeats,
) -> float:
    return benchmark_npu_us(
        warmup,
        repeats,
        lambda i: run_separate_hadamard_quant_iteration(
            hadamard_func,
            quantize_func,
            pool_item(x_list, i),
            pool_item(y_list, i),
            scale,
            block_dim=block_dim,
        ),
    )


def benchmark_torch_unfused_hadamard_quant_us(
    x_list,
    work_list,
    scale_tensor,
    *,
    warmup,
    repeats,
    hadamard_stagewise_fn,
    torch_quantize_fn,
) -> float:
    return benchmark_npu_us(
        warmup,
        repeats,
        lambda i: torch_quantize_fn(
            hadamard_stagewise_fn(pool_item(x_list, i), pool_item(work_list, i)),
            scale_tensor,
        ),
    )


def bandwidth_gbs(data_bytes: int, duration_us: float) -> float:
    return (data_bytes / 1e9) / (duration_us / 1e6) if duration_us > 0 else 0.0


def summarize_fused_hadamard_quant_shape(
    batch: int,
    n: int,
    fused_us: float,
    separate_us: float,
    torch_unfused_us: float,
):
    effective_bytes = batch * n * (2 + 1)
    return {
        "batch": batch,
        "n": n,
        "fused_us": fused_us,
        "separate_us": separate_us,
        "torch_unfused_us": torch_unfused_us,
        "fused_bw": bandwidth_gbs(effective_bytes, fused_us),
        "separate_bw": bandwidth_gbs(effective_bytes, separate_us),
        "fused_speedup_vs_separate": separate_us / fused_us if fused_us > 0 else 0.0,
        "fused_speedup_vs_unfused": (
            torch_unfused_us / fused_us if fused_us > 0 else 0.0
        ),
    }


def print_fused_hadamard_quant_shape_summary(result) -> None:
    print(
        f"{result['batch']:>6d}  {result['n']:>6d}"
        f"  {result['fused_us']:>10.2f}  {result['fused_cv_pct']:>8.2f}"
        f"  {result['separate_us']:>12.2f}  {result['separate_cv_pct']:>8.2f}"
        f"  {result['torch_unfused_us']:>16.2f}  {result['torch_unfused_cv_pct']:>8.2f}"
        f"  {result['fused_bw']:>14.2f}  {result['separate_bw']:>17.2f}"
        f"  {result['fused_speedup_vs_separate']:>12.3f}"
        f"  {result['fused_speedup_vs_unfused']:>17.3f}"
    )


def format_fused_hadamard_quant_csv_record(scale: float, result) -> str:
    return (
        f"{result['batch']},{result['n']},{scale:.4f},{result['fused_us']:.4f},"
        f"{result['separate_us']:.4f},{result['torch_unfused_us']:.4f},"
        f"{result['fused_bw']:.4f},{result['separate_bw']:.4f},"
        f"{result['fused_speedup_vs_separate']:.4f},"
        f"{result['fused_speedup_vs_unfused']:.4f},{result['trials']},"
        f"{result['fused_mean_us']:.4f},{result['fused_std_us']:.4f},"
        f"{result['fused_min_us']:.4f},{result['fused_max_us']:.4f},"
        f"{result['fused_cv_pct']:.4f},{result['separate_mean_us']:.4f},"
        f"{result['separate_std_us']:.4f},{result['separate_min_us']:.4f},"
        f"{result['separate_max_us']:.4f},{result['separate_cv_pct']:.4f},"
        f"{result['torch_unfused_mean_us']:.4f},"
        f"{result['torch_unfused_std_us']:.4f},{result['torch_unfused_min_us']:.4f},"
        f"{result['torch_unfused_max_us']:.4f},{result['torch_unfused_cv_pct']:.4f}"
    )


def measure_fused_hadamard_quant_shape(
    fused_func,
    hadamard_func,
    quantize_func,
    *,
    batch,
    n,
    scale,
    scale_tensor,
    block_dim,
    warmup,
    repeats,
    trials,
    device,
    hadamard_stagewise_fn=hadamard_torch_stagewise,
    torch_quantize_fn=torch_npu_quantize,
):
    pools = make_shape_pools(
        batch,
        n,
        warmup,
        repeats,
        device=device,
        pool_kinds=FUSED_HADAMARD_QUANT_POOL_KINDS,
    )
    fused_stats = benchmark_trials_us(
        trials,
        lambda: benchmark_fused_hadamard_quant_us(
            fused_func,
            pools["fused_x"],
            pools["fused_y"],
            scale,
            block_dim=block_dim,
            warmup=warmup,
            repeats=repeats,
        ),
    )
    separate_stats = benchmark_trials_us(
        trials,
        lambda: benchmark_separate_hadamard_quant_us(
            hadamard_func,
            quantize_func,
            pools["separate_x"],
            pools["separate_y"],
            scale,
            block_dim=block_dim,
            warmup=warmup,
            repeats=repeats,
        ),
    )
    torch_stats = benchmark_trials_us(
        trials,
        lambda: benchmark_torch_unfused_hadamard_quant_us(
            pools["torch_x"],
            pools["torch_work"],
            scale_tensor,
            warmup=warmup,
            repeats=repeats,
            hadamard_stagewise_fn=hadamard_stagewise_fn,
            torch_quantize_fn=torch_quantize_fn,
        ),
    )
    result = summarize_fused_hadamard_quant_shape(
        batch,
        n,
        fused_stats["median_us"],
        separate_stats["median_us"],
        torch_stats["median_us"],
    )
    result.update(
        {
            "trials": trials,
            "fused_mean_us": fused_stats["mean_us"],
            "fused_std_us": fused_stats["std_us"],
            "fused_min_us": fused_stats["min_us"],
            "fused_max_us": fused_stats["max_us"],
            "fused_cv_pct": fused_stats["cv_pct"],
            "separate_mean_us": separate_stats["mean_us"],
            "separate_std_us": separate_stats["std_us"],
            "separate_min_us": separate_stats["min_us"],
            "separate_max_us": separate_stats["max_us"],
            "separate_cv_pct": separate_stats["cv_pct"],
            "torch_unfused_mean_us": torch_stats["mean_us"],
            "torch_unfused_std_us": torch_stats["std_us"],
            "torch_unfused_min_us": torch_stats["min_us"],
            "torch_unfused_max_us": torch_stats["max_us"],
            "torch_unfused_cv_pct": torch_stats["cv_pct"],
        }
    )
    return result


def write_csv_records(csv_path: Path, header: str, records) -> None:
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n".join(records) + "\n")
    chmod_output_path(csv_path)
