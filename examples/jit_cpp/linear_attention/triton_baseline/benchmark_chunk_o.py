import argparse
import importlib.util
import os
from pathlib import Path
from statistics import median

import torch
import torch_npu  # noqa: F401

from chunk_o import chunk_o


DTYPE = torch.float16
THIS_DIR = Path(__file__).resolve().parent
PTO_DIR = THIS_DIR.parent
PTO_SRC = PTO_DIR / "linear_attention.cpp"
PTO_UTIL = PTO_DIR / "jit_util_linear_attention.py"
_DEFAULT_MAX_CACHE_SIZE = 256 * 1024 * 1024

DEFAULT_SHAPES = [
    (8, 20, 1024, 128, 128),
    (16, 20, 1024, 128, 128),
    (24, 20, 2048, 128, 128),
]


def parse_shapes(shape_text: str):
    shapes = []
    for item in shape_text.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = tuple(int(x) for x in item.split("x"))
        if len(parts) != 5:
            raise ValueError(
                "Each shape must be formatted as BxHxLxDxC, e.g. 16x20x1024x128x128"
            )
        shapes.append(parts)
    return shapes


def estimate_flops(batch: int, heads: int, seq: int, hidden: int, chunk: int) -> int:
    chunk_num = (seq + chunk - 1) // chunk
    flops_per_chunk = 4 * chunk * hidden * (chunk + hidden)
    return batch * heads * chunk_num * flops_per_chunk


def estimate_gm_bytes(batch: int, heads: int, seq: int, hidden: int, chunk: int) -> int:
    chunk_num = (seq + chunk - 1) // chunk
    qkv_and_output_bytes = chunk_num * (4 * chunk * hidden * 2)
    return batch * heads * qkv_and_output_bytes


def make_inputs(batch: int, heads: int, seq: int, hidden: int):
    q = torch.randn((batch, heads, seq, hidden), device="npu", dtype=DTYPE)
    k = torch.randn((batch, heads, seq, hidden), device="npu", dtype=DTYPE)
    v = torch.randn((batch, heads, seq, hidden), device="npu", dtype=DTYPE)
    q = q / (q.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-6)
    k = k / (k.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-6)
    return q, k, v


def benchmark_callable(fn, warmup: int, repeats: int):
    device = torch.npu
    device.synchronize()
    for _ in range(warmup):
        fn()
    device.synchronize()

    # Match the stronger timing pattern used in local profiling helpers.
    cache = torch.ones(_DEFAULT_MAX_CACHE_SIZE, dtype=torch.int8, device="npu")

    samples_ms = []
    for _ in range(repeats):
        cache.zero_()
        device.synchronize()
        start = device.Event(enable_timing=True)
        end = device.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples_ms.append(start.elapsed_time(end))
    return median(samples_ms)


def load_pto_helpers():
    spec = importlib.util.spec_from_file_location("pto_jit_util", PTO_UTIL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def benchmark_triton_shape(
    batch: int, heads: int, seq: int, hidden: int, chunk: int, warmup: int, repeats: int
):
    q, k, v = make_inputs(batch, heads, seq, hidden)
    med_ms = benchmark_callable(lambda: chunk_o(q, k, v, chunk_size=chunk), warmup, repeats)
    return summarize_result("triton_ascend", batch, heads, seq, hidden, chunk, med_ms)


def benchmark_pto_shape(
    batch: int, heads: int, seq: int, hidden: int, chunk: int, warmup: int, repeats: int
):
    if seq % chunk != 0:
        raise ValueError("PTO benchmark path requires L to be a multiple of C.")

    pto = load_pto_helpers()
    kernel = pto.jit_compile(str(PTO_SRC), num_heads=heads, hidden_size=hidden, chunk_size=chunk)
    q, k, v = make_inputs(batch, heads, seq, hidden)
    workspace_1 = torch.zeros(
        (pto.BLOCK_DIM, 2, chunk, chunk), device="npu", dtype=DTYPE
    )
    workspace_2 = torch.zeros(
        (pto.BLOCK_DIM, 2, hidden, hidden), device="npu", dtype=DTYPE
    )
    causal_mask = pto.get_causal_mask(chunk, DTYPE, 0)
    out = torch.zeros((batch, heads, seq, hidden), device="npu", dtype=DTYPE)

    def run():
        kernel(q, k, v, workspace_1, workspace_2, causal_mask, out, block_dim=pto.BLOCK_DIM)

    med_ms = benchmark_callable(run, warmup, repeats)
    return summarize_result("pto_cpp", batch, heads, seq, hidden, chunk, med_ms)


def summarize_result(
    kernel_name: str,
    batch: int,
    heads: int,
    seq: int,
    hidden: int,
    chunk: int,
    median_ms: float,
):
    secs = median_ms / 1e3
    flops = estimate_flops(batch, heads, seq, hidden, chunk)
    gm_bytes = estimate_gm_bytes(batch, heads, seq, hidden, chunk)
    return {
        "kernel": kernel_name,
        "shape": (batch, heads, seq, hidden, chunk),
        "median_ms": median_ms,
        "tflops": flops / secs / 1e12,
        "gib_s": gm_bytes / secs / (2**30),
    }


def render_markdown(results):
    lines = [
        "# Triton-Ascend `chunk_o` Performance",
        "",
        "| Kernel | Shape `(B,H,L,D,C)` | Median ms | TFLOP/s | GiB/s |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            f"| {result['kernel']} | `{result['shape']}` | "
            f"{result['median_ms']:.3f} | {result['tflops']:.2f} | {result['gib_s']:.2f} |"
        )

    grouped = {}
    for result in results:
        grouped.setdefault(result["shape"], {})[result["kernel"]] = result

    lines.extend(["", "## Comparison", ""])
    lines.append("| Shape `(B,H,L,D,C)` | Triton / PTO speedup | Triton - PTO TFLOP/s delta |")
    lines.append("| --- | ---: | ---: |")
    for shape, pair in grouped.items():
        if "triton_ascend" not in pair or "pto_cpp" not in pair:
            continue
        triton_ms = pair["triton_ascend"]["median_ms"]
        pto_ms = pair["pto_cpp"]["median_ms"]
        speedup = pto_ms / triton_ms
        tflops_delta = pair["triton_ascend"]["tflops"] - pair["pto_cpp"]["tflops"]
        lines.append(f"| `{shape}` | {speedup:.2f}x | {tflops_delta:+.2f} |")

    lines.extend(
        [
            "",
            "Notes:",
            "- Reported TFLOP/s and GiB/s are computed from the same algorithm-level model for both kernels.",
            "- The Triton kernel is forward-only, head-first only, and currently omits gating and varlen support.",
            "- `TRITON_ALL_BLOCKS_PARALLEL` is intentionally left disabled here because it produced incorrect outputs for this kernel.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Triton-Ascend chunk_o against the PTO linear-attention kernel."
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--shapes",
        type=str,
        default="",
        help="Semicolon-separated BxHxLxDxC list, e.g. 16x20x1024x128x128",
    )
    parser.add_argument(
        "--skip-pto",
        action="store_true",
        help="Benchmark only the Triton-Ascend kernel.",
    )
    parser.add_argument(
        "--markdown-out",
        type=str,
        default="",
        help="Optional path to write a markdown summary.",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.npu.set_device("npu:0")
    shapes = parse_shapes(args.shapes) if args.shapes else DEFAULT_SHAPES

    header = f"{'kernel':>14}  {'shape (B,H,L,D,C)':>24}  {'ms':>9}  {'TFLOP/s':>10}  {'GiB/s':>10}"
    print(header)
    print("-" * len(header))

    results = []
    for shape in shapes:
        batch, heads, seq, hidden, chunk = shape
        print(f"Running Triton-Ascend {shape} ...")
        triton_result = benchmark_triton_shape(
            batch, heads, seq, hidden, chunk, args.warmup, args.repeats
        )
        results.append(triton_result)
        print(
            f"{triton_result['kernel']:>14}  {str(triton_result['shape']):>24}  "
            f"{triton_result['median_ms']:>9.3f}  {triton_result['tflops']:>10.2f}  "
            f"{triton_result['gib_s']:>10.2f}"
        )

        if not args.skip_pto:
            print(f"Running PTO C++ {shape} ...")
            pto_result = benchmark_pto_shape(
                batch, heads, seq, hidden, chunk, args.warmup, args.repeats
            )
            results.append(pto_result)
            print(
                f"{pto_result['kernel']:>14}  {str(pto_result['shape']):>24}  "
                f"{pto_result['median_ms']:>9.3f}  {pto_result['tflops']:>10.2f}  "
                f"{pto_result['gib_s']:>10.2f}"
            )

    if args.markdown_out:
        markdown = render_markdown(results)
        output_path = Path(args.markdown_out)
        if not output_path.is_absolute():
            output_path = THIS_DIR / output_path
        output_path.write_text(markdown)
        print(f"\nWrote markdown summary to {output_path}")


if __name__ == "__main__":
    main()
