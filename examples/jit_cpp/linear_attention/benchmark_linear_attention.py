import argparse
import os
from statistics import median

import torch
import torch_npu  # noqa: F401

from jit_util_linear_attention import BLOCK_DIM, get_causal_mask, jit_compile

DTYPE = torch.float16

# Larger presets intended to drive better utilization while keeping H/D/C static
# within each compiled kernel.
DEFAULT_SHAPES = [
    (16, 20, 1024, 128, 64),
    (16, 20, 2048, 128, 64),
    (32, 20, 1024, 128, 64),
    (8, 20, 4096, 128, 64),
]

QUICK_SHAPES = [
    (8, 20, 1024, 128, 64),
    (16, 20, 1024, 128, 64),
]


def parse_shapes(shape_text: str):
    shapes = []
    for item in shape_text.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [int(x) for x in item.split("x")]
        if len(parts) != 5:
            raise ValueError(
                "Each shape must be formatted as BxHxLxDxC, e.g. 16x20x1024x128x64"
            )
        shapes.append(tuple(parts))
    return shapes


def estimate_flops(batch: int, heads: int, seq: int, hidden: int, chunk: int) -> int:
    if seq % chunk != 0:
        raise ValueError("This benchmark requires L to be a multiple of C.")
    chunk_num = seq // chunk
    flops_per_chunk = 4 * chunk * hidden * (chunk + hidden)
    return batch * heads * chunk_num * flops_per_chunk


def estimate_gm_bytes(batch: int, heads: int, seq: int, hidden: int, chunk: int) -> int:
    if seq % chunk != 0:
        raise ValueError("This benchmark requires L to be a multiple of C.")
    chunk_num = seq // chunk
    workspace_init_bytes = 2 * hidden * hidden
    bytes_per_chunk = 8 * chunk * hidden + 8 * chunk * chunk + 8 * hidden * hidden
    return batch * heads * (workspace_init_bytes + chunk_num * bytes_per_chunk)


def make_inputs(batch: int, heads: int, seq: int, hidden: int):
    q = torch.randn((batch, heads, seq, hidden), device="npu", dtype=DTYPE)
    k = torch.randn((batch, heads, seq, hidden), device="npu", dtype=DTYPE)
    v = torch.randn((batch, heads, seq, hidden), device="npu", dtype=DTYPE)
    q = q / (q.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-6)
    k = k / (k.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-6)
    return q, k, v


def benchmark_shape(
    src: str,
    batch: int,
    heads: int,
    seq: int,
    hidden: int,
    chunk: int,
    warmup: int,
    repeats: int,
):
    kernel = jit_compile(src, num_heads=heads, hidden_size=hidden, chunk_size=chunk)
    q, k, v = make_inputs(batch, heads, seq, hidden)
    workspace_1 = torch.zeros((BLOCK_DIM, chunk, chunk), device="npu", dtype=DTYPE)
    workspace_2 = torch.zeros((BLOCK_DIM, hidden, hidden), device="npu", dtype=DTYPE)
    causal_mask = get_causal_mask(chunk, DTYPE, 0)
    out = torch.zeros((batch, heads, seq, hidden), device="npu", dtype=DTYPE)

    for _ in range(warmup):
        kernel(
            q, k, v, workspace_1, workspace_2, causal_mask, out, block_dim=BLOCK_DIM
        )
    torch.npu.synchronize()

    samples_ms = []
    for _ in range(repeats):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        kernel(
            q, k, v, workspace_1, workspace_2, causal_mask, out, block_dim=BLOCK_DIM
        )
        end.record()
        torch.npu.synchronize()
        samples_ms.append(start.elapsed_time(end))

    med_ms = median(samples_ms)
    secs = med_ms / 1e3
    flops = estimate_flops(batch, heads, seq, hidden, chunk)
    gm_bytes = estimate_gm_bytes(batch, heads, seq, hidden, chunk)
    tflops = flops / secs / 1e12
    gib_s = gm_bytes / secs / (2**30)

    return {
        "shape": (batch, heads, seq, hidden, chunk),
        "median_ms": med_ms,
        "tflops": tflops,
        "gib_s": gib_s,
        "flops": flops,
        "gm_bytes": gm_bytes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the standalone PTO-ISA linear attention kernel."
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--shapes",
        type=str,
        default="",
        help="Semicolon-separated BxHxLxDxC list, e.g. 16x20x1024x128x64;8x20x4096x128x64",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a shorter preset shape list.",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.npu.set_device("npu:0")

    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "linear_attention.cpp")
    if args.shapes:
        shapes = parse_shapes(args.shapes)
    elif args.quick:
        shapes = QUICK_SHAPES
    else:
        shapes = DEFAULT_SHAPES

    header = (
        f"{'shape (B,H,L,D,C)':>24}  {'ms':>9}  {'TFLOP/s':>10}  {'GiB/s':>10}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for batch, heads, seq, hidden, chunk in shapes:
        print(f"Running {batch}x{heads}x{seq}x{hidden}x{chunk} ...")
        result = benchmark_shape(
            src,
            batch=batch,
            heads=heads,
            seq=seq,
            hidden=hidden,
            chunk=chunk,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        results.append(result)
        print(
            f"{str(result['shape']):>24}  "
            f"{result['median_ms']:>9.3f}  "
            f"{result['tflops']:>10.2f}  "
            f"{result['gib_s']:>10.2f}"
        )

    if results:
        best_tflops = max(results, key=lambda x: x["tflops"])
        best_bw = max(results, key=lambda x: x["gib_s"])
        print("\nBest throughput:")
        print(
            f"  TFLOP/s: {best_tflops['tflops']:.2f} at shape {best_tflops['shape']}"
        )
        print(f"  GiB/s:   {best_bw['gib_s']:.2f} at shape {best_bw['shape']}")


if __name__ == "__main__":
    main()
