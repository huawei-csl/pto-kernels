import argparse
import os
from statistics import median

import torch
import torch_npu  # noqa: F401

from jit_util_linear_attention import BLOCK_DIM, get_causal_mask, jit_compile
from run_linear_attention import _apply_gating, _build_precomputed_h

DTYPE = torch.float16
_DEFAULT_MAX_CACHE_SIZE = 256 * 1024 * 1024

# Larger presets intended to drive better utilization while keeping H/D/C static
# within each compiled kernel.
DEFAULT_SHAPES = [
    (32, 20, 2048, 128, 128),
    (24, 20, 4096, 128, 128),
    (12, 20, 8192, 128, 128),
    (24, 20, 6144, 128, 128),
]

QUICK_SHAPES = [
    (8, 20, 1024, 128, 128),
    (16, 20, 1024, 128, 128),
]

THROUGHPUT_HUNT_SHAPES = [
    (24, 20, 2048, 128, 128),
    (48, 20, 1024, 128, 128),
    (12, 20, 8192, 128, 128),
    (24, 20, 1536, 128, 128),
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
    qkv_and_output_bytes = chunk_num * (4 * chunk * hidden * 2)
    causal_mask_bytes = chunk * chunk * 2
    return batch * heads * qkv_and_output_bytes + causal_mask_bytes


def make_inputs(batch: int, heads: int, seq: int, hidden: int):
    q = torch.randn((batch, heads, seq, hidden), device="npu", dtype=DTYPE)
    k = torch.randn((batch, heads, seq, hidden), device="npu", dtype=DTYPE)
    v = torch.randn((batch, heads, seq, hidden), device="npu", dtype=DTYPE)
    q = q / (q.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-6)
    k = k / (k.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-6)
    return q, k, v


def make_inputs_seq_first(batch: int, heads: int, seq: int, hidden: int):
    q = torch.randn((batch, seq, heads, hidden), device="npu", dtype=DTYPE)
    k = torch.randn((batch, seq, heads, hidden), device="npu", dtype=DTYPE)
    v = torch.randn((batch, seq, heads, hidden), device="npu", dtype=DTYPE)
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
    *,
    seq_first: bool = False,
    use_g: bool = False,
    varlen_uniform: bool = False,
):
    kernel = jit_compile(src, num_heads=heads, hidden_size=hidden, chunk_size=chunk)
    causal_mask = get_causal_mask(chunk, DTYPE, 0)
    cache = torch.ones(_DEFAULT_MAX_CACHE_SIZE, dtype=torch.int8, device="npu")

    if not seq_first and not use_g and not varlen_uniform:
        q, k, v = make_inputs(batch, heads, seq, hidden)
        workspace_1 = torch.zeros((BLOCK_DIM, 2, chunk, chunk), device="npu", dtype=DTYPE)
        workspace_2 = torch.zeros((BLOCK_DIM, 2, hidden, hidden), device="npu", dtype=DTYPE)
        out = torch.zeros((batch, heads, seq, hidden), device="npu", dtype=DTYPE)

        def launch():
            kernel(q, k, v, workspace_1, workspace_2, causal_mask, out, block_dim=BLOCK_DIM)

    else:
        q, k, v = make_inputs_seq_first(batch, heads, seq, hidden)
        g = torch.zeros((batch, seq, heads), device="npu", dtype=torch.float32) if use_g else None
        cu_seqlens = None
        if varlen_uniform:
            total_t = batch * seq
            cu_seqlens = torch.arange(0, total_t + 1, seq, device="npu", dtype=torch.int32)
            q = q.reshape(1, total_t, heads, hidden).contiguous()
            k = k.reshape(1, total_t, heads, hidden).contiguous()
            v = v.reshape(1, total_t, heads, hidden).contiguous()
            if g is not None:
                g = g.reshape(1, total_t, heads).contiguous()
            batch_for_kernel = batch
        else:
            batch_for_kernel = batch

        q_scaled, k_scaled = _apply_gating(q, k, g, head_first=False)
        h_states = _build_precomputed_h(
            k_scaled,
            v,
            chunk,
            head_first=False,
            cu_seqlens=cu_seqlens,
        ).contiguous()
        workspace_1 = torch.zeros((BLOCK_DIM, chunk, chunk), device="npu", dtype=DTYPE)
        out = torch.zeros_like(v)

        def launch():
            kernel(
                q_scaled,
                k_scaled,
                v,
                workspace_1,
                h_states,
                causal_mask,
                out,
                cu_seqlens=cu_seqlens,
                seq_first=True,
                use_precomputed_h=True,
                batch_size_override=batch_for_kernel,
                block_dim=BLOCK_DIM,
            )

        batch = batch_for_kernel

    for _ in range(warmup):
        launch()
    torch.npu.synchronize()

    samples_ms = []
    for _ in range(repeats):
        cache.zero_()
        torch.npu.synchronize()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        launch()
        end.record()
        end.synchronize()
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
    parser.add_argument(
        "--throughput-hunt",
        action="store_true",
        help="Run a larger-shape preset to search for higher steady-state utilization.",
    )
    parser.add_argument("--seq-first", action="store_true", help="Benchmark native (B, T, H, D) mode.")
    parser.add_argument("--use-g", action="store_true", help="Benchmark gated mode using uniform zero gate.")
    parser.add_argument(
        "--varlen-uniform",
        action="store_true",
        help="Benchmark the seq-first varlen path with uniform cu_seqlens.",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.npu.set_device("npu:0")

    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "linear_attention.cpp")
    if args.shapes:
        shapes = parse_shapes(args.shapes)
    elif args.throughput_hunt:
        shapes = THROUGHPUT_HUNT_SHAPES
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
            seq_first=args.seq_first or args.varlen_uniform,
            use_g=args.use_g,
            varlen_uniform=args.varlen_uniform,
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
