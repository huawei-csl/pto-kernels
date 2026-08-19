import argparse
import os
from statistics import median

import torch
import torch_npu  # noqa: F401

DTYPE = torch.float16
RTOL = 1e-2


def kernel_src_path(script_file: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(script_file)), "linear_attention.cpp")


def ref_linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    batch, heads, seq_len, hidden = q.shape
    q = q.float()
    k = k.float()
    v = v.float()

    state = torch.zeros((batch, heads, hidden, hidden), device=q.device, dtype=torch.float32)
    output = torch.zeros((batch, heads, seq_len, hidden), device=q.device, dtype=torch.float32)

    for index in range(seq_len):
        q_t = q[:, :, index, :]
        k_t = k[:, :, index, :]
        v_t = v[:, :, index, :]
        state = state + torch.einsum("bhi,bhj->bhij", k_t, v_t)
        output[:, :, index, :] = torch.einsum("bhi,bhij->bhj", q_t, state)

    return output.to(DTYPE)


def make_inputs(batch: int, heads: int, seq_len: int, hidden: int):
    q = torch.randn((batch, heads, seq_len, hidden), device="npu", dtype=DTYPE)
    k = torch.randn((batch, heads, seq_len, hidden), device="npu", dtype=DTYPE)
    v = torch.randn((batch, heads, seq_len, hidden), device="npu", dtype=DTYPE)
    q = q / (q.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-6)
    k = k / (k.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-6)
    return q, k, v


def atol_for_seq(seq_len: int) -> float:
    if seq_len >= 4096:
        return 4e-2
    if seq_len >= 2048:
        return 2e-2
    return 1e-2


def validate_output(output: torch.Tensor, reference: torch.Tensor, seq_len: int):
    torch.testing.assert_close(
        output.cpu(),
        reference.cpu(),
        rtol=RTOL,
        atol=atol_for_seq(seq_len),
    )


def _workspace_shape(block_dim: int, stage_count: int, rows: int, cols: int):
    if stage_count == 1:
        return (block_dim, rows, cols)
    return (block_dim, stage_count, rows, cols)


def run_dynamic_kernel(
    src: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int,
    *,
    jit_compile,
    block_dim: int,
    stage_count: int,
    use_mask: bool,
    mask_factory=None,
):
    batch, heads, seq_len, hidden = q.shape
    if seq_len % chunk_size != 0:
        raise ValueError("This PTO-ISA example currently requires L to be a multiple of C.")

    kernel = jit_compile(src, num_heads=heads, hidden_size=hidden, chunk_size=chunk_size)
    workspace_1 = torch.zeros(
        _workspace_shape(block_dim, stage_count, chunk_size, chunk_size),
        device=q.device,
        dtype=DTYPE,
    )
    workspace_2 = torch.zeros(
        _workspace_shape(block_dim, stage_count, hidden, hidden),
        device=q.device,
        dtype=DTYPE,
    )
    output = torch.zeros((batch, heads, seq_len, hidden), device=q.device, dtype=DTYPE)

    if use_mask:
        causal_mask = mask_factory(chunk_size, DTYPE, q.device.index or 0)
        kernel(
            q,
            k,
            v,
            workspace_1,
            workspace_2,
            causal_mask,
            output,
            block_dim=block_dim,
        )
    else:
        kernel(q, k, v, workspace_1, workspace_2, output, block_dim=block_dim)

    torch.npu.synchronize()
    return output


def run_correctness_cases(script_file: str, test_configs, run_kernel):
    torch.manual_seed(0)
    torch.npu.set_device("npu:0")
    src = kernel_src_path(script_file)

    for batch, heads, seq_len, hidden, chunk in test_configs:
        print(f"Testing B={batch}, H={heads}, L={seq_len}, D={hidden}, C={chunk}  (B*H={batch * heads})")
        q, k, v = make_inputs(batch, heads, seq_len, hidden)
        output = run_kernel(src, q, k, v, chunk)
        reference = ref_linear_attention(q, k, v)
        validate_output(output, reference, seq_len)
        print("  passed!")

    print("Kernel Output Match!")


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


def estimate_flops(batch: int, heads: int, seq_len: int, hidden: int, chunk: int) -> int:
    if seq_len % chunk != 0:
        raise ValueError("This benchmark requires L to be a multiple of C.")
    chunk_count = seq_len // chunk
    flops_per_chunk = 4 * chunk * hidden * (chunk + hidden)
    return batch * heads * chunk_count * flops_per_chunk


def estimate_gm_bytes(
    batch: int,
    heads: int,
    seq_len: int,
    hidden: int,
    chunk: int,
    *,
    include_workspace: bool,
    include_mask: bool,
) -> int:
    if seq_len % chunk != 0:
        raise ValueError("This benchmark requires L to be a multiple of C.")
    chunk_count = seq_len // chunk
    if include_workspace:
        workspace_init_bytes = 2 * hidden * hidden
        bytes_per_chunk = 8 * chunk * hidden + 8 * chunk * chunk + 8 * hidden * hidden
        return batch * heads * (workspace_init_bytes + chunk_count * bytes_per_chunk)

    qkv_and_output_bytes = chunk_count * (4 * chunk * hidden * 2)
    mask_bytes = chunk * chunk * 2 if include_mask else 0
    return batch * heads * qkv_and_output_bytes + mask_bytes


def measure_kernel_ms(run_once, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        run_once()
    torch.npu.synchronize()

    samples_ms = []
    for _ in range(repeats):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        run_once()
        end.record()
        torch.npu.synchronize()
        samples_ms.append(start.elapsed_time(end))
    return median(samples_ms)


def benchmark_dynamic_kernel(
    src: str,
    *,
    batch: int,
    heads: int,
    seq_len: int,
    hidden: int,
    chunk: int,
    warmup: int,
    repeats: int,
    jit_compile,
    block_dim: int,
    stage_count: int,
    use_mask: bool,
    include_workspace_bytes: bool,
    mask_factory=None,
):
    kernel = jit_compile(src, num_heads=heads, hidden_size=hidden, chunk_size=chunk)
    q, k, v = make_inputs(batch, heads, seq_len, hidden)
    workspace_1 = torch.zeros(
        _workspace_shape(block_dim, stage_count, chunk, chunk),
        device="npu",
        dtype=DTYPE,
    )
    workspace_2 = torch.zeros(
        _workspace_shape(block_dim, stage_count, hidden, hidden),
        device="npu",
        dtype=DTYPE,
    )
    output = torch.zeros((batch, heads, seq_len, hidden), device="npu", dtype=DTYPE)
    causal_mask = mask_factory(chunk, DTYPE, 0) if use_mask else None

    def run_once():
        if use_mask:
            kernel(
                q,
                k,
                v,
                workspace_1,
                workspace_2,
                causal_mask,
                output,
                block_dim=block_dim,
            )
        else:
            kernel(q, k, v, workspace_1, workspace_2, output, block_dim=block_dim)

    median_ms = measure_kernel_ms(run_once, warmup=warmup, repeats=repeats)
    seconds = median_ms / 1e3
    flops = estimate_flops(batch, heads, seq_len, hidden, chunk)
    gm_bytes = estimate_gm_bytes(
        batch,
        heads,
        seq_len,
        hidden,
        chunk,
        include_workspace=include_workspace_bytes,
        include_mask=use_mask,
    )
    return {
        "shape": (batch, heads, seq_len, hidden, chunk),
        "median_ms": median_ms,
        "tflops": flops / seconds / 1e12,
        "gib_s": gm_bytes / seconds / (2**30),
        "flops": flops,
        "gm_bytes": gm_bytes,
    }


def benchmark_cli(
    *,
    script_file: str,
    default_shapes,
    quick_shapes,
    benchmark_shape,
    throughput_hunt_shapes=None,
    description: str = "Benchmark the standalone PTO-ISA linear attention kernel.",
):
    parser = argparse.ArgumentParser(description=description)
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
    if throughput_hunt_shapes is not None:
        parser.add_argument(
            "--throughput-hunt",
            action="store_true",
            help="Run a larger-shape preset to search for higher steady-state utilization.",
        )
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.npu.set_device("npu:0")
    src = kernel_src_path(script_file)

    if args.shapes:
        shapes = parse_shapes(args.shapes)
    elif throughput_hunt_shapes is not None and args.throughput_hunt:
        shapes = throughput_hunt_shapes
    elif args.quick:
        shapes = quick_shapes
    else:
        shapes = default_shapes

    header = f"{'shape (B,H,L,D,C)':>24}  {'ms':>9}  {'TFLOP/s':>10}  {'GiB/s':>10}"
    print(header)
    print("-" * len(header))

    results = []
    for batch, heads, seq_len, hidden, chunk in shapes:
        print(f"Running {batch}x{heads}x{seq_len}x{hidden}x{chunk} ...")
        result = benchmark_shape(
            src,
            batch=batch,
            heads=heads,
            seq_len=seq_len,
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
        print(f"  TFLOP/s: {best_tflops['tflops']:.2f} at shape {best_tflops['shape']}")
        print(f"  GiB/s:   {best_bw['gib_s']:.2f} at shape {best_bw['shape']}")
