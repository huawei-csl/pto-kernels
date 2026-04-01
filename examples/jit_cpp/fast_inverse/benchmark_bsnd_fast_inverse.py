#!/usr/bin/env python3
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------

"""
Benchmark the standalone BSND fast-inverse kernel.

This script benchmarks the PTO-ISA BSND kernel in two modes using Triton-unit-
test-like inputs:

1. `bsnd-fixed`:
   Original aligned BSND layout with shape `(B, T, H, D)`.
2. `bsnd-varlen-uniform`:
   The new varlen path using packed shape `(1, B*T, H, D)` with uniform
   `cu_seqlens = [0, T, 2T, ...]`.

The two modes use the same total token count and the same underlying `k` / `beta`
inputs. `A` is generated in eager PyTorch with an emulation of
`chunk_scaled_dot_kkt_fwd`, then each valid chunk is transposed before launch so
the PTO kernel still sees its expected upper-triangular layout. The script also
checks that both modes produce numerically matching results after transposing
outputs back to the lower-triangular convention used by the Triton tests.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from host_metadata_util import (
    build_chunk_sequence_prefix_cpp,
    build_varlen_chunk_metadata_cpp,
)
from jit_util_fast_inverse import jit_compile


DEFAULT_SEQLENS = (512, 1024, 2048, 4096, 8192, 16384)
DEFAULT_CACHE_SIZE = 256 * 1024 * 1024
DEFAULT_FEATURE_DIM = 64
NPU_DEVICE = os.getenv("GDN_TRI_INVERSE_NPU_DEVICE", "npu:0")
THIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = THIS_DIR / "benchmark_results"
DEFAULT_TRUE_VARLEN_SAMPLES = 6


def parse_int_list(spec: str) -> tuple[int, ...]:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected at least one integer")
    try:
        return tuple(int(p, 10) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer list {spec!r}: {exc}") from exc


def make_minus_identity(matrix_size: int, device: str) -> torch.Tensor:
    minus_identity = torch.zeros(
        matrix_size,
        matrix_size,
        dtype=torch.half,
        device=device,
    )
    minus_identity.fill_diagonal_(-1)
    return minus_identity


def count_varlen_chunks(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
) -> int:
    cu_seqlens_list = [int(x) for x in cu_seqlens.detach().cpu().tolist()]
    return sum(
        (cu_seqlens_list[i + 1] - cu_seqlens_list[i] + chunk_size - 1) // chunk_size
        for i in range(len(cu_seqlens_list) - 1)
    )


def chunk_scaled_dot_kkt_fwd_emulated(
    k: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    total_tokens = int(cu_seqlens[-1].item())
    num_heads = k.shape[2]
    A = torch.zeros((1, total_tokens, num_heads, chunk_size), dtype=k.dtype, device=k.device)

    for bos, eos in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=False):
        for chunk_start in range(bos, eos, chunk_size):
            chunk_end = min(chunk_start + chunk_size, eos)
            actual_size = chunk_end - chunk_start
            k_chunk = k[:, chunk_start:chunk_end].transpose(1, 2).to(torch.float32)
            beta_chunk = (
                beta[:, chunk_start:chunk_end]
                .transpose(1, 2)
                .unsqueeze(-1)
                .to(torch.float32)
            )
            scores = torch.matmul(k_chunk, k_chunk.transpose(-1, -2))
            scores = torch.tril(scores * beta_chunk, diagonal=-1).to(k.dtype)
            A[:, chunk_start:chunk_end, :, :actual_size] = scores.transpose(1, 2)

    return A


def transpose_valid_chunks(
    A: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    transposed = torch.zeros_like(A)
    for bos, eos in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=False):
        for chunk_start in range(bos, eos, chunk_size):
            actual_size = min(chunk_size, eos - chunk_start)
            chunk = A[:, chunk_start : chunk_start + actual_size, :, :actual_size]
            transposed[:, chunk_start : chunk_start + actual_size, :, :actual_size] = chunk.transpose(
                1, 3
            )
    return transposed


def build_fixed_bsnd_input(
    batch_size: int,
    seqlen: int,
    num_heads: int,
    chunk_size: int,
    feature_dim: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_tokens = batch_size * seqlen
    cu_seqlens = torch.arange(
        0,
        total_tokens + 1,
        seqlen,
        dtype=torch.int32,
        device=device,
    )
    k = F.normalize(
        torch.randn((1, total_tokens, num_heads, feature_dim), dtype=torch.float16, device=device),
        dim=-1,
    )
    beta = torch.randn((1, total_tokens, num_heads), dtype=torch.float16, device=device).sigmoid()
    A = transpose_valid_chunks(
        chunk_scaled_dot_kkt_fwd_emulated(k, beta, cu_seqlens, chunk_size),
        cu_seqlens,
        chunk_size,
    )
    return A.reshape(batch_size, seqlen, num_heads, chunk_size).contiguous(), cu_seqlens


def build_uniform_varlen_input(
    fixed_input: torch.Tensor,
    batch_size: int,
    seqlen: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_tokens = batch_size * seqlen
    packed_input = fixed_input.reshape(1, total_tokens, fixed_input.shape[2], chunk_size).contiguous()
    cu_seqlens = torch.arange(
        0,
        total_tokens + 1,
        seqlen,
        dtype=torch.int32,
        device=fixed_input.device,
    )
    return packed_input, cu_seqlens


def sample_true_varlen_lengths(
    batch_size: int,
    aggregated_tokens: int,
    rng: np.random.Generator,
) -> list[int]:
    if aggregated_tokens < batch_size:
        raise ValueError("aggregated_tokens must be >= batch_size.")

    remaining = aggregated_tokens - batch_size
    while True:
        weights = rng.dirichlet(np.ones(batch_size))
        extras = np.floor(weights * remaining).astype(np.int64)
        deficit = remaining - int(extras.sum())
        if deficit > 0:
            extras[:deficit] += 1
        lengths = (extras + 1).tolist()
        if any(length != lengths[0] for length in lengths):
            return lengths


def build_true_varlen_input(
    seq_lens: list[int],
    num_heads: int,
    chunk_size: int,
    feature_dim: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    cu_seqlens = np.cumsum([0, *seq_lens], dtype=np.int64)
    cu_seqlens_tensor = torch.tensor(cu_seqlens.tolist(), dtype=torch.int32, device=device)
    total_tokens = int(cu_seqlens[-1])
    k = F.normalize(
        torch.randn((1, total_tokens, num_heads, feature_dim), dtype=torch.float16, device=device),
        dim=-1,
    )
    beta = torch.randn((1, total_tokens, num_heads), dtype=torch.float16, device=device).sigmoid()
    packed_input = transpose_valid_chunks(
        chunk_scaled_dot_kkt_fwd_emulated(k, beta, cu_seqlens_tensor, chunk_size),
        cu_seqlens_tensor,
        chunk_size,
    )
    return packed_input.contiguous(), cu_seqlens_tensor


def make_fixed_runner(
    tri_inv_func,
    tensor_in: torch.Tensor,
) -> tuple[callable, torch.Tensor]:
    matrix_size = tensor_in.shape[-1]
    num_bsnd_heads = tensor_in.shape[-2]
    num_matrices = tensor_in.numel() // (matrix_size * matrix_size)
    tensor_out = torch.empty_like(tensor_in, dtype=torch.float32)
    minus_identity = make_minus_identity(matrix_size, str(tensor_in.device))

    def run():
        tri_inv_func(
            tensor_out,
            tensor_in,
            minus_identity,
            matrix_size,
            num_matrices,
            num_bsnd_heads,
        )

    return run, tensor_out


def make_varlen_runner(
    tri_inv_func,
    tensor_in: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[callable, torch.Tensor]:
    matrix_size = tensor_in.shape[-1]
    num_bsnd_heads = tensor_in.shape[-2]
    num_matrices = count_varlen_chunks(cu_seqlens, matrix_size) * num_bsnd_heads
    tensor_out = torch.empty_like(tensor_in, dtype=torch.float32)
    minus_identity = make_minus_identity(matrix_size, str(tensor_in.device))

    def run():
        tri_inv_func(
            tensor_out,
            tensor_in,
            minus_identity,
            matrix_size,
            num_matrices,
            num_bsnd_heads,
            cu_seqlens=cu_seqlens,
        )

    return run, tensor_out


def make_varlen_runner_host_metadata(
    tri_inv_func,
    tensor_in: torch.Tensor,
    chunk_indices: torch.Tensor,
    chunk_valid_sizes: torch.Tensor,
) -> tuple[callable, torch.Tensor]:
    matrix_size = tensor_in.shape[-1]
    num_bsnd_heads = tensor_in.shape[-2]
    num_matrices = int(chunk_indices.numel()) * num_bsnd_heads
    tensor_out = torch.empty_like(tensor_in, dtype=torch.float32)
    minus_identity = make_minus_identity(matrix_size, str(tensor_in.device))

    def run():
        tri_inv_func(
            tensor_out,
            tensor_in,
            minus_identity,
            matrix_size,
            num_matrices,
            num_bsnd_heads,
            chunk_indices=chunk_indices,
            chunk_valid_sizes=chunk_valid_sizes,
        )

    return run, tensor_out


def make_varlen_runner_prefix_metadata(
    tri_inv_func,
    tensor_in: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_sequence_prefix: torch.Tensor,
) -> tuple[callable, torch.Tensor]:
    matrix_size = tensor_in.shape[-1]
    num_bsnd_heads = tensor_in.shape[-2]
    num_matrices = count_varlen_chunks(cu_seqlens, matrix_size) * num_bsnd_heads
    tensor_out = torch.empty_like(tensor_in, dtype=torch.float32)
    minus_identity = make_minus_identity(matrix_size, str(tensor_in.device))

    def run():
        tri_inv_func(
            tensor_out,
            tensor_in,
            minus_identity,
            matrix_size,
            num_matrices,
            num_bsnd_heads,
            cu_seqlens=cu_seqlens,
            chunk_sequence_prefix=chunk_sequence_prefix,
        )

    return run, tensor_out


def benchmark_ms(
    fn,
    warmup_iters: int,
    benchmark_iters: int,
    device: str,
) -> list[float]:
    start_events = [torch.npu.Event(enable_timing=True) for _ in range(benchmark_iters)]
    end_events = [torch.npu.Event(enable_timing=True) for _ in range(benchmark_iters)]

    torch.npu.synchronize()
    for _ in range(warmup_iters):
        fn()
    torch.npu.synchronize()

    cache = torch.ones(DEFAULT_CACHE_SIZE, dtype=torch.int8, device=device)
    times_ms: list[float] = []
    for idx in range(benchmark_iters):
        cache.zero_()
        torch.npu.synchronize()
        start_events[idx].record()
        fn()
        end_events[idx].record()
        end_events[idx].synchronize()
        times_ms.append(start_events[idx].elapsed_time(end_events[idx]))
    return times_ms


def build_host_metadata_on_npu(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    chunk_indices_cpu, chunk_valid_sizes_cpu = build_varlen_chunk_metadata_cpp(
        cu_seqlens,
        chunk_size,
    )
    return (
        chunk_indices_cpu.to(device=device).contiguous(),
        chunk_valid_sizes_cpu.to(device=device).contiguous(),
    )


def build_prefix_metadata_on_npu(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
    device: str,
) -> torch.Tensor:
    return build_chunk_sequence_prefix_cpp(cu_seqlens, chunk_size).to(
        device=device
    ).contiguous()


def benchmark_host_metadata_prep_ms(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
    benchmark_iters: int,
    device: str,
) -> list[float]:
    times_ms: list[float] = []
    cache = torch.ones(DEFAULT_CACHE_SIZE, dtype=torch.int8, device=device)
    for _ in range(benchmark_iters):
        cache.zero_()
        torch.npu.synchronize()
        start = time.perf_counter()
        build_host_metadata_on_npu(cu_seqlens, chunk_size, device)
        torch.npu.synchronize()
        times_ms.append((time.perf_counter() - start) * 1000.0)
    return times_ms


def benchmark_prefix_metadata_prep_ms(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
    benchmark_iters: int,
    device: str,
) -> list[float]:
    times_ms: list[float] = []
    cache = torch.ones(DEFAULT_CACHE_SIZE, dtype=torch.int8, device=device)
    for _ in range(benchmark_iters):
        cache.zero_()
        torch.npu.synchronize()
        start = time.perf_counter()
        build_prefix_metadata_on_npu(cu_seqlens, chunk_size, device)
        torch.npu.synchronize()
        times_ms.append((time.perf_counter() - start) * 1000.0)
    return times_ms


def add_bandwidth_fields(row: dict[str, float | int | str], input_dtype_bytes: int = 2) -> None:
    size_elems = int(row.get("valid_numel", row["numel"]))
    mem_bytes = size_elems * (input_dtype_bytes + 4)
    row["mem_bytes"] = mem_bytes
    row["bw_gbs"] = (mem_bytes / 1e9) / (float(row["time_us"]) / 1e6)


def accuracy_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[float, float]:
    ref = reference.detach().cpu().to(torch.float64)
    cand = candidate.detach().cpu().to(torch.float64)
    diff = ref - cand
    max_abs = diff.abs().max().item()
    denom = torch.sum(ref * ref).item()
    rel_frob = 0.0 if denom == 0 else math.sqrt(torch.sum(diff * diff).item() / denom)
    return max_abs, rel_frob


def write_csv(csv_path: Path, rows: list[dict[str, float | int | str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "inverse_type",
        "metadata_strategy",
        "dtype",
        "B",
        "T",
        "aggregated_T",
        "padded_T",
        "H",
        "numel",
        "valid_numel",
        "chunk_size",
        "time_us",
        "kernel_time_us",
        "metadata_time_us",
        "mem_bytes",
        "bw_gbs",
        "max_abs_diff_to_fixed",
        "rel_frob_diff_to_fixed",
        "sample_id",
        "seq_lens",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_bandwidth(plot_path: Path, rows: list[dict[str, float | int | str]], batch_size: int, num_heads: int, chunk_size: int) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fixed_rows = [row for row in rows if row["inverse_type"] == "bsnd-fixed"]
    varlen_device_rows = [
        row
        for row in rows
        if row["inverse_type"] == "bsnd-varlen-uniform"
        and row["metadata_strategy"] == "device-cu_seqlens"
    ]
    varlen_host_rows = [
        row
        for row in rows
        if row["inverse_type"] == "bsnd-varlen-uniform"
        and row["metadata_strategy"] == "host-cpp"
    ]
    varlen_prefix_rows = [
        row
        for row in rows
        if row["inverse_type"] == "bsnd-varlen-uniform"
        and row["metadata_strategy"] == "device-chunk-prefix"
    ]

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(
        [int(row["T"]) / 1000.0 for row in fixed_rows],
        [float(row["bw_gbs"]) for row in fixed_rows],
        marker="o",
        linewidth=2,
        label="BSND fixed",
    )
    ax.plot(
        [int(row["T"]) / 1000.0 for row in varlen_device_rows],
        [float(row["bw_gbs"]) for row in varlen_device_rows],
        marker="s",
        linewidth=2,
        label="BSND varlen device metadata",
    )
    ax.plot(
        [int(row["T"]) / 1000.0 for row in varlen_host_rows],
        [float(row["bw_gbs"]) for row in varlen_host_rows],
        marker="^",
        linewidth=2,
        label="BSND varlen host metadata",
    )
    ax.plot(
        [int(row["T"]) / 1000.0 for row in varlen_prefix_rows],
        [float(row["bw_gbs"]) for row in varlen_prefix_rows],
        marker="d",
        linewidth=2,
        label="BSND varlen prefix metadata",
    )
    ax.set_xlabel("Sequence length T (K)")
    ax.set_ylabel("Effective bandwidth (GB/s)")
    ax.set_title(
        f"Fast inverse BSND bandwidth\n"
        f"(batch={batch_size}, head={num_heads}, chunk_size={chunk_size})"
    )
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def plot_true_varlen_scatter(
    plot_path: Path,
    rows: list[dict[str, float | int | str]],
    batch_size: int,
    num_heads: int,
    chunk_size: int,
) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.scatter(
        [int(row["aggregated_T"]) for row in rows],
        [float(row["bw_gbs"]) for row in rows],
        alpha=0.8,
        s=32,
    )
    ax.set_xlabel("Aggregated sequence length")
    ax.set_ylabel("Effective bandwidth (GB/s)")
    ax.set_title(
        f"Fast inverse true-varlen BSND bandwidth\n"
        f"(batch={batch_size}, head={num_heads}, chunk_size={chunk_size})"
    )
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark standalone BSND fast-inverse kernel.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--B", type=int, default=32, help="Dense BSND batch size.")
    parser.add_argument("--H", type=int, default=4, help="Number of BSND heads.")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=DEFAULT_FEATURE_DIM,
        help="Feature dimension used to generate Triton-like `k` inputs.",
    )
    parser.add_argument(
        "--seqlens",
        type=parse_int_list,
        default=DEFAULT_SEQLENS,
        metavar="T[,T,...]",
        help=(
            "Comma-separated dense per-sequence lengths to benchmark "
            f"(default: {','.join(map(str, DEFAULT_SEQLENS))})"
        ),
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="",
        help="Optional CSV output path. Defaults to bench_results_bsnd_fast_inverse_<chunk>.csv",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="",
        help="Optional plot output path. Defaults to bench_results_bsnd_fast_inverse_bw_<chunk>.png",
    )
    parser.add_argument(
        "--true-varlen-csv",
        type=str,
        default="",
        help="Optional CSV path for true-varlen benchmark points.",
    )
    parser.add_argument(
        "--true-varlen-plot",
        type=str,
        default="",
        help="Optional scatter plot path for true-varlen benchmark points.",
    )
    parser.add_argument(
        "--true-varlen-samples",
        type=int,
        default=DEFAULT_TRUE_VARLEN_SAMPLES,
        help="Number of random true-varlen batches per aggregated sequence length.",
    )
    args = parser.parse_args()

    torch.npu.set_device(NPU_DEVICE)

    src = THIS_DIR / "fast_inverse.cpp"
    print(f"Compiling {src} ...")
    tri_inv_func = jit_compile(str(src))
    print("Compilation successful.\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = (
        Path(args.csv)
        if args.csv
        else RESULTS_DIR / f"bench_results_bsnd_fast_inverse_{args.chunk_size}.csv"
    )
    plot_path = (
        Path(args.plot)
        if args.plot
        else RESULTS_DIR / f"bench_results_bsnd_fast_inverse_bw_{args.chunk_size}.png"
    )
    true_varlen_csv_path = (
        Path(args.true_varlen_csv)
        if args.true_varlen_csv
        else RESULTS_DIR / f"bench_results_bsnd_fast_inverse_true_varlen_{args.chunk_size}.csv"
    )
    true_varlen_plot_path = (
        Path(args.true_varlen_plot)
        if args.true_varlen_plot
        else RESULTS_DIR / f"bench_results_bsnd_fast_inverse_true_varlen_bw_{args.chunk_size}.png"
    )

    rows: list[dict[str, float | int | str]] = []
    true_varlen_rows: list[dict[str, float | int | str]] = []
    rng = np.random.default_rng(42)

    for seqlen in args.seqlens:
        if seqlen % args.chunk_size != 0:
            print(
                f"Skipping T={seqlen}: requires T to be a multiple of chunk_size={args.chunk_size} "
                "for matched fixed vs uniform-varlen comparison."
            )
            continue

        total_tokens = args.B * seqlen
        print(
            f"Profiling T={seqlen}, total_tokens={total_tokens}, "
            f"B={args.B}, H={args.H}, chunk_size={args.chunk_size}, feature_dim={args.feature_dim}"
        )

        fixed_input, uniform_cu_seqlens = build_fixed_bsnd_input(
            batch_size=args.B,
            seqlen=seqlen,
            num_heads=args.H,
            chunk_size=args.chunk_size,
            feature_dim=args.feature_dim,
            device=NPU_DEVICE,
        )
        varlen_input, cu_seqlens = build_uniform_varlen_input(
            fixed_input,
            batch_size=args.B,
            seqlen=seqlen,
            chunk_size=args.chunk_size,
        )
        cu_seqlens_cpu = cu_seqlens.cpu()

        print(f"  uniform cu_seqlens: {cu_seqlens.cpu().tolist()}")

        fixed_run, fixed_out = make_fixed_runner(tri_inv_func, fixed_input)
        varlen_run_device, varlen_out_device = make_varlen_runner(
            tri_inv_func,
            varlen_input,
            cu_seqlens,
        )
        chunk_sequence_prefix = build_prefix_metadata_on_npu(
            cu_seqlens_cpu,
            args.chunk_size,
            NPU_DEVICE,
        )
        varlen_run_prefix, varlen_out_prefix = make_varlen_runner_prefix_metadata(
            tri_inv_func,
            varlen_input,
            cu_seqlens,
            chunk_sequence_prefix,
        )
        chunk_indices, chunk_valid_sizes = build_host_metadata_on_npu(
            cu_seqlens_cpu,
            args.chunk_size,
            NPU_DEVICE,
        )
        varlen_run_host, varlen_out_host = make_varlen_runner_host_metadata(
            tri_inv_func,
            varlen_input,
            chunk_indices,
            chunk_valid_sizes,
        )

        fixed_run()
        varlen_run_device()
        varlen_run_prefix()
        varlen_run_host()
        torch.npu.synchronize()

        packed_fixed_out = transpose_valid_chunks(
            fixed_out.reshape(1, total_tokens, args.H, args.chunk_size),
            uniform_cu_seqlens,
            args.chunk_size,
        )
        packed_varlen_out_device = transpose_valid_chunks(
            varlen_out_device,
            cu_seqlens,
            args.chunk_size,
        )
        packed_varlen_out_prefix = transpose_valid_chunks(
            varlen_out_prefix,
            cu_seqlens,
            args.chunk_size,
        )
        packed_varlen_out_host = transpose_valid_chunks(
            varlen_out_host,
            cu_seqlens,
            args.chunk_size,
        )
        max_abs_diff_device, rel_frob_diff_device = accuracy_metrics(
            packed_fixed_out,
            packed_varlen_out_device,
        )
        max_abs_diff_host, rel_frob_diff_host = accuracy_metrics(
            packed_fixed_out,
            packed_varlen_out_host,
        )
        max_abs_diff_prefix, rel_frob_diff_prefix = accuracy_metrics(
            packed_fixed_out,
            packed_varlen_out_prefix,
        )
        print(
            f"  accuracy vs fixed: device max_abs_diff={max_abs_diff_device:.3e}, "
            f"device rel_frob_diff={rel_frob_diff_device:.3e}, "
            f"prefix max_abs_diff={max_abs_diff_prefix:.3e}, "
            f"prefix rel_frob_diff={rel_frob_diff_prefix:.3e}, "
            f"host max_abs_diff={max_abs_diff_host:.3e}, "
            f"host rel_frob_diff={rel_frob_diff_host:.3e}"
        )

        fixed_times_ms = benchmark_ms(
            fixed_run,
            warmup_iters=args.warmup,
            benchmark_iters=args.repeats,
            device=NPU_DEVICE,
        )
        varlen_device_times_ms = benchmark_ms(
            varlen_run_device,
            warmup_iters=args.warmup,
            benchmark_iters=args.repeats,
            device=NPU_DEVICE,
        )
        prefix_metadata_times_ms = benchmark_prefix_metadata_prep_ms(
            cu_seqlens_cpu,
            args.chunk_size,
            benchmark_iters=args.repeats,
            device=NPU_DEVICE,
        )
        varlen_prefix_kernel_times_ms = benchmark_ms(
            varlen_run_prefix,
            warmup_iters=args.warmup,
            benchmark_iters=args.repeats,
            device=NPU_DEVICE,
        )
        host_metadata_times_ms = benchmark_host_metadata_prep_ms(
            cu_seqlens_cpu,
            args.chunk_size,
            benchmark_iters=args.repeats,
            device=NPU_DEVICE,
        )
        varlen_host_kernel_times_ms = benchmark_ms(
            varlen_run_host,
            warmup_iters=args.warmup,
            benchmark_iters=args.repeats,
            device=NPU_DEVICE,
        )

        fixed_row = {
            "inverse_type": "bsnd-fixed",
            "metadata_strategy": "none",
            "dtype": "fp16",
            "B": args.B,
            "T": seqlen,
            "aggregated_T": total_tokens,
            "padded_T": total_tokens,
            "H": args.H,
            "numel": fixed_input.numel(),
            "valid_numel": fixed_input.numel(),
            "chunk_size": args.chunk_size,
            "time_us": int(round(np.mean(fixed_times_ms) * 1000.0)),
            "kernel_time_us": int(round(np.mean(fixed_times_ms) * 1000.0)),
            "metadata_time_us": 0,
            "max_abs_diff_to_fixed": 0.0,
            "rel_frob_diff_to_fixed": 0.0,
            "sample_id": "",
            "seq_lens": "",
        }
        add_bandwidth_fields(fixed_row)

        varlen_device_row = {
            "inverse_type": "bsnd-varlen-uniform",
            "metadata_strategy": "device-cu_seqlens",
            "dtype": "fp16",
            "B": args.B,
            "T": seqlen,
            "aggregated_T": total_tokens,
            "padded_T": total_tokens,
            "H": args.H,
            "numel": varlen_input.numel(),
            "valid_numel": total_tokens * args.H * args.chunk_size,
            "chunk_size": args.chunk_size,
            "time_us": int(round(np.mean(varlen_device_times_ms) * 1000.0)),
            "kernel_time_us": int(round(np.mean(varlen_device_times_ms) * 1000.0)),
            "metadata_time_us": 0,
            "max_abs_diff_to_fixed": max_abs_diff_device,
            "rel_frob_diff_to_fixed": rel_frob_diff_device,
            "sample_id": "",
            "seq_lens": ",".join([str(seqlen)] * args.B),
        }
        add_bandwidth_fields(varlen_device_row)

        avg_prefix_metadata_us = int(round(np.mean(prefix_metadata_times_ms) * 1000.0))
        avg_prefix_kernel_us = int(round(np.mean(varlen_prefix_kernel_times_ms) * 1000.0))
        varlen_prefix_row = {
            "inverse_type": "bsnd-varlen-uniform",
            "metadata_strategy": "device-chunk-prefix",
            "dtype": "fp16",
            "B": args.B,
            "T": seqlen,
            "aggregated_T": total_tokens,
            "padded_T": total_tokens,
            "H": args.H,
            "numel": varlen_input.numel(),
            "valid_numel": total_tokens * args.H * args.chunk_size,
            "chunk_size": args.chunk_size,
            "time_us": avg_prefix_metadata_us + avg_prefix_kernel_us,
            "kernel_time_us": avg_prefix_kernel_us,
            "metadata_time_us": avg_prefix_metadata_us,
            "max_abs_diff_to_fixed": max_abs_diff_prefix,
            "rel_frob_diff_to_fixed": rel_frob_diff_prefix,
            "sample_id": "",
            "seq_lens": ",".join([str(seqlen)] * args.B),
        }
        add_bandwidth_fields(varlen_prefix_row)

        avg_host_metadata_us = int(round(np.mean(host_metadata_times_ms) * 1000.0))
        avg_host_kernel_us = int(round(np.mean(varlen_host_kernel_times_ms) * 1000.0))
        varlen_host_row = {
            "inverse_type": "bsnd-varlen-uniform",
            "metadata_strategy": "host-cpp",
            "dtype": "fp16",
            "B": args.B,
            "T": seqlen,
            "aggregated_T": total_tokens,
            "padded_T": total_tokens,
            "H": args.H,
            "numel": varlen_input.numel(),
            "valid_numel": total_tokens * args.H * args.chunk_size,
            "chunk_size": args.chunk_size,
            "time_us": avg_host_metadata_us + avg_host_kernel_us,
            "kernel_time_us": avg_host_kernel_us,
            "metadata_time_us": avg_host_metadata_us,
            "max_abs_diff_to_fixed": max_abs_diff_host,
            "rel_frob_diff_to_fixed": rel_frob_diff_host,
            "sample_id": "",
            "seq_lens": ",".join([str(seqlen)] * args.B),
        }
        add_bandwidth_fields(varlen_host_row)

        rows.extend([fixed_row, varlen_device_row, varlen_prefix_row, varlen_host_row])
        print(
            f"  fixed: time_us={fixed_row['time_us']}, bw_gbs={fixed_row['bw_gbs']:.2f} | "
            f"varlen-device: time_us={varlen_device_row['time_us']}, "
            f"bw_gbs={varlen_device_row['bw_gbs']:.2f} | "
            f"varlen-prefix: time_us={varlen_prefix_row['time_us']} "
            f"(meta={varlen_prefix_row['metadata_time_us']}, kernel={varlen_prefix_row['kernel_time_us']}), "
            f"bw_gbs={varlen_prefix_row['bw_gbs']:.2f} | "
            f"varlen-host: time_us={varlen_host_row['time_us']} "
            f"(meta={varlen_host_row['metadata_time_us']}, kernel={varlen_host_row['kernel_time_us']}), "
            f"bw_gbs={varlen_host_row['bw_gbs']:.2f}"
        )
        device_metadata_overhead_us = (
            varlen_device_row["kernel_time_us"] - varlen_host_row["kernel_time_us"]
        )
        prefix_metadata_overhead_us = (
            varlen_device_row["kernel_time_us"] - varlen_prefix_row["kernel_time_us"]
        )
        print(
            f"  metadata overhead comparison: device_vs_host_kernel_delta_us={device_metadata_overhead_us}, "
            f"device_vs_prefix_kernel_delta_us={prefix_metadata_overhead_us}, "
            f"prefix_metadata_us={varlen_prefix_row['metadata_time_us']}, "
            f"host_cpp_metadata_us={varlen_host_row['metadata_time_us']}"
        )

        for sample_idx in range(args.true_varlen_samples):
            seq_lens = sample_true_varlen_lengths(args.B, total_tokens, rng)
            packed_input, cu_seqlens = build_true_varlen_input(
                seq_lens=seq_lens,
                num_heads=args.H,
                chunk_size=args.chunk_size,
                feature_dim=args.feature_dim,
                device=NPU_DEVICE,
            )
            varlen_run_true, _ = make_varlen_runner(
                tri_inv_func,
                packed_input,
                cu_seqlens,
            )
            times_ms = benchmark_ms(
                varlen_run_true,
                warmup_iters=args.warmup,
                benchmark_iters=args.repeats,
                device=NPU_DEVICE,
            )
            row = {
                "inverse_type": "bsnd-varlen-true",
                "metadata_strategy": "device-cu_seqlens",
                "dtype": "fp16",
                "B": args.B,
                "T": seqlen,
                "aggregated_T": total_tokens,
                "padded_T": int(packed_input.shape[1]),
                "H": args.H,
                "numel": packed_input.numel(),
                "valid_numel": total_tokens * args.H * args.chunk_size,
                "chunk_size": args.chunk_size,
                "time_us": int(round(np.mean(times_ms) * 1000.0)),
                "kernel_time_us": int(round(np.mean(times_ms) * 1000.0)),
                "metadata_time_us": 0,
                "max_abs_diff_to_fixed": "",
                "rel_frob_diff_to_fixed": "",
                "sample_id": sample_idx,
                "seq_lens": ",".join(map(str, seq_lens)),
            }
            add_bandwidth_fields(row)
            true_varlen_rows.append(row)
            print(
                f"  true-varlen sample={sample_idx}: aggregated_T={total_tokens}, "
                f"padded_T={row['padded_T']}, bw_gbs={row['bw_gbs']:.2f}"
            )

    if not rows:
        raise RuntimeError("No benchmark rows were generated.")

    write_csv(csv_path, rows)
    plot_bandwidth(
        plot_path,
        rows,
        batch_size=args.B,
        num_heads=args.H,
        chunk_size=args.chunk_size,
    )
    write_csv(true_varlen_csv_path, true_varlen_rows)
    plot_true_varlen_scatter(
        true_varlen_plot_path,
        true_varlen_rows,
        batch_size=args.B,
        num_heads=args.H,
        chunk_size=args.chunk_size,
    )
    print(f"\nWrote CSV: {csv_path}")
    print(f"Wrote plot: {plot_path}")
    print(f"Wrote true-varlen CSV: {true_varlen_csv_path}")
    print(f"Wrote true-varlen plot: {true_varlen_plot_path}")


if __name__ == "__main__":
    main()
