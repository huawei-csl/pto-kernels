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

This script only benchmarks the PTO-ISA BSND kernel in two modes:

1. `bsnd-fixed`:
   Original aligned BSND layout with shape `(B, T, H, D)`.
2. `bsnd-varlen-uniform`:
   The new varlen path using packed shape `(1, B*T, H, D)` with uniform
   `cu_seqlens = [0, T, 2T, ...]`.

The two modes use the same total token count and the same underlying chunk data,
so their latency / effective bandwidth can be compared directly. The script also
checks that both modes produce numerically matching results.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_npu  # noqa: F401

from jit_util_fast_inverse import jit_compile


DEFAULT_SEQLENS = (512, 1024, 2048, 4096, 8192, 16384)
DEFAULT_CACHE_SIZE = 256 * 1024 * 1024
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


def chunk_metadata_from_cu_seqlens(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cu_seqlens_np = cu_seqlens.detach().cpu().numpy().astype(np.int64, copy=False)
    seq_starts = cu_seqlens_np[:-1]
    seq_lens = cu_seqlens_np[1:] - seq_starts
    seq_num_chunks = (seq_lens + chunk_size - 1) // chunk_size
    total_chunks = int(seq_num_chunks.sum())

    chunk_indices = np.empty(total_chunks, dtype=np.int32)
    chunk_valid_sizes = np.empty(total_chunks, dtype=np.int32)
    cursor = 0
    for seq_start, seq_len, num_chunks in zip(seq_starts, seq_lens, seq_num_chunks):
        num_chunks_int = int(num_chunks)
        local_offsets = np.arange(num_chunks_int, dtype=np.int64) * chunk_size
        next_cursor = cursor + num_chunks_int
        chunk_indices[cursor:next_cursor] = (seq_start + local_offsets).astype(
            np.int32,
            copy=False,
        )
        chunk_valid_sizes[cursor:next_cursor] = np.minimum(
            chunk_size,
            seq_len - local_offsets,
        ).astype(np.int32, copy=False)
        cursor = next_cursor

    return (
        torch.from_numpy(chunk_indices).to(device=cu_seqlens.device),
        torch.from_numpy(chunk_valid_sizes).to(device=cu_seqlens.device),
    )


def random_chunk_mats(
    total_chunks: int,
    num_heads: int,
    chunk_size: int,
    scale: float,
    device: str,
) -> torch.Tensor:
    return scale * torch.triu(
        torch.rand(
            (total_chunks, num_heads, chunk_size, chunk_size),
            dtype=torch.half,
            device=device,
        ),
        diagonal=1,
    )


def build_fixed_bsnd_input(
    chunk_mats: torch.Tensor,
    batch_size: int,
    seqlen: int,
    num_heads: int,
    chunk_size: int,
) -> torch.Tensor:
    return (
        chunk_mats.transpose(1, 2)
        .contiguous()
        .reshape(batch_size, seqlen, num_heads, chunk_size)
    )


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
    scale: float,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    cu_seqlens = np.cumsum([0, *seq_lens], dtype=np.int64)
    num_chunks = sum((seq_len + chunk_size - 1) // chunk_size for seq_len in seq_lens)
    chunk_mats = random_chunk_mats(
        total_chunks=num_chunks,
        num_heads=num_heads,
        chunk_size=chunk_size,
        scale=scale,
        device=device,
    )

    packed_input = torch.zeros(
        (1, int(cu_seqlens[-1]), num_heads, chunk_size),
        dtype=torch.half,
        device=device,
    )
    chunk_idx = 0
    token_row = 0

    for seq_len in seq_lens:
        for local_chunk_start in range(0, seq_len, chunk_size):
            actual_size = min(chunk_size, seq_len - local_chunk_start)
            chunk = chunk_mats[chunk_idx]
            for head_idx in range(num_heads):
                packed_input[
                    0,
                    token_row : token_row + actual_size,
                    head_idx,
                    :actual_size,
                ] = chunk[head_idx, :actual_size, :actual_size]
            token_row += actual_size
            chunk_idx += 1

    return (
        packed_input.contiguous(),
        torch.tensor(cu_seqlens.tolist(), dtype=torch.int32, device=device),
    )


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
    seq_lens = cu_seqlens[1:].to(torch.int64) - cu_seqlens[:-1].to(torch.int64)
    num_chunks = ((seq_lens + matrix_size - 1) // matrix_size).sum().item()
    num_matrices = int(num_chunks) * num_bsnd_heads
    tensor_out = torch.empty_like(tensor_in, dtype=torch.float32)
    minus_identity = make_minus_identity(matrix_size, str(tensor_in.device))
    chunk_indices, chunk_valid_sizes = chunk_metadata_from_cu_seqlens(
        cu_seqlens,
        matrix_size,
    )

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
    varlen_rows = [row for row in rows if row["inverse_type"] == "bsnd-varlen-uniform"]

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(
        [int(row["T"]) / 1000.0 for row in fixed_rows],
        [float(row["bw_gbs"]) for row in fixed_rows],
        marker="o",
        linewidth=2,
        label="BSND fixed",
    )
    ax.plot(
        [int(row["T"]) / 1000.0 for row in varlen_rows],
        [float(row["bw_gbs"]) for row in varlen_rows],
        marker="s",
        linewidth=2,
        label="BSND varlen-uniform",
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
        "--seqlens",
        type=parse_int_list,
        default=DEFAULT_SEQLENS,
        metavar="T[,T,...]",
        help=(
            "Comma-separated dense per-sequence lengths to benchmark "
            f"(default: {','.join(map(str, DEFAULT_SEQLENS))})"
        ),
    )
    parser.add_argument("--scale", type=float, default=0.1)
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

        total_chunks = args.B * seqlen // args.chunk_size
        total_tokens = args.B * seqlen
        print(
            f"Profiling T={seqlen}, total_tokens={total_tokens}, "
            f"B={args.B}, H={args.H}, chunk_size={args.chunk_size}"
        )

        chunk_mats = random_chunk_mats(
            total_chunks=total_chunks,
            num_heads=args.H,
            chunk_size=args.chunk_size,
            scale=args.scale,
            device=NPU_DEVICE,
        )
        fixed_input = build_fixed_bsnd_input(
            chunk_mats,
            batch_size=args.B,
            seqlen=seqlen,
            num_heads=args.H,
            chunk_size=args.chunk_size,
        )
        varlen_input, cu_seqlens = build_uniform_varlen_input(
            fixed_input,
            batch_size=args.B,
            seqlen=seqlen,
            chunk_size=args.chunk_size,
        )

        print(f"  uniform cu_seqlens: {cu_seqlens.cpu().tolist()}")

        fixed_run, fixed_out = make_fixed_runner(tri_inv_func, fixed_input)
        varlen_run, varlen_out = make_varlen_runner(
            tri_inv_func,
            varlen_input,
            cu_seqlens,
        )

        fixed_run()
        varlen_run()
        torch.npu.synchronize()

        packed_fixed_out = fixed_out.reshape(1, total_tokens, args.H, args.chunk_size)
        max_abs_diff, rel_frob_diff = accuracy_metrics(packed_fixed_out, varlen_out)
        print(
            f"  accuracy vs fixed: max_abs_diff={max_abs_diff:.3e}, "
            f"rel_frob_diff={rel_frob_diff:.3e}"
        )

        fixed_times_ms = benchmark_ms(
            fixed_run,
            warmup_iters=args.warmup,
            benchmark_iters=args.repeats,
            device=NPU_DEVICE,
        )
        varlen_times_ms = benchmark_ms(
            varlen_run,
            warmup_iters=args.warmup,
            benchmark_iters=args.repeats,
            device=NPU_DEVICE,
        )

        fixed_row = {
            "inverse_type": "bsnd-fixed",
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
            "max_abs_diff_to_fixed": 0.0,
            "rel_frob_diff_to_fixed": 0.0,
            "sample_id": "",
            "seq_lens": "",
        }
        add_bandwidth_fields(fixed_row)

        varlen_row = {
            "inverse_type": "bsnd-varlen-uniform",
            "dtype": "fp16",
            "B": args.B,
            "T": seqlen,
            "aggregated_T": total_tokens,
            "padded_T": total_tokens,
            "H": args.H,
            "numel": varlen_input.numel(),
            "valid_numel": total_tokens * args.H * args.chunk_size,
            "chunk_size": args.chunk_size,
            "time_us": int(round(np.mean(varlen_times_ms) * 1000.0)),
            "max_abs_diff_to_fixed": max_abs_diff,
            "rel_frob_diff_to_fixed": rel_frob_diff,
            "sample_id": "",
            "seq_lens": ",".join([str(seqlen)] * args.B),
        }
        add_bandwidth_fields(varlen_row)

        rows.extend([fixed_row, varlen_row])
        print(
            f"  fixed: time_us={fixed_row['time_us']}, bw_gbs={fixed_row['bw_gbs']:.2f} | "
            f"varlen-uniform: time_us={varlen_row['time_us']}, bw_gbs={varlen_row['bw_gbs']:.2f}"
        )

        for sample_idx in range(args.true_varlen_samples):
            seq_lens = sample_true_varlen_lengths(args.B, total_tokens, rng)
            packed_input, cu_seqlens = build_true_varlen_input(
                seq_lens=seq_lens,
                num_heads=args.H,
                chunk_size=args.chunk_size,
                scale=args.scale,
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
