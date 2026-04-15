"""
Shared GDN kernel benchmark helpers (TileLang JIT or static ctypes). No TileLang import.
"""
from __future__ import annotations

from typing import Callable, Literal

KERNEL_ORDER = [
    "chunk_cumsum",
    "chunk_scaled_dot_kkt",
    "wy_fast",
    "chunk_h",
    "chunk_o",
]


def do_bench(
    fn: Callable[[], object],
    warmup_iters: int = 5,
    benchmark_iters: int = 15,
    aggregation: Literal["mean", "none"] = "mean",
    unit: Literal["s", "ms", "us", "ns"] = "ms",
    flush_cache: bool = True,
) -> float | list[float]:
    import torch
    import torch_npu

    start_events = [torch.npu.Event(enable_timing=True) for _ in range(benchmark_iters)]
    end_events = [torch.npu.Event(enable_timing=True) for _ in range(benchmark_iters)]

    cache = None
    if flush_cache:
        cache = torch.empty((256 * 1024 * 1024,), dtype=torch.int8).npu()

    for _ in range(warmup_iters):
        fn()
    torch_npu.npu.synchronize()

    for i in range(benchmark_iters):
        if cache is not None:
            cache.zero_()
        start_events[i].record()
        fn()
        end_events[i].record()

    torch_npu.npu.synchronize()
    factor = {"s": 1e-3, "ms": 1e0, "us": 1e3, "ns": 1e6}[unit]
    times = [
        factor * start.elapsed_time(end) for start, end in zip(start_events, end_events)
    ]
    if aggregation == "mean":
        return sum(times) / len(times)
    return times


def do_bench_triton(
    fn: Callable[[], object],
    warmup_iters: int = 5,
    benchmark_iters: int = 15,
    aggregation: Literal["mean", "none"] = "mean",
    unit: Literal["s", "ms", "us", "ns"] = "ms",
    flush_cache: bool = True,
) -> float | list[float]:
    """
    Triton kernel timing on NPU: use ``end.synchronize()`` on the timing event
    (see ``pto-kernels/.skills/npu_kernel_general/skills.md``); plain
    ``torch.npu.synchronize()`` may not wait for Triton work.
    """
    import torch
    import torch_npu

    cache = None
    if flush_cache:
        cache = torch.empty((256 * 1024 * 1024,), dtype=torch.int8).npu()

    for _ in range(warmup_iters):
        fn()
    torch_npu.npu.synchronize()

    times: list[float] = []
    factor = {"s": 1e-3, "ms": 1e0, "us": 1e3, "ns": 1e6}[unit]
    for _ in range(benchmark_iters):
        if cache is not None:
            cache.zero_()
        torch_npu.npu.synchronize()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(factor * start.elapsed_time(end))

    if aggregation == "mean":
        return sum(times) / len(times)
    return times


def format_ops(ops: int) -> str:
    return f"{ops:.2e}"


def format_ms(ms: float) -> str:
    return f"{ms:.2f}"


def format_tflops(ops: int, ms: float) -> str:
    return f"{ops / (ms * 1e9):.4f}"


def approx_ops_gdn(
    B: int, H: int, L: int, DK: int, DV: int, C: int
) -> dict[str, int]:
    """Approximate op counts (tilelang-ascend GDN README)."""
    return {
        "chunk_cumsum": B * H * L,
        "chunk_scaled_dot_kkt": B * H * L * C * DK,
        "solve_tril": B * H * L * C * C // 3,
        "wy_fast": B * H * L * C * (DK + DV),
        "chunk_h": 4 * B * H * L * DK * DV,
        "chunk_o": 5 * B * H * L * DK * DV,
    }


def approx_ops_gdn_triton(
    B: int, H: int, L: int, DK: int, DV: int, BT: int = 64
) -> dict[str, int]:
    """Op counts for vLLM Triton path: tile size ``BT`` (64) replaces README ``C`` (128)."""
    return {
        "chunk_cumsum": B * H * L,
        "chunk_scaled_dot_kkt": B * H * L * BT * DK,
        "wy_fast": B * H * L * BT * (DK + DV),
        "chunk_h": 4 * B * H * L * DK * DV,
        "chunk_o": 5 * B * H * L * DK * DV,
    }
