from __future__ import annotations

from typing import Callable, Literal

import torch


def do_bench(
    fn: Callable[[], object],
    warmup_iters: int = 5,
    benchmark_iters: int = 15,
    aggregation: Literal["mean", "none"] = "mean",
    unit: Literal["s", "ms", "us", "ns"] = "us",
    flush_cache: bool = True,
) -> float | list[float]:
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
