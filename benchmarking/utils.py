from __future__ import annotations

from typing import Callable, Literal

import torch

FlushMode = Literal["write", "read", "none"]


def do_bench(
    fn: Callable[[], object],
    warmup_iters: int = 5,
    benchmark_iters: int = 15,
    aggregation: Literal["mean", "none"] = "mean",
    unit: Literal["s", "ms", "us", "ns"] = "us",
    flush_cache: bool = True,
    flush_mode: FlushMode = "write",
) -> float | list[float]:
    """Time `fn` with NPU events, optionally flushing the cache between iterations.

    `flush_mode` selects what the flush leaves behind in L2, and it is not a
    neutral choice:

    * ``"write"`` zeroes a 256 MB buffer. This is the historical behaviour and
      the default. It evicts the previous iteration's data but leaves L2 full
      of **dirty** lines, so the timed body pays to write them back. On a
      910B2 (192 MiB L2) that tax is bounded by L2 capacity, so it amortizes as
      the timed body grows: a measurement can appear to get faster with size
      while nothing about the code changed.
    * ``"read"`` reads the same buffer instead. It evicts just as much but
      leaves L2 **clean**, so the timed body pays only for its own traffic.
    * ``"none"`` skips the flush. Cheapest, but anything that fits in L2 stays
      resident across iterations and the result is a cache-hit number.

    How much this matters, measured on one 910B2 at a 512 MB fp32 input:

    =================  ========  =======  =======
    case               "write"   "read"   "none"
    =================  ========  =======  =======
    ``torch.sum``         940     1477     1488
    ``pto_cube_reduce``   790     1064     1063
    =================  ========  =======  =======

    The write flush charges both arms the same writeback, so implementations
    that differ by 1.4x measure 1.19x apart. A/B ratios between two builds of
    the same kernel survive that, because both pay it; comparisons against an
    external baseline, and any claim about headroom against hardware limits,
    do not.

    Neither regime is universally correct -- which one matches production
    depends on what ran before the kernel under test. Prefer reporting both
    over picking one silently.

    `flush_cache=False` is kept as a synonym for ``flush_mode="none"``.
    """
    import torch_npu

    start_events = [torch.npu.Event(enable_timing=True) for _ in range(benchmark_iters)]
    end_events = [torch.npu.Event(enable_timing=True) for _ in range(benchmark_iters)]

    if not flush_cache:
        flush_mode = "none"
    cache = None
    if flush_mode != "none":
        cache = torch.empty((256 * 1024 * 1024,), dtype=torch.int8).npu()

    for _ in range(warmup_iters):
        fn()
    torch_npu.npu.synchronize()

    for i in range(benchmark_iters):
        if cache is not None:
            if flush_mode == "write":
                cache.zero_()
            else:
                torch.max(cache)
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
