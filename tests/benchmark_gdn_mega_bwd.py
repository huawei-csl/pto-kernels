"""End-to-end wall time, including allocation and launch; compile/warmup excluded."""

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch
import torch_npu  # noqa: F401

from csrc.kernel.gdn_mega_bwd.backward import backward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--key-heads", type=int, default=2)
    parser.add_argument("--inverse-cache", action="store_true")
    parser.add_argument("--gate-cache", action="store_true")
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument(
        "--profile", type=Path, help="Trace one call at the final token length"
    )
    args = parser.parse_args()
    if args.forward_only and args.profile:
        parser.error("--profile records backward; omit it with --forward-only")
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = False
    torch.manual_seed(42)
    for length in args.tokens:
        q = (
            torch.randn(
                1, length, args.key_heads, 128, device="npu", dtype=torch.float16
            )
            * 0.1
        )
        k = torch.nn.functional.normalize(torch.randn_like(q), dim=-1)
        v = torch.randn(1, length, args.heads, 128, device="npu", dtype=torch.float16)
        g = -torch.rand(1, length, args.heads, device="npu") * 0.1
        beta = torch.rand_like(g).half()
        do = torch.randn_like(v)
        cache, gate_cache, forward_times, forward_device = None, None, {}, {}
        if args.inverse_cache or args.gate_cache or args.forward_only:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from test_gdn_mega import _prepare_gdn_mega, pto_gdn_mega

            cu = torch.tensor([0, length], device="npu", dtype=torch.int32)
            launch = _prepare_gdn_mega(q, k, v, g, beta, cu)
            if args.inverse_cache:
                cache = torch.empty(
                    (length + 127) // 128,
                    args.heads,
                    128,
                    128,
                    device="npu",
                    dtype=torch.float32,
                )
                pto_gdn_mega(**launch)
                before = launch["out"].clone(), launch["final_state"].clone()
                pto_gdn_mega(**launch, inverse_cache=cache)
                for name, ref in zip(("out", "final_state"), before):
                    torch.testing.assert_close(launch[name], ref, atol=0, rtol=0)
            forward_times = {"forward_no_cache_ms": []}
            if cache is not None:
                forward_times["forward_cache_ms"] = []
            for trial in range(args.repeats + 2):
                modes = (
                    (False,)
                    if cache is None
                    else ((False, True) if trial % 2 else (True, False))
                )
                for enabled in modes:
                    torch.npu.synchronize()
                    start = time.perf_counter()
                    pto_gdn_mega(**launch, inverse_cache=cache if enabled else None)
                    torch.npu.synchronize()
                    if trial >= 2:
                        forward_times[
                            "forward_cache_ms" if enabled else "forward_no_cache_ms"
                        ].append((time.perf_counter() - start) * 1000)
            if args.gate_cache:
                gate_cache = tuple(launch[name] for name in ("g_t", "g_sum", "beta_t"))
            from benchmarking.gdn_mega import time_launch

            forward_device["no_cache"] = time_launch(launch, trials=args.repeats)
            if cache is not None:
                forward_device["cache"] = time_launch(
                    dict(launch, inverse_cache=cache), trials=args.repeats
                )
        if args.forward_only:
            print(
                json.dumps(
                    {
                        "tokens": length,
                        "heads": args.heads,
                        "key_heads": args.key_heads,
                        "forward_samples": forward_times,
                        "forward_device": forward_device,
                        "forward_medians": {
                            k: statistics.median(v) for k, v in forward_times.items()
                        },
                    }
                ),
                flush=True,
            )
            continue
        backward(q, k, v, g, beta, do, inverse_cache=cache, gate_cache=gate_cache)
        torch.npu.synchronize()
        times, enqueue = [], []
        for _ in range(args.repeats):
            start = time.perf_counter()
            backward(q, k, v, g, beta, do, inverse_cache=cache, gate_cache=gate_cache)
            enqueue.append((time.perf_counter() - start) * 1000)
            torch.npu.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        print(
            json.dumps(
                {
                    "tokens": length,
                    "heads": args.heads,
                    "key_heads": args.key_heads,
                    "median_ms": statistics.median(times),
                    "enqueue_median_ms": statistics.median(enqueue),
                    "samples_ms": times,
                    "inverse_cache": args.inverse_cache,
                    "gate_cache": args.gate_cache,
                    "forward_samples": forward_times,
                    "forward_device": forward_device,
                    "forward_medians": {
                        k: statistics.median(v) for k, v in forward_times.items()
                    },
                }
            ),
            flush=True,
        )
    if args.profile:
        directory = args.profile.resolve()
        directory.mkdir(parents=True, exist_ok=True)
        previous = Path.cwd()
        try:
            os.chdir(directory)  # Keep CANN's intermediate exports here too.
            activities = [
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ]
            with torch_npu.profiler.profile(activities=activities) as prof:
                backward(
                    q, k, v, g, beta, do, inverse_cache=cache, gate_cache=gate_cache
                )
                torch.npu.synchronize()
            prof.export_chrome_trace("trace.json")
            trace = json.loads(Path("trace.json").read_text())
            events = trace if isinstance(trace, list) else trace["traceEvents"]
            stages = [
                {"name": e["name"], "duration_us": e["dur"]}
                for e in events
                if "gdn_bwd_stage" in e.get("name", "") and e.get("ph") == "X"
            ]
            Path("stages.json").write_text(json.dumps(stages, indent=2) + "\n")
            print(
                json.dumps({"device_stage_launches": len(stages), "stages": stages}),
                flush=True,
            )
        finally:
            os.chdir(previous)


if __name__ == "__main__":
    main()
