"""End-to-end wall time, including allocation and launch; compile/warmup excluded."""

import argparse
import json
import os
from pathlib import Path
import statistics
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
    parser.add_argument(
        "--profile", type=Path, help="Trace one call at the final token length"
    )
    args = parser.parse_args()
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
        backward(q, k, v, g, beta, do)
        torch.npu.synchronize()
        times, enqueue = [], []
        for _ in range(args.repeats):
            start = time.perf_counter()
            backward(q, k, v, g, beta, do)
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
                backward(q, k, v, g, beta, do)
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
