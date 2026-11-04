"""End-to-end wall time, including allocation and launch; compile/warmup excluded."""

import argparse
import json
import statistics
import time

import torch
import torch_npu  # noqa: F401

from csrc.kernel.gdn_mega_bwd.backward import backward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = False
    torch.manual_seed(42)
    for length in args.tokens:
        q = torch.randn(1, length, 2, 128, device="npu", dtype=torch.float16) * 0.1
        k = torch.nn.functional.normalize(torch.randn_like(q), dim=-1)
        v = torch.randn(1, length, 8, 128, device="npu", dtype=torch.float16)
        g = -torch.rand(1, length, 8, device="npu") * 0.1
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
                    "heads": 8,
                    "median_ms": statistics.median(times),
                    "enqueue_median_ms": statistics.median(enqueue),
                    "samples_ms": times,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
