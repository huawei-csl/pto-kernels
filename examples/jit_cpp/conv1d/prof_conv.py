"""Minimal profiling driver: run ONLY our depthwise conv kernel, N times, at a
given (L, W). Used under msprof to isolate per-kernel behaviour.

    msprof --application="python prof_conv.py 4096 2048 200" \
           --output=/tmp/prof_slow --ai-core=on --aic-metrics=PipeUtilization
"""
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from jit_util_conv1d_dw import jit_compile, K

torch.npu.set_device("npu")
torch.manual_seed(0)

L = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
W = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
N = int(sys.argv[3]) if len(sys.argv) > 3 else 200

fn = jit_compile(str(Path(__file__).resolve().parent / "conv1d_dw_pto.cpp"),
                 verbose=False)

x = (2 * torch.rand((L, W), device="npu", dtype=torch.float16) - 1)
w = (torch.rand((K, W), device="npu", dtype=torch.float32) - 0.5)
bias = (torch.rand((W,), device="npu", dtype=torch.float32) - 0.5)

# warmup (kept small; msprof aggregates per-op so warmup ops are just extra rows)
for _ in range(10):
    fn(x, w, bias)
torch.npu.synchronize()

for _ in range(N):
    fn(x, w, bias)
torch.npu.synchronize()
print(f"done L={L} W={W} N={N}")
