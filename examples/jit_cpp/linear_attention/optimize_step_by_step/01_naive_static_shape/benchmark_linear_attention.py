import os
import time

import torch
import torch_npu  # noqa: F401

from jit_util_linear_attention import jit_compile

DTYPE = torch.float16
B, H, L, D, C = 2, 2, 512, 128, 64


def make_inputs():
    q = torch.randn((B, H, L, D), device='npu', dtype=DTYPE)
    k = torch.randn((B, H, L, D), device='npu', dtype=DTYPE)
    v = torch.randn((B, H, L, D), device='npu', dtype=DTYPE)
    q = q / (q.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-6)
    k = k / (k.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-6)
    return q, k, v


def main():
    torch.npu.set_device('npu:0')
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'linear_attention.cpp')
    kernel = jit_compile(src)
    q, k, v = make_inputs()
    workspace_1 = torch.zeros((B, H, C, C), device='npu', dtype=DTYPE)
    workspace_2 = torch.zeros((B, H, D, D), device='npu', dtype=DTYPE)
    out = torch.zeros((B, H, L, D), device='npu', dtype=DTYPE)
    for _ in range(3):
        kernel(q, k, v, workspace_1, workspace_2, out, block_dim=B * H)
    torch.npu.synchronize()
    samples = []
    for _ in range(5):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        kernel(q, k, v, workspace_1, workspace_2, out, block_dim=B * H)
        end.record()
        torch.npu.synchronize()
        samples.append(start.elapsed_time(end))
    print('shape', (B, H, L, D, C), 'median_ms', sorted(samples)[len(samples)//2])


if __name__ == '__main__':
    main()
