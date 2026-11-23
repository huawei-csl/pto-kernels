from pathlib import Path
import sys

COMMON_DIR = Path(__file__).resolve().parents[1] / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

import torch
import torch_npu  # noqa: F401

from jit_util_linear_attention import jit_compile
from linear_attention_shared import kernel_src_path, make_inputs, measure_kernel_ms

B, H, L, D, C = 2, 2, 512, 128, 64
DTYPE = torch.float16


def main():
    torch.npu.set_device("npu:0")
    src = kernel_src_path(__file__)
    kernel = jit_compile(src)
    q, k, v = make_inputs(B, H, L, D)
    workspace_1 = torch.zeros((B, H, C, C), device="npu", dtype=DTYPE)
    workspace_2 = torch.zeros((B, H, D, D), device="npu", dtype=DTYPE)
    output = torch.zeros((B, H, L, D), device="npu", dtype=DTYPE)

    median_ms = measure_kernel_ms(
        lambda: kernel(q, k, v, workspace_1, workspace_2, output, block_dim=B * H),
        warmup=3,
        repeats=5,
    )
    print("shape", (B, H, L, D, C), "median_ms", median_ms)


if __name__ == "__main__":
    main()
