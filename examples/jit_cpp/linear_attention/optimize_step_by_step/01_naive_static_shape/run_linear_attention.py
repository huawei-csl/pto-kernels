from pathlib import Path
import sys

COMMON_DIR = Path(__file__).resolve().parents[1] / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

import torch
import torch_npu  # noqa: F401

from jit_util_linear_attention import jit_compile
from linear_attention_shared import run_correctness_cases

B = 2
H = 2
L = 512
D = 128
C = 64


def run_kernel(src: str, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, chunk_size: int):
    del chunk_size
    kernel = jit_compile(src)
    workspace_1 = torch.zeros((B, H, C, C), device=q.device, dtype=torch.float16)
    workspace_2 = torch.zeros((B, H, D, D), device=q.device, dtype=torch.float16)
    output = torch.zeros((B, H, L, D), device=q.device, dtype=torch.float16)
    kernel(q, k, v, workspace_1, workspace_2, output, block_dim=B * H)
    torch.npu.synchronize()
    return output


def main():
    run_correctness_cases(__file__, [(B, H, L, D, C)], run_kernel)


if __name__ == "__main__":
    main()
