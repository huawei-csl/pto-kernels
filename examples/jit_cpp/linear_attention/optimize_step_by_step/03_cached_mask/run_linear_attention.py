import os
from functools import lru_cache

import torch
import torch_npu  # noqa: F401

from jit_util_linear_attention import BLOCK_DIM, get_causal_mask, jit_compile

DTYPE = torch.float16
RTOL = 1e-2


def ref_linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    B, H, L, D = q.shape
    q = q.float()
    k = k.float()
    v = v.float()

    h = torch.zeros((B, H, D, D), device=q.device, dtype=torch.float32)
    o = torch.zeros((B, H, L, D), device=q.device, dtype=torch.float32)

    for i in range(L):
        q_i = q[:, :, i, :]
        k_i = k[:, :, i, :]
        v_i = v[:, :, i, :]
        h = h + torch.einsum("bhi,bhj->bhij", k_i, v_i)
        o[:, :, i, :] = torch.einsum("bhi,bhij->bhj", q_i, h)

    return o.to(DTYPE)


@lru_cache(maxsize=None)
def _compiled_kernel(src: str, h: int, d: int, c: int):
    return jit_compile(src, num_heads=h, hidden_size=d, chunk_size=c)


def run_kernel(
    src: str, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, chunk_size: int
):
    b, h, l, d = q.shape
    if l % chunk_size != 0:
        raise ValueError("This PTO-ISA example currently requires L to be a multiple of C.")

    linear_attention_func = _compiled_kernel(src, h, d, chunk_size)
    workspace_1 = torch.zeros((BLOCK_DIM, chunk_size, chunk_size), device=q.device, dtype=DTYPE)
    workspace_2 = torch.zeros((BLOCK_DIM, d, d), device=q.device, dtype=DTYPE)
    causal_mask = get_causal_mask(chunk_size, DTYPE, q.device.index or 0)
    o = torch.zeros((b, h, l, d), device=q.device, dtype=DTYPE)

    linear_attention_func(
        q, k, v, workspace_1, workspace_2, causal_mask, o, block_dim=BLOCK_DIM
    )
    torch.npu.synchronize()
    return o


def main():
    torch.manual_seed(0)
    torch.npu.set_device("npu:0")

    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "linear_attention.cpp")
    test_configs = [
        (1, 2, 64, 128, 64),
        (1, 2, 256, 128, 64),
        (4, 2, 128, 128, 64),
        (8, 2, 512, 128, 64),
        (10, 2, 512, 128, 64),
        (16, 2, 256, 128, 64),
        (32, 2, 128, 128, 64),
        (1, 2, 1024, 128, 64),
        (8, 2, 2048, 128, 64),
        (2, 2, 4096, 128, 64),
        (16, 2, 1024, 128, 64),
        (50, 20, 128, 128, 64),
    ]

    for b, h, l, d, c in test_configs:
        print(f"Testing B={b}, H={h}, L={l}, D={d}, C={c}  (B*H={b * h})")
        q = torch.randn((b, h, l, d), device="npu", dtype=DTYPE)
        k = torch.randn((b, h, l, d), device="npu", dtype=DTYPE)
        v = torch.randn((b, h, l, d), device="npu", dtype=DTYPE)
        q = q / (q.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-6)
        k = k / (k.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-6)

        o = run_kernel(src, q, k, v, c)
        ref_o = ref_linear_attention(q, k, v)

        if l >= 4096:
            atol = 4e-2
        elif l >= 2048:
            atol = 2e-2
        else:
            atol = 1e-2

        torch.testing.assert_close(o.cpu(), ref_o.cpu(), rtol=RTOL, atol=atol)
        print("  passed!")

    print("Kernel Output Match!")


if __name__ == "__main__":
    main()
