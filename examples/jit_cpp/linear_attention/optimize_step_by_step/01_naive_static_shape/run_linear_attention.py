import os

import torch
import torch_npu  # noqa: F401

from jit_util_linear_attention import jit_compile

B = 2
H = 2
L = 512
D = 128
C = 64
DTYPE = torch.float16
ATOL = 1e-2
RTOL = 1e-2


def ref_linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
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


def run_kernel(linear_attention_func, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    workspace_1 = torch.zeros((B, H, C, C), device=q.device, dtype=DTYPE)
    workspace_2 = torch.zeros((B, H, D, D), device=q.device, dtype=DTYPE)
    o = torch.zeros((B, H, L, D), device=q.device, dtype=DTYPE)

    linear_attention_func(q, k, v, workspace_1, workspace_2, o, block_dim=B * H)
    torch.npu.synchronize()
    return o


def main():
    torch.manual_seed(0)
    torch.npu.set_device("npu:0")

    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "linear_attention.cpp")
    print(f"Compiling {src} ...")
    linear_attention_func = jit_compile(src)
    print("Compilation successful.\n")

    q = torch.randn((B, H, L, D), device="npu", dtype=DTYPE)
    k = torch.randn((B, H, L, D), device="npu", dtype=DTYPE)
    v = torch.randn((B, H, L, D), device="npu", dtype=DTYPE)
    q = q / (q.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-6)
    k = k / (k.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-6)

    o = run_kernel(linear_attention_func, q, k, v)
    ref_o = ref_linear_attention(q, k, v)
    torch.testing.assert_close(o.cpu(), ref_o.cpu(), rtol=RTOL, atol=ATOL)
    print("Test passed!")


if __name__ == "__main__":
    main()
