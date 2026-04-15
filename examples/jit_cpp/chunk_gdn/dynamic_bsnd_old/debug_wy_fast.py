from __future__ import annotations
import math
import torch
import pto_dynamic_common  # noqa: F401
from dynamic_kernel_libs import (
    pack_bsh_tensor,
    pack_bshd_tensor,
    run_wy_fast_kernel,
)
from run_chunk_cumsum_dynamic_bsnd import total_chunks_from_cu


torch_npu = torch.npu  # noqa: F401
CHUNK = 128
RTOL = 1e-3
ATOL = 1e-3


def ref_wy_fast_bsnd(k, v, beta, g_packed, a_packed, *, chunk_size, cu_seqlens=None):
    k_packed = pack_bshd_tensor(k, chunk_size=chunk_size, cu_seqlens=cu_seqlens).float()
    v_packed = pack_bshd_tensor(v, chunk_size=chunk_size, cu_seqlens=cu_seqlens).float()
    beta_packed = pack_bsh_tensor(beta, chunk_size=chunk_size, cu_seqlens=cu_seqlens)
    a_float = a_packed.float()
    a2 = (a_float * beta_packed.unsqueeze(-1)).to(torch.float16)
    a1 = (a_float * (beta_packed * torch.exp(g_packed.float())).unsqueeze(-1)).to(torch.float16)
    w = torch.matmul(a1.float(), k_packed).to(torch.float16)
    u = torch.matmul(a2.float(), v_packed).to(torch.float16)
    return w, u


def main():
    torch.manual_seed(0)
    torch.npu.set_device("npu:0")

    shape = (2, 256, 2, 128)
    k = torch.randn(shape, device="npu", dtype=torch.float16)
    v = torch.randn(shape, device="npu", dtype=torch.float16)
    beta = torch.rand(shape[:-1], device="npu", dtype=torch.float16)
    total_chunks = shape[0] * math.ceil(shape[1] / CHUNK)
    g_packed = torch.randn((total_chunks, shape[2], CHUNK), device="npu", dtype=torch.float32)
    a_packed = torch.randn((total_chunks, shape[2], CHUNK, CHUNK), device="npu", dtype=torch.float16)
    w_out = torch.zeros((total_chunks, shape[2], CHUNK, shape[3]), device="npu", dtype=torch.float16)
    u_out = torch.zeros_like(w_out)
    ref_w, ref_u = ref_wy_fast_bsnd(k, v, beta, g_packed, a_packed, chunk_size=CHUNK)

    run_wy_fast_kernel(k, v, beta, g_packed, a_packed, w_out, u_out, chunk_size=CHUNK)
    torch.npu.synchronize()

    # Check u_out (A2 path) first
    try:
        torch.testing.assert_close(u_out.cpu(), ref_u.cpu(), rtol=RTOL, atol=ATOL)
        print("u_out (A2 path): PASSED")
    except AssertionError as e:
        print(f"u_out (A2 path): FAILED\n{e}")

    # Check w_out (A1 path)
    try:
        torch.testing.assert_close(w_out.cpu(), ref_w.cpu(), rtol=RTOL, atol=ATOL)
        print("w_out (A1 path): PASSED")
    except AssertionError as e:
        print(f"w_out (A1 path): FAILED\n{e}")

    # Detailed analysis of w_out errors
    w_cpu = w_out.cpu().float()
    ref_w_cpu = ref_w.cpu().float()
    diff = (w_cpu - ref_w_cpu).abs()
    max_diff_flat = diff.reshape(-1).argmax()
    max_diff_idx = []
    remaining = max_diff_flat.item()
    for s in reversed(diff.shape):
        max_diff_idx.insert(0, remaining % s)
        remaining //= s
    print(f"\nMax abs diff at index {tuple(max_diff_idx)}: {diff.max().item():.6f}")
    print(f"  actual: {w_cpu.reshape(-1)[max_diff_flat].item():.6f}")
    print(f"  expected: {ref_w_cpu.reshape(-1)[max_diff_flat].item():.6f}")

    # Check per-chunk, per-head
    for c in range(w_cpu.shape[0]):
        for h in range(w_cpu.shape[1]):
            chunk_diff = diff[c, h]
            max_err = chunk_diff.max().item()
            if max_err > ATOL:
                bad_rows = (chunk_diff.max(dim=1).values > ATOL).nonzero().squeeze(-1).tolist()
                print(f"  chunk={c} head={h}: max_err={max_err:.4f}, bad_rows={bad_rows[:10]}{'...' if len(bad_rows)>10 else ''}")


if __name__ == "__main__":
    main()
