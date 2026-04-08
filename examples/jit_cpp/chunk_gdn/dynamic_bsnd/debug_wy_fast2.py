from __future__ import annotations
import math
import ctypes
import os
import torch
import pto_dynamic_common  # noqa: F401
from dynamic_kernel_libs import (
    pack_bsh_tensor,
    pack_bshd_tensor,
    wy_fast_kernel,
)
from pto_dynamic_common import torch_to_ctypes, optional_torch_to_ctypes, BLOCK_DIM
from run_chunk_cumsum_dynamic_bsnd import total_chunks_from_cu


torch_npu = torch.npu
CHUNK = 128


def main():
    torch.manual_seed(0)
    torch.npu.set_device("npu:0")

    shape = (2, 256, 2, 128)
    B, S, H, D = shape
    k = torch.randn(shape, device="npu", dtype=torch.float16)
    v = torch.randn(shape, device="npu", dtype=torch.float16)
    beta = torch.rand((B, S, H), device="npu", dtype=torch.float16)
    total_chunks = B * math.ceil(S / CHUNK)
    g_packed = torch.randn((total_chunks, H, CHUNK), device="npu", dtype=torch.float32)
    a_packed = torch.randn((total_chunks, H, CHUNK, CHUNK), device="npu", dtype=torch.float16)

    # Reference computation
    beta_packed = pack_bsh_tensor(beta, chunk_size=CHUNK)
    a_float = a_packed.float()
    ref_a2 = (a_float * beta_packed.unsqueeze(-1)).to(torch.float16)
    ref_a1 = (a_float * (beta_packed * torch.exp(g_packed.float())).unsqueeze(-1)).to(torch.float16)

    # Run the kernel and inspect workspace
    w_out = torch.zeros((total_chunks, H, CHUNK, D), device="npu", dtype=torch.float16)
    u_out = torch.zeros_like(w_out)
    workspace_a1 = torch.zeros((total_chunks, H, CHUNK, CHUNK), device="npu", dtype=torch.float16)
    workspace_a2 = torch.zeros_like(workspace_a1)

    lib = wy_fast_kernel(H, D, CHUNK)
    stream = torch.npu.current_stream()._as_parameter_
    lib.call_kernel(
        BLOCK_DIM, stream,
        torch_to_ctypes(k.contiguous()),
        torch_to_ctypes(v.contiguous()),
        torch_to_ctypes(beta.contiguous()),
        torch_to_ctypes(g_packed.contiguous()),
        torch_to_ctypes(a_packed.contiguous()),
        torch_to_ctypes(workspace_a1),
        torch_to_ctypes(workspace_a2),
        torch_to_ctypes(w_out),
        torch_to_ctypes(u_out),
        optional_torch_to_ctypes(None),
        B,
        S,
    )
    torch.npu.synchronize()

    # Check workspace A2 (should be A * beta)
    print("=== Checking workspace_a2 (A * beta) ===")
    for c in range(total_chunks):
        for h in range(H):
            actual = workspace_a2[c, h].cpu().float()
            expected = ref_a2[c, h].cpu().float()
            diff = (actual - expected).abs()
            max_err = diff.max().item()
            if max_err > 0.01:
                bad_rows = (diff.max(dim=1).values > 0.01).nonzero().squeeze(-1).tolist()
                print(f"  A2[chunk={c}, head={h}]: max_err={max_err:.4f}, bad_rows={bad_rows[:20]}")
                # Show first bad row details
                if bad_rows:
                    r = bad_rows[0]
                    print(f"    row {r}: actual[:5]={actual[r,:5].tolist()}, expected[:5]={expected[r,:5].tolist()}")
            else:
                print(f"  A2[chunk={c}, head={h}]: OK (max_err={max_err:.6f})")

    print("\n=== Checking workspace_a1 (A * exp(g) * beta) ===")
    for c in range(total_chunks):
        for h in range(H):
            actual = workspace_a1[c, h].cpu().float()
            expected = ref_a1[c, h].cpu().float()
            diff = (actual - expected).abs()
            max_err = diff.max().item()
            if max_err > 0.01:
                bad_rows = (diff.max(dim=1).values > 0.01).nonzero().squeeze(-1).tolist()
                print(f"  A1[chunk={c}, head={h}]: max_err={max_err:.4f}, bad_rows={bad_rows[:20]}")
            else:
                print(f"  A1[chunk={c}, head={h}]: OK (max_err={max_err:.6f})")


if __name__ == "__main__":
    main()
