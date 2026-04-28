from __future__ import annotations
import math
import torch
import pto_dynamic_common  # noqa: F401
from dynamic_kernel_libs import (
    pack_bsh_tensor,
    pack_bshd_tensor,
    wy_fast_kernel,
)
from pto_dynamic_common import torch_to_ctypes, optional_torch_to_ctypes, BLOCK_DIM


torch_npu = torch.npu
CHUNK = 128


def main():
    torch.manual_seed(42)
    torch.npu.set_device("npu:0")

    # Test with g=0 so exp(g)=1, making A1 == A2
    # Also use identity-like A to isolate scaling
    B, S, H, D = 1, 128, 2, 128
    total_chunks = B * (S // CHUNK)
    
    k = torch.randn((B, S, H, D), device="npu", dtype=torch.float16)
    v = torch.randn((B, S, H, D), device="npu", dtype=torch.float16)
    beta = torch.ones((B, S, H), device="npu", dtype=torch.float16)
    g_packed = torch.zeros((total_chunks, H, CHUNK), device="npu", dtype=torch.float32)
    
    # Use identity A: a_packed[chunk, head, i, j] = 1 if i==j else 0
    a_packed = torch.zeros((total_chunks, H, CHUNK, CHUNK), device="npu", dtype=torch.float16)
    for c in range(total_chunks):
        for h in range(H):
            a_packed[c, h] = torch.eye(CHUNK, device="npu", dtype=torch.float16)

    w_out = torch.zeros((total_chunks, H, CHUNK, D), device="npu", dtype=torch.float16)
    u_out = torch.zeros_like(w_out)

    # Reference: A1 = A * beta * exp(g) = I * 1 * 1 = I
    # w = I @ k_packed = k_packed
    # u = I @ v_packed = v_packed
    k_packed = pack_bshd_tensor(k, chunk_size=CHUNK).to(torch.float16)
    v_packed = pack_bshd_tensor(v, chunk_size=CHUNK).to(torch.float16)

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

    # A1 and A2 should both be I (identity)
    print("=== Workspace A2 (should be identity) ===")
    for c in range(total_chunks):
        for h in range(H):
            actual = workspace_a2[c, h].cpu()
            expected = torch.eye(CHUNK, dtype=torch.float16)
            diff = (actual.float() - expected.float()).abs()
            max_err = diff.max().item()
            bad_rows = (diff.max(dim=1).values > 0.01).nonzero().squeeze(-1).tolist()
            if max_err > 0.01:
                print(f"  A2[{c},{h}]: max_err={max_err:.4f}, bad_rows={bad_rows}")
                for r in bad_rows[:3]:
                    print(f"    row {r}: diag={actual[r,r].item():.4f}, should be 1.0")
                    # Check if row is all zero
                    nz = actual[r].abs().sum().item()
                    print(f"    row {r}: sum_abs={nz:.6f}")
            else:
                print(f"  A2[{c},{h}]: OK")

    # w_out should be k_packed, u_out should be v_packed
    print("\n=== w_out vs k_packed ===")
    w_diff = (w_out.cpu().float() - k_packed.cpu().float()).abs()
    print(f"max diff: {w_diff.max().item():.6f}")
    bad = (w_diff.max(dim=-1).values > 0.01)
    if bad.any():
        idxs = bad.nonzero()[:5]
        for idx in idxs:
            c, h, r = idx.tolist()
            print(f"  bad at [{c},{h},{r}]: actual[:3]={w_out[c,h,r,:3].cpu().tolist()}, expected[:3]={k_packed[c,h,r,:3].cpu().tolist()}")

    print("\n=== u_out vs v_packed ===")
    u_diff = (u_out.cpu().float() - v_packed.cpu().float()).abs()
    print(f"max diff: {u_diff.max().item():.6f}")


if __name__ == "__main__":
    main()
