"""
Numerical checks for vLLM FLA Triton GDN kernels on NPU (varlen ``cu_seqlens``).

1. ``chunk_local_cumsum`` / ``chunk_scaled_dot_kkt_fwd`` vs PyTorch refs in ``refs_bthd.py``.
2. ``recompute_w_u_fwd`` vs a loop reference matching the Triton math.
3. End-to-end smoke: manual forward with ``solve_tril`` then ``chunk_h`` + ``chunk_o``; assert finite outputs.

Kernels are vendored in ``fla_vendor/`` (see ``fla_vendor/SOURCES.md``). The timed benchmark omits ``solve_tril``;
this script runs it for stage (3).

Environment: run from ``chunk_gdn`` on ``PYTHONPATH`` (see README) so ``triton_baseline`` imports resolve.
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_CHUNK_GDN = os.path.dirname(_ROOT)
if _CHUNK_GDN not in sys.path:
    sys.path.insert(0, _CHUNK_GDN)

import torch
import torch.nn.functional as F

from triton_baseline.fla_vendor.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from triton_baseline.fla_vendor.chunk_o import chunk_fwd_o
from triton_baseline.fla_vendor.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from triton_baseline.fla_vendor.cumsum import chunk_local_cumsum
from triton_baseline.fla_vendor.solve_tril import solve_tril
from triton_baseline.fla_vendor.utils import prepare_chunk_indices, prepare_chunk_offsets
from triton_baseline.fla_vendor.wy_fast import recompute_w_u_fwd

from triton_baseline.refs_bthd import ref_chunk_local_cumsum, ref_scaled_dot_kkt_bthd

NPU_DEVICE = os.getenv("GDN_TRITON_NPU_DEVICE", "npu:0")
CHUNK_SIZE = 64
RTOL, ATOL = 1e-2, 1e-5


def ref_recompute_w_u(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g_cumsum: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, Kdim = k.shape
    V = v.shape[-1]
    w_ref = torch.zeros(B, T, H, Kdim, device=k.device, dtype=torch.float32)
    u_ref = torch.zeros(B, T, H, V, device=k.device, dtype=torch.float32)
    kf, vf, bf = k.float(), v.float(), beta.float()
    Af, gf = A.float(), g_cumsum.float()
    cu = cu_seqlens.cpu().tolist()
    for i in range(len(cu) - 1):
        bos, eos = cu[i], cu[i + 1]
        for s in range(bos, eos - (eos - bos) % chunk_size, chunk_size):
            e = s + chunk_size
            for h in range(H):
                Ablk = Af[0, s:e, h, :]
                gc = gf[0, s:e, h]
                b_g = torch.exp(gc)
                vb = vf[0, s:e, h, :] * bf[0, s:e, h, None]
                kb = kf[0, s:e, h, :] * bf[0, s:e, h, None] * b_g[:, None]
                u_ref[0, s:e, h, :] = Ablk @ vb
                w_ref[0, s:e, h, :] = Ablk @ kb
    return w_ref.to(k.dtype), u_ref.to(v.dtype)


def main():
    torch.manual_seed(1)
    torch.npu.set_device(NPU_DEVICE)
    dev = torch.device(NPU_DEVICE)

    N_seq, L_seg = 2, 128
    H, DK, DV = 4, 32, 32
    T = N_seq * L_seg
    cu_seqlens = torch.arange(0, T + 1, L_seg, dtype=torch.long, device=dev)
    chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, CHUNK_SIZE)

    q = torch.randn(1, T, H, DK, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, T, H, DK, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, T, H, DV, device=dev, dtype=torch.bfloat16)
    g_in = F.logsigmoid(torch.randn(1, T, H, device=dev, dtype=torch.float32))
    beta = torch.rand(1, T, H, device=dev, dtype=torch.bfloat16)
    initial_state = torch.zeros(N_seq, H, DK, DV, device=dev, dtype=torch.bfloat16)
    scale = DK**-0.5

    g_tr = chunk_local_cumsum(g_in, chunk_size=CHUNK_SIZE, cu_seqlens=cu_seqlens)
    g_cpu = ref_chunk_local_cumsum(g_in.detach().cpu(), CHUNK_SIZE, cu_seqlens.cpu())
    assert torch.allclose(g_tr.float().cpu(), g_cpu, rtol=RTOL, atol=ATOL), "chunk_local_cumsum"

    A_tr = chunk_scaled_dot_kkt_fwd(
        k=k,
        beta=beta,
        g_cumsum=g_tr,
        cu_seqlens=cu_seqlens,
        output_dtype=torch.float32,
    )
    A_ref = ref_scaled_dot_kkt_bthd(
        k.detach().cpu(),
        beta.detach().cpu(),
        g_tr.detach().cpu(),
        CHUNK_SIZE,
        cu_seqlens.cpu(),
    )
    assert torch.allclose(A_tr.float().cpu(), A_ref, rtol=RTOL, atol=ATOL), "chunk_scaled_dot_kkt_fwd"

    w_tr, u_tr = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A_tr,
        g_cumsum=g_tr,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    w_ref, u_ref = ref_recompute_w_u(
        k.cpu(), v.cpu(), beta.cpu(), A_tr.cpu(), g_tr.cpu(), cu_seqlens.cpu(), CHUNK_SIZE
    )
    w_ref, u_ref = w_ref.to(dev), u_ref.to(dev)
    assert torch.allclose(w_tr.float(), w_ref.float(), rtol=RTOL, atol=ATOL), "recompute_w_u_fwd w"
    assert torch.allclose(u_tr.float(), u_ref.float(), rtol=RTOL, atol=ATOL), "recompute_w_u_fwd u"

    # --- Full forward with solve_tril (smoke: finite outputs) ---
    A_s = solve_tril(A=A_tr, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
    w2, u2 = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A_s,
        g_cumsum=g_tr,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    h_m, v_new_m, _ = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w2,
        u=u2,
        g=g_tr,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )
    o_m = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new_m,
        h=h_m,
        g=g_tr,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )

    assert torch.isfinite(o_m).all(), "chunk_fwd_o output"
    assert torch.isfinite(h_m).all(), "chunk_gated_delta_rule_fwd_h h"
    assert torch.isfinite(v_new_m).all(), "chunk_gated_delta_rule_fwd_h v_new"

    print("verify_triton_gdn_kernels: all checks passed.")


if __name__ == "__main__":
    main()
