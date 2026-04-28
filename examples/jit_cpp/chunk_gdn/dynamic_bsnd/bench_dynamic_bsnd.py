#!/usr/bin/env python3
"""
Benchmark dynamic BSND PTO kernels (bisheng-compiled, ctypes) for chunk GDN.

Uses the same timing infrastructure as bench_static_gdn.py and bench_triton_gdn.py.
"""
from __future__ import annotations

import ctypes
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CHUNK_GDN = os.path.abspath(os.path.join(_HERE, ".."))
_JIT_CPP = os.path.abspath(os.path.join(_CHUNK_GDN, ".."))
_FAST_INV = os.path.join(_JIT_CPP, "fast_inverse")
if _CHUNK_GDN not in sys.path:
    sys.path.insert(0, _CHUNK_GDN)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _FAST_INV not in sys.path:
    sys.path.insert(0, _FAST_INV)

import torch
import torch.nn.functional as F

from gdn_bench_common import (
    KERNEL_ORDER,
    approx_ops_gdn,
    do_bench,
    format_ms,
    format_ops,
    format_tflops,
)
from dynamic_kernel_libs import (
    BLOCK_DIM,
    _transpose_beta,
    _transpose_g,
    load_chunk_cumsum,
    load_chunk_h,
    load_chunk_o,
    load_scaled_dot_kkt,
    load_wy_fast,
    total_chunks,
)
from jit_util_fast_inverse import jit_compile

NPU_DEVICE = os.getenv("GDN_NPU_DEVICE", "npu:0")

KERNEL_ORDER_FULL = [
    "chunk_cumsum",
    "chunk_scaled_dot_kkt",
    "solve_tril",
    "wy_fast",
    "chunk_h",
    "chunk_o",
]


def _vp(t):
    return ctypes.c_void_p(t.data_ptr()) if t is not None else ctypes.c_void_p()


def bench_stage(name: str, fn) -> float:
    import torch_npu
    print(f"[bench] {name}")
    fn()
    torch_npu.npu.synchronize()
    ms = do_bench(fn)
    print(f"[bench-ok] {name}: {ms:.2f} ms")
    return ms


def _make_minus_identity(matrix_size: int, device: torch.device) -> torch.Tensor:
    minus_identity = torch.zeros(
        (matrix_size, matrix_size), dtype=torch.float16, device=device,
    )
    minus_identity.fill_diagonal_(-1)
    return minus_identity


def main():
    torch.manual_seed(0)
    torch.npu.set_device(NPU_DEVICE)
    dev = torch.device(NPU_DEVICE)

    if "PTO_LIB_PATH" not in os.environ:
        fb = "/sources/pto-isa"
        if os.path.isdir(os.path.join(fb, "include")):
            os.environ["PTO_LIB_PATH"] = fb

    N_seq = 16
    L_seg = 16384
    H, DK, DV = 16, 128, 128
    C = 128
    T = N_seq * L_seg

    cu_seqlens = torch.arange(0, T + 1, L_seg, dtype=torch.int32, device=dev)
    tc = total_chunks(N_seq, T, C, cu_seqlens)

    stream = torch.npu.current_stream()._as_parameter_
    bd = BLOCK_DIM

    l_cumsum = load_chunk_cumsum(H, C)
    l_kkt = load_scaled_dot_kkt(H, DK, C)
    l_wy = load_wy_fast(H, DK, C)
    l_h = load_chunk_h(H, DK, C)
    l_o = load_chunk_o(H, DK, C)

    cpp = os.path.join(_FAST_INV, "fast_inverse.cpp")
    print(f"Compiling fast_inverse: {cpp}")
    tri_inv = jit_compile(cpp, verbose=False)
    print("Compilation OK.")

    q = torch.randn(1, T, H, DK, device=dev, dtype=torch.float16)
    k = torch.randn(1, T, H, DK, device=dev, dtype=torch.float16)
    v = torch.randn(1, T, H, DV, device=dev, dtype=torch.float16)
    q, k = F.normalize(q, dim=-1, p=2), F.normalize(k, dim=-1, p=2)
    g = F.logsigmoid(torch.randn(1, T, H, device=dev, dtype=torch.float32))
    beta = torch.rand(1, T, H, device=dev, dtype=torch.float16)

    g_sum = torch.empty(1, T, H, device=dev, dtype=torch.float32)
    msk1 = torch.tril(torch.ones(C, C, device=dev), diagonal=-1).float()
    workspace_kkt = torch.zeros(bd * 2, C, C, device=dev, dtype=torch.float16)
    A = torch.empty(1, T, H, C, device=dev, dtype=torch.float16)

    num_matrices = tc * H
    A_sol_fp32 = torch.zeros(1, T, H, C, device=dev, dtype=torch.float32)
    A_sol = torch.empty(1, T, H, C, device=dev, dtype=torch.float16)
    minus_identity = _make_minus_identity(C, dev)

    workspace_a1 = torch.zeros(bd, C, C, device=dev, dtype=torch.float16)
    workspace_a2 = torch.zeros(bd, C, C, device=dev, dtype=torch.float16)
    w = torch.empty(1, T, H, DK, device=dev, dtype=torch.float16)
    u = torch.empty(1, T, H, DV, device=dev, dtype=torch.float16)

    workspace_h = torch.zeros(bd * 4, DK, DV, device=dev, dtype=torch.float16)
    s = torch.zeros(tc * H, DK, DV, device=dev, dtype=torch.float16)
    nv = torch.empty(1, T, H, DV, device=dev, dtype=torch.float16)
    fs = torch.empty(N_seq * H, DK, DV, device=dev, dtype=torch.float16)

    workspace_o1 = torch.zeros(bd, C, C, device=dev, dtype=torch.float16)
    workspace_o2 = torch.zeros(bd, C, DV, device=dev, dtype=torch.float16)
    workspace_o3 = torch.zeros(bd, C, C, device=dev, dtype=torch.float16)
    msk2 = torch.tril(torch.ones(C, C, device=dev), diagonal=0).float()
    o = torch.empty(1, T, H, DV, device=dev, dtype=torch.float16)

    cu_p = _vp(cu_seqlens)
    batch_arg = N_seq
    seq_arg = T

    l_cumsum.call_kernel(bd, stream, _vp(g), _vp(g_sum), cu_p, batch_arg, seq_arg)
    torch.npu.synchronize()
    g_t = _transpose_g(g_sum)
    beta_t = _transpose_beta(beta)

    l_kkt.call_kernel(bd, stream, _vp(k), _vp(beta_t), _vp(g_t), _vp(msk1),
                       _vp(workspace_kkt), _vp(A), cu_p, batch_arg, seq_arg, T)
    tri_inv(A_sol_fp32, A, minus_identity, C, num_matrices, H,
            cu_seqlens=cu_seqlens, block_dim=bd, is_lower=True)
    A_sol.copy_(A_sol_fp32.to(torch.float16))
    l_wy.call_kernel(bd, stream, _vp(k), _vp(v), _vp(beta_t), _vp(g_t), _vp(A_sol),
                      _vp(workspace_a1), _vp(workspace_a2), _vp(w), _vp(u),
                      cu_p, batch_arg, seq_arg, T)
    l_h.call_kernel(bd, stream, _vp(k), _vp(w), _vp(u), _vp(g_t),
                     _vp(s), _vp(nv), _vp(fs), _vp(workspace_h),
                     cu_p, batch_arg, seq_arg, T)
    l_o.call_kernel(bd, stream, _vp(q), _vp(k), _vp(nv), _vp(s), _vp(g_t),
                     _vp(msk2), _vp(workspace_o1), _vp(workspace_o2), _vp(workspace_o3),
                     _vp(o), cu_p, batch_arg, seq_arg, T)
    torch.npu.synchronize()

    print()
    print(f"Shape: (N_seq,L_seg,H,DK,DV,C)=({N_seq},{L_seg},{H},{DK},{DV},{C})")
    print(f"  B=1, T={T} (packed varlen BSND), BLOCK_DIM={bd}")
    print()

    B_equiv = N_seq

    latencies = {
        "chunk_cumsum": bench_stage(
            "chunk_cumsum",
            lambda: l_cumsum.call_kernel(bd, stream, _vp(g), _vp(g_sum), cu_p,
                                          batch_arg, seq_arg),
        ),
        "chunk_scaled_dot_kkt": bench_stage(
            "chunk_scaled_dot_kkt",
            lambda: l_kkt.call_kernel(bd, stream, _vp(k), _vp(beta_t), _vp(g_t),
                                       _vp(msk1), _vp(workspace_kkt), _vp(A),
                                       cu_p, batch_arg, seq_arg, T),
        ),
        "solve_tril": bench_stage(
            "solve_tril",
            lambda: tri_inv(A_sol_fp32, A, minus_identity, C, num_matrices, H,
                            cu_seqlens=cu_seqlens, block_dim=bd, is_lower=True),
        ),
        "wy_fast": bench_stage(
            "wy_fast",
            lambda: l_wy.call_kernel(bd, stream, _vp(k), _vp(v), _vp(beta_t),
                                      _vp(g_t), _vp(A_sol),
                                      _vp(workspace_a1), _vp(workspace_a2),
                                      _vp(w), _vp(u), cu_p, batch_arg, seq_arg, T),
        ),
        "chunk_h": bench_stage(
            "chunk_h",
            lambda: l_h.call_kernel(bd, stream, _vp(k), _vp(w), _vp(u), _vp(g_t),
                                     _vp(s), _vp(nv), _vp(fs), _vp(workspace_h),
                                     cu_p, batch_arg, seq_arg, T),
        ),
        "chunk_o": bench_stage(
            "chunk_o",
            lambda: l_o.call_kernel(bd, stream, _vp(q), _vp(k), _vp(nv), _vp(s),
                                     _vp(g_t), _vp(msk2),
                                     _vp(workspace_o1), _vp(workspace_o2),
                                     _vp(workspace_o3), _vp(o),
                                     cu_p, batch_arg, seq_arg, T),
        ),
    }

    ops = approx_ops_gdn(B_equiv, H, L_seg, DK, DV, C)
    total_summed_ms = sum(latencies[n] for n in KERNEL_ORDER_FULL)
    total_summed_ops = sum(ops[n] for n in KERNEL_ORDER_FULL)

    def _run_e2e():
        l_cumsum.call_kernel(bd, stream, _vp(g), _vp(g_sum), cu_p, batch_arg, seq_arg)
        l_kkt.call_kernel(bd, stream, _vp(k), _vp(beta_t), _vp(g_t), _vp(msk1),
                           _vp(workspace_kkt), _vp(A), cu_p, batch_arg, seq_arg, T)
        tri_inv(A_sol_fp32, A, minus_identity, C, num_matrices, H,
                cu_seqlens=cu_seqlens, block_dim=bd, is_lower=True)
        A_sol.copy_(A_sol_fp32.to(torch.float16))
        l_wy.call_kernel(bd, stream, _vp(k), _vp(v), _vp(beta_t), _vp(g_t), _vp(A_sol),
                          _vp(workspace_a1), _vp(workspace_a2), _vp(w), _vp(u),
                          cu_p, batch_arg, seq_arg, T)
        l_h.call_kernel(bd, stream, _vp(k), _vp(w), _vp(u), _vp(g_t),
                         _vp(s), _vp(nv), _vp(fs), _vp(workspace_h),
                         cu_p, batch_arg, seq_arg, T)
        l_o.call_kernel(bd, stream, _vp(q), _vp(k), _vp(nv), _vp(s), _vp(g_t),
                         _vp(msk2), _vp(workspace_o1), _vp(workspace_o2),
                         _vp(workspace_o3), _vp(o), cu_p, batch_arg, seq_arg, T)

    total_measured_ms = bench_stage("total_e2e", _run_e2e)

    print()
    print(f"Shape: (N_seq,L_seg,H,DK,DV,C)=({N_seq},{L_seg},{H},{DK},{DV},{C})")
    print("| Kernel | Latency (ms) | #ops (approx) | TFLOPS |")
    print("| :-- | --: | --: | --: |")
    for name in KERNEL_ORDER_FULL:
        print(
            f"| {name} | {format_ms(latencies[name])} | {format_ops(ops[name])} "
            f"| {format_tflops(ops[name], latencies[name])} |"
        )
    print(
        f"| **total_summed** | **{format_ms(total_summed_ms)}** | {format_ops(total_summed_ops)} "
        f"| {format_tflops(total_summed_ops, total_summed_ms)} |"
    )
    print(
        f"| **total_measured** | **{format_ms(total_measured_ms)}** | {format_ops(total_summed_ops)} "
        f"| {format_tflops(total_summed_ops, total_measured_ms)} |"
    )


if __name__ == "__main__":
    main()
