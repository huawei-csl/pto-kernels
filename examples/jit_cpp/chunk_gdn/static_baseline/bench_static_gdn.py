#!/usr/bin/env python3
"""
Benchmark static PTO kernels (bisheng-compiled ``*_kernel.cpp``, ctypes) with the same
shape and op model as ``tilelang_codegen/bench_tilelang_gdn.py``.

Stream handle is obtained once per run; it is not recomputed inside timed regions.
"""
from __future__ import annotations

import ctypes
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CHUNK_GDN = os.path.abspath(os.path.join(_HERE, ".."))
if _CHUNK_GDN not in sys.path:
    sys.path.insert(0, _CHUNK_GDN)

import torch
import torch.nn.functional as F

import pto_static_common  # noqa: F401 — ASCEND_* env
from gdn_bench_common import (
    KERNEL_ORDER,
    approx_ops_gdn,
    do_bench,
    format_ms,
    format_ops,
    format_tflops,
)
from static_kernel_libs import (
    lib_chunk_cumsum,
    lib_chunk_h,
    lib_chunk_o,
    lib_scaled_dot_kkt,
    lib_wy_fast,
)

NPU_DEVICE = os.getenv("GDN_TRI_INVERSE_NPU_DEVICE", "npu:0")


def vp(p) -> ctypes.c_void_p:
    return ctypes.c_void_p(p)


def bench_stage(name: str, fn) -> float:
    import torch_npu

    print(f"[bench] {name}")
    fn()
    torch_npu.npu.synchronize()
    ms = do_bench(fn)
    print(f"[bench-ok] {name}: {ms:.2f} ms")
    return ms


def main():
    torch.manual_seed(0)
    torch.npu.set_device(NPU_DEVICE)

    B, H, L, DK, DV, BK, BV = 16, 16, 16384, 128, 128, 128, 128
    C = 128
    CHUNK_NUM = (L + C - 1) // C
    BV_NUM = (DV + BV - 1) // BV
    nblk = B * H * CHUNK_NUM

    assert H % 2 == 0
    assert L % C == 0
    assert L % (8 * C) == 0

    # One stream handle for all kernel launches (do not call current_stream inside timed fn).
    stream = torch.npu.current_stream()._as_parameter_

    l_cumsum = lib_chunk_cumsum()
    l_kkt = lib_scaled_dot_kkt()
    l_wy = lib_wy_fast()
    l_h = lib_chunk_h()
    l_o = lib_chunk_o()

    q = torch.randn((B, H, L, DK), device="npu", dtype=torch.float16)
    k = torch.randn((B, H, L, DK), device="npu", dtype=torch.float16)
    v = torch.randn((B, H, L, DV), device="npu", dtype=torch.float16)
    q, k = F.normalize(q, dim=-1, p=2), F.normalize(k, dim=-1, p=2)
    g = torch.randn((B, H, L), device="npu", dtype=torch.float32)
    g = F.logsigmoid(g)
    beta = torch.rand((B, H, L), device="npu", dtype=torch.float16)

    g_sum = torch.empty((B, H, L), device="npu", dtype=torch.float32)
    msk1 = torch.tril(torch.ones((C, C), device="npu"), diagonal=-1).to(torch.float32)
    workspace_kkt = torch.zeros((B, H, L, C), device="npu", dtype=torch.float16)
    a_raw = torch.empty((B, H, L, C), device="npu", dtype=torch.float16)

    workspace_a1 = torch.zeros((B, H, L, C), device="npu", dtype=torch.float16)
    workspace_a2 = torch.zeros((B, H, L, C), device="npu", dtype=torch.float16)
    w = torch.empty((B, H, L, DK), device="npu", dtype=torch.float16)
    u = torch.empty((B, H, L, DV), device="npu", dtype=torch.float16)

    workspace_1 = torch.zeros((B * H * BV_NUM, C, DV), device="npu", dtype=torch.float16)
    workspace_2 = torch.zeros((B * H * BV_NUM, C, DK), device="npu", dtype=torch.float16)
    workspace_3 = torch.zeros((B * H * BV_NUM, DK, DV), device="npu", dtype=torch.float16)
    workspace_4 = torch.zeros((B * H * BV_NUM, DK, DV), device="npu", dtype=torch.float16)
    s = torch.zeros((B, H, CHUNK_NUM, DK, DV), device="npu", dtype=torch.float16)
    nv = torch.empty((B, H, L, DV), device="npu", dtype=torch.float16)
    fs = torch.empty((B, H, DK, DV), device="npu", dtype=torch.float16)

    workspace_o1 = torch.zeros((nblk, C, C), device="npu", dtype=torch.float16)
    workspace_o2 = torch.zeros((nblk, C, DV), device="npu", dtype=torch.float16)
    workspace_o3 = torch.zeros((nblk, C, C), device="npu", dtype=torch.float16)
    msk2 = torch.tril(torch.ones((C, C), device="npu"), diagonal=0).to(torch.float32)
    o = torch.empty((B, H, L, DV), device="npu", dtype=torch.float16)

    print()
    print(f"Shape: (B,H,L,DK,DV,C)=({B},{H},{L},{DK},{DV},{C}) (static PTO kernels)")

    l_cumsum.call(vp(g.data_ptr()), vp(g_sum.data_ptr()), stream)
    l_kkt.call(
        vp(k.data_ptr()),
        vp(beta.data_ptr()),
        vp(g_sum.data_ptr()),
        vp(msk1.data_ptr()),
        vp(workspace_kkt.data_ptr()),
        vp(a_raw.data_ptr()),
        stream,
    )
    l_wy.call(
        vp(k.data_ptr()),
        vp(v.data_ptr()),
        vp(beta.data_ptr()),
        vp(g_sum.data_ptr()),
        vp(a_raw.data_ptr()),
        vp(workspace_a1.data_ptr()),
        vp(workspace_a2.data_ptr()),
        vp(w.data_ptr()),
        vp(u.data_ptr()),
        stream,
    )
    l_h.call(
        vp(k.data_ptr()),
        vp(w.data_ptr()),
        vp(u.data_ptr()),
        vp(g_sum.data_ptr()),
        vp(workspace_1.data_ptr()),
        vp(workspace_2.data_ptr()),
        vp(workspace_3.data_ptr()),
        vp(workspace_4.data_ptr()),
        vp(s.data_ptr()),
        vp(nv.data_ptr()),
        vp(fs.data_ptr()),
        stream,
    )
    l_o.call(
        vp(q.data_ptr()),
        vp(k.data_ptr()),
        vp(nv.data_ptr()),
        vp(s.data_ptr()),
        vp(g_sum.data_ptr()),
        vp(msk2.data_ptr()),
        vp(workspace_o1.data_ptr()),
        vp(workspace_o2.data_ptr()),
        vp(workspace_o3.data_ptr()),
        vp(o.data_ptr()),
        stream,
    )
    torch.npu.synchronize()

    latencies = {
        "chunk_cumsum": bench_stage(
            "chunk_cumsum",
            lambda: l_cumsum.call(
                vp(g.data_ptr()), vp(g_sum.data_ptr()), stream
            ),
        ),
        "chunk_scaled_dot_kkt": bench_stage(
            "chunk_scaled_dot_kkt",
            lambda: l_kkt.call(
                vp(k.data_ptr()),
                vp(beta.data_ptr()),
                vp(g_sum.data_ptr()),
                vp(msk1.data_ptr()),
                vp(workspace_kkt.data_ptr()),
                vp(a_raw.data_ptr()),
                stream,
            ),
        ),
        "wy_fast": bench_stage(
            "wy_fast",
            lambda: l_wy.call(
                vp(k.data_ptr()),
                vp(v.data_ptr()),
                vp(beta.data_ptr()),
                vp(g_sum.data_ptr()),
                vp(a_raw.data_ptr()),
                vp(workspace_a1.data_ptr()),
                vp(workspace_a2.data_ptr()),
                vp(w.data_ptr()),
                vp(u.data_ptr()),
                stream,
            ),
        ),
        "chunk_h": bench_stage(
            "chunk_h",
            lambda: l_h.call(
                vp(k.data_ptr()),
                vp(w.data_ptr()),
                vp(u.data_ptr()),
                vp(g_sum.data_ptr()),
                vp(workspace_1.data_ptr()),
                vp(workspace_2.data_ptr()),
                vp(workspace_3.data_ptr()),
                vp(workspace_4.data_ptr()),
                vp(s.data_ptr()),
                vp(nv.data_ptr()),
                vp(fs.data_ptr()),
                stream,
            ),
        ),
        "chunk_o": bench_stage(
            "chunk_o",
            lambda: l_o.call(
                vp(q.data_ptr()),
                vp(k.data_ptr()),
                vp(nv.data_ptr()),
                vp(s.data_ptr()),
                vp(g_sum.data_ptr()),
                vp(msk2.data_ptr()),
                vp(workspace_o1.data_ptr()),
                vp(workspace_o2.data_ptr()),
                vp(workspace_o3.data_ptr()),
                vp(o.data_ptr()),
                stream,
            ),
        ),
    }

    ops = {name: approx_ops_gdn(B, H, L, DK, DV, C)[name] for name in KERNEL_ORDER}
    total_ms = sum(latencies[name] for name in KERNEL_ORDER)
    total_ops = sum(ops[name] for name in KERNEL_ORDER)

    print()
    print(f"Shape: (B,H,L,DK,DV,C)=({B},{H},{L},{DK},{DV},{C})")
    print("| Kernel | Latency (ms) | #ops (approx) | TFLOPS |")
    print("| :-- | --: | --: | --: |")
    for name in KERNEL_ORDER:
        print(
            f"| {name} | {format_ms(latencies[name])} | {format_ops(ops[name])} | "
            f"{format_tflops(ops[name], latencies[name])} |"
        )
    print(
        f"| total | {format_ms(total_ms)} | {format_ops(total_ops)} | "
        f"{format_tflops(total_ops, total_ms)} |"
    )


if __name__ == "__main__":
    main()
