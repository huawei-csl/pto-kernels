"""
End-to-end NPU benchmark for TileLang kernels in `kernels/`, matching the methodology
of gdn-tri-inverse/profiling/bench_tilelang_full_gdn.py (TFLOPs from approximate op
counts and measured latency). The triangular solve stage is omitted — it is not part
of this tilelang_codegen package.

Default shape matches `tilelang-ascend/examples/linear_attention_and_rnn/README.md`
(GDN “Optimize Results”): (B,H,L,DK,DV,C)=(16,16,16384,128,128,128). Approximate op
counts follow that README; `chunk_o` uses `5 * B * H * L * DK * DV` (same as the README
table’s ~3.44e11 ops), not `B*H*L*(C*DK+DK*DV+C*DV)`.

`do_bench` uses elapsed time in milliseconds (`unit="ms"`) so latency labels and the
TFLOPS formula `ops / (latency_ms * 1e9)` stay consistent (the upstream script
defaults to microseconds but prints “ms”, which skews TFLOPS).
"""
from __future__ import annotations

import os
import sys
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_CHUNK_GDN = os.path.dirname(_ROOT)
if _CHUNK_GDN not in sys.path:
    sys.path.insert(0, _CHUNK_GDN)

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

from kernels.opt_gdn_chunk_cumsum import cumsum_ker
from kernels.opt_gdn_chunk_h import chunk_h_ker
from kernels.opt_gdn_chunk_o import chunk_o_ker
from kernels.opt_gdn_chunk_scaled_dot_kkt import kkt_ker
from kernels.opt_gdn_wy_fast import wy_fast_ker

NPU_DEVICE = os.getenv("GDN_TRI_INVERSE_NPU_DEVICE", "npu:0")

# Latency (ms) from tilelang-ascend/examples/linear_attention_and_rnn/README.md (Optimize Results).
REF_README_MS = {
    "chunk_cumsum": 1.93,
    "chunk_scaled_dot_kkt": 8.76,
    "solve_tril": 24.89,
    "wy_fast": 9.92,
    "chunk_h": 9.38,
    "chunk_o": 13.19,
}


def run_stage(name: str, fn):
    print(f"[run] {name}")
    out = fn()
    torch.npu.synchronize()
    print(f"[ok] {name}")
    return out


def bench_stage(name: str, fn) -> float:
    print(f"[bench] {name}")
    fn()
    torch.npu.synchronize()
    ms = do_bench(fn)
    print(f"[bench-ok] {name}: {ms:.2f} ms")
    return ms


def main():
    torch.manual_seed(0)
    torch.npu.set_device(NPU_DEVICE)

    # Same shape as tilelang-ascend/examples/linear_attention_and_rnn/README.md (GDN Optimize Results).
    B, H, L, DK, DV, BK, BV = 16, 16, 16384, 128, 128, 128, 128
    C = 128

    ops_base = approx_ops_gdn(B, H, L, DK, DV, C)
    print(
        "Reference TFLOPS from README latencies (same #ops formulas as that README; "
        "should match its per-kernel TFLOPS column within rounding):"
    )
    print("| Kernel | README ms | #ops (approx) | TFLOPS |")
    print("| :-- | --: | --: | --: |")
    for name in KERNEL_ORDER:
        o = ops_base[name]
        ms = REF_README_MS[name]
        print(f"| {name} | {ms:.2f} | {format_ops(o)} | {format_tflops(o, ms)} |")
    total_ref_ms = sum(REF_README_MS[n] for n in KERNEL_ORDER)
    total_ref_ops = sum(ops_base[n] for n in KERNEL_ORDER)
    print(
        f"| total (5 kernels, no solve_tril) | {total_ref_ms:.2f} | "
        f"{format_ops(total_ref_ops)} | {format_tflops(total_ref_ops, total_ref_ms)} |"
    )
    readme_6way_ms = sum(REF_README_MS[n] for n in REF_README_MS)
    readme_6way_ops = sum(ops_base[n] for n in KERNEL_ORDER) + ops_base["solve_tril"]
    print(
        f"README 6-kernel total (includes solve_tril): {readme_6way_ms:.2f} ms, "
        f"{format_ops(readme_6way_ops)} ops, "
        f"{format_tflops(readme_6way_ops, readme_6way_ms)} TFLOPS (cf. README ~68.07 ms, "
        f"~8.48e11 ops, ~12.45 TFLOPS)."
    )
    print()

    assert H % 2 == 0, "optimized kernels assume even H"
    assert L % C == 0, "optimized kernels assume full chunks"
    assert L % (8 * C) == 0, "opt_gdn_chunk_cumsum assumes L % (8 * C) == 0"

    q = torch.randn((B, H, L, DK)).npu().to(torch.float16)
    k = torch.randn((B, H, L, DK)).npu().to(torch.float16)
    v = torch.randn((B, H, L, DV)).npu().to(torch.float16)
    q, k = F.normalize(q, dim=-1, p=2), F.normalize(k, dim=-1, p=2)
    g = torch.randn((B, H, L)).npu().to(torch.float)
    g = F.logsigmoid(g)
    beta = torch.rand((B, H, L)).npu().to(torch.float16)

    ker1 = cumsum_ker(B, H, L, C)
    ker2 = kkt_ker(B, H, L, DK, C, BK)
    ker4 = wy_fast_ker(B, H, L, DK, DV, C, BK, BV)
    ker5 = chunk_h_ker(B, H, L, DK, DV, C, BK, BV)
    ker6 = chunk_o_ker(B, H, L, DK, DV, C, BK, BV)

    msk1 = torch.tril(torch.ones((C, C)), diagonal=-1).npu().to(torch.float)
    msk2 = torch.tril(torch.ones((C, C)), diagonal=0).npu().to(torch.float)
    workspace = (
        torch.zeros((B * H * ((DV + BV - 1) // BV), DK, BV)).npu().to(torch.float16)
    )
    s = torch.zeros((B, H, (L + C - 1) // C, DK, DV)).npu().to(torch.float16)

    print()
    print(f"Shape: (B,H,L,DK,DV,C)=({B},{H},{L},{DK},{DV},{C})")

    g_sum = run_stage("chunk_cumsum", lambda: ker1(g))
    a_raw = run_stage("chunk_scaled_dot_kkt", lambda: ker2(k, beta, g_sum, msk1))
    # No solve_tril in this package: feed KKT output directly into wy_fast.
    w, u = run_stage("wy_fast", lambda: ker4(k, v, beta, g_sum, a_raw))
    nv, _ = run_stage("chunk_h", lambda: ker5(k, w, u, g_sum, workspace, s))
    run_stage("chunk_o", lambda: ker6(q, k, nv, s, g_sum, msk2))

    latencies = {
        "chunk_cumsum": bench_stage("chunk_cumsum", lambda: ker1(g)),
        "chunk_scaled_dot_kkt": bench_stage(
            "chunk_scaled_dot_kkt", lambda: ker2(k, beta, g_sum, msk1)
        ),
        "wy_fast": bench_stage(
            "wy_fast", lambda: ker4(k, v, beta, g_sum, a_raw)
        ),
        "chunk_h": bench_stage(
            "chunk_h", lambda: ker5(k, w, u, g_sum, workspace, s)
        ),
        "chunk_o": bench_stage(
            "chunk_o", lambda: ker6(q, k, nv, s, g_sum, msk2)
        ),
    }

    ops = {name: approx_ops_gdn(B, H, L, DK, DV, C)[name] for name in KERNEL_ORDER}

    total_ms = sum(latencies[name] for name in KERNEL_ORDER)
    total_ops = sum(ops[name] for name in KERNEL_ORDER)

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
