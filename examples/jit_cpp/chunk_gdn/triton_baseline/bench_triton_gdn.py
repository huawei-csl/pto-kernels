"""
NPU benchmark for vLLM-Ascend FLA Triton GDN stages (no ``solve_tril``), mirroring
``tilelang_codegen/bench_tilelang_gdn.py``. Tensors use native layout ``[B, T, H, …]``
(batch, sequence, head); varlen is emulated with stepped ``cu_seqlens`` (``B`` must be 1).

Timing uses :func:`gdn_bench_common.do_bench_triton` (``end.synchronize()`` on events).

Triton kernels are vendored under ``triton_baseline/fla_vendor/`` (see ``fla_vendor/SOURCES.md``).
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
from triton_baseline.fla_vendor.utils import prepare_chunk_indices, prepare_chunk_offsets
from triton_baseline.fla_vendor.wy_fast import recompute_w_u_fwd

from gdn_bench_common import (
    KERNEL_ORDER,
    approx_ops_gdn_triton,
    do_bench_triton,
    format_ms,
    format_ops,
    format_tflops,
)

NPU_DEVICE = os.getenv("GDN_TRITON_NPU_DEVICE", "npu:0")
CHUNK_SIZE = 64


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
    ms = do_bench_triton(fn)
    print(f"[bench-ok] {name}: {ms:.2f} ms")
    return ms


def main():
    torch.manual_seed(0)
    torch.npu.set_device(NPU_DEVICE)

    # Match total tokens with tilelang default: B_tile * L_tile = 16 * 16384 = 262144
    N_seq = int(os.getenv("GDN_TRITON_N_SEQ", "16"))
    L_seg = int(os.getenv("GDN_TRITON_L_SEG", "16384"))
    H = int(os.getenv("GDN_TRITON_H", "16"))
    DK = int(os.getenv("GDN_TRITON_DK", "128"))
    DV = int(os.getenv("GDN_TRITON_DV", "128"))

    T = N_seq * L_seg
    assert L_seg % CHUNK_SIZE == 0, "each segment length must be divisible by 64"
    assert T % CHUNK_SIZE == 0

    dev = torch.device(NPU_DEVICE)
    cu_seqlens = torch.arange(0, T + 1, L_seg, dtype=torch.long, device=dev)

    chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, CHUNK_SIZE)

    q = torch.randn(1, T, H, DK, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, T, H, DK, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, T, H, DV, device=dev, dtype=torch.bfloat16)
    q, k = F.normalize(q, dim=-1, p=2), F.normalize(k, dim=-1, p=2)
    g_in = torch.randn(1, T, H, device=dev, dtype=torch.float32)
    g_in = F.logsigmoid(g_in)
    beta = torch.rand(1, T, H, device=dev, dtype=torch.bfloat16)
    initial_state = torch.zeros(N_seq, H, DK, DV, device=dev, dtype=torch.bfloat16)

    scale = DK**-0.5
    BT = CHUNK_SIZE

    ops = {name: approx_ops_gdn_triton(1, H, T, DK, DV, BT)[name] for name in KERNEL_ORDER}

    print()
    print(
        f"Shape (packed): B=1, T={T}, H={H}, DK={DK}, DV={DV}; "
        f"varlen cu_seqlens step {L_seg} ({N_seq} segments). "
        f"Triton chunk tile BT={BT}."
    )

    g_sum = run_stage(
        "chunk_cumsum",
        lambda: chunk_local_cumsum(
            g_in,
            chunk_size=CHUNK_SIZE,
            cu_seqlens=cu_seqlens,
        ),
    )
    a_raw = run_stage(
        "chunk_scaled_dot_kkt",
        lambda: chunk_scaled_dot_kkt_fwd(
            k=k,
            beta=beta,
            g_cumsum=g_sum,
            cu_seqlens=cu_seqlens,
            output_dtype=torch.float32,
        ),
    )
    w, u = run_stage(
        "wy_fast",
        lambda: recompute_w_u_fwd(
            k=k,
            v=v,
            beta=beta,
            A=a_raw,
            g_cumsum=g_sum,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        ),
    )
    h, v_new, _ = run_stage(
        "chunk_h",
        lambda: chunk_gated_delta_rule_fwd_h(
            k=k,
            w=w,
            u=u,
            g=g_sum,
            initial_state=initial_state,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
        ),
    )
    run_stage(
        "chunk_o",
        lambda: chunk_fwd_o(
            q=q,
            k=k,
            v=v_new,
            h=h,
            g=g_sum,
            scale=scale,
            cu_seqlens=cu_seqlens,
        ),
    )

    latencies = {
        "chunk_cumsum": bench_stage(
            "chunk_cumsum",
            lambda: chunk_local_cumsum(
                g_in,
                chunk_size=CHUNK_SIZE,
                cu_seqlens=cu_seqlens,
            ),
        ),
        "chunk_scaled_dot_kkt": bench_stage(
            "chunk_scaled_dot_kkt",
            lambda: chunk_scaled_dot_kkt_fwd(
                k=k,
                beta=beta,
                g_cumsum=g_sum,
                cu_seqlens=cu_seqlens,
                output_dtype=torch.float32,
            ),
        ),
        "wy_fast": bench_stage(
            "wy_fast",
            lambda: recompute_w_u_fwd(
                k=k,
                v=v,
                beta=beta,
                A=a_raw,
                g_cumsum=g_sum,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            ),
        ),
        "chunk_h": bench_stage(
            "chunk_h",
            lambda: chunk_gated_delta_rule_fwd_h(
                k=k,
                w=w,
                u=u,
                g=g_sum,
                initial_state=initial_state,
                output_final_state=False,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                chunk_offsets=chunk_offsets,
            ),
        ),
        "chunk_o": bench_stage(
            "chunk_o",
            lambda: chunk_fwd_o(
                q=q,
                k=k,
                v=v_new,
                h=h,
                g=g_sum,
                scale=scale,
                cu_seqlens=cu_seqlens,
            ),
        ),
    }

    total_ms = sum(latencies[name] for name in KERNEL_ORDER)
    total_ops = sum(ops[name] for name in KERNEL_ORDER)

    print()
    print("| Kernel | Latency (ms) | #ops (approx) | TFLOPS |")
    print("| :-- | --: | --: | --: |")
    for name in KERNEL_ORDER:
        print(
            f"| {name} | {format_ms(latencies[name])} | {format_ops(ops[name])} | "
            f"{format_tflops(ops[name], latencies[name])} |"
        )
    print(
        f"| total (no solve_tril) | {format_ms(total_ms)} | {format_ops(total_ops)} | "
        f"{format_tflops(total_ops, total_ms)} |"
    )


if __name__ == "__main__":
    main()
