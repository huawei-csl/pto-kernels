#!/usr/bin/env python3
"""
Benchmark ``chunk_o`` group-value kernel (Hg key heads, H value heads).

Uses the same packed varlen shape as ``bench_dynamic_bsnd_groupvalue.py``
(``N_seq=16``, ``L_seg=16384``, ``T=262144``, ``D=128``). PTO ``chunk_o`` uses
``C=128``. FLA Triton ``chunk_fwd_o`` defaults to ``BT=64`` (``GDN_TRITON_CHUNK_O_CHUNK``):
Ascend JIT hits UB overflow compiling ``chunk_fwd_o`` at ``BT=128``. Warm up
``chunk_h`` (PTO ctypes, then Triton tensors), then time ``chunk_o`` / ``chunk_fwd_o``
only — same pattern as ``dynamic_bsnd/bench_dynamic_bsnd.py``.

Run from this directory so ``pto_dynamic_common`` resolves with ``key_heads``.

Usage::
  cd .../dynamic_bsnd_groupvalue
  python3 bench_chunk_o_groupvalue.py
"""
from __future__ import annotations

import ctypes
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CHUNK_GDN = os.path.abspath(os.path.join(_HERE, ".."))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _CHUNK_GDN not in sys.path:
    sys.path.insert(0, _CHUNK_GDN)

import importlib.util
import torch
import torch.nn.functional as F

_pc_path = os.path.join(_HERE, "pto_dynamic_common.py")
_spec_pc = importlib.util.spec_from_file_location(
    "pto_dynamic_common_groupvalue", _pc_path,
)
_pc_mod = importlib.util.module_from_spec(_spec_pc)
assert _spec_pc.loader is not None
_spec_pc.loader.exec_module(_pc_mod)
sys.modules["pto_dynamic_common"] = _pc_mod

_lib_here = os.path.join(_HERE, "dynamic_kernel_libs.py")
_spec_g = importlib.util.spec_from_file_location("dkgv_chunk_o", _lib_here)
dkgv_mod = importlib.util.module_from_spec(_spec_g)
assert _spec_g.loader is not None
_spec_g.loader.exec_module(dkgv_mod)
BLOCK_DIM = dkgv_mod.BLOCK_DIM
load_chunk_h_group = dkgv_mod.load_chunk_h
load_chunk_o_group = dkgv_mod.load_chunk_o
total_chunks = dkgv_mod.total_chunks

from gdn_bench_common import do_bench, do_bench_triton, format_ms


def _vp(t):
    return ctypes.c_void_p(t.data_ptr())


def _transpose_g(g_sum):
    return g_sum.squeeze(0).t().contiguous()


NPU_DEVICE = os.getenv("GDN_NPU_DEVICE", "npu:0")


def bench_chunk_o(lib_o, bd, stream, tensors, cu_p, batch_arg, seq_arg, T_val):
    q, k, nv, s, g_t, msk2, w1, w2, w3, o = tensors

    def fn():
        lib_o.call_kernel(
            bd,
            stream,
            _vp(q),
            _vp(k),
            _vp(nv),
            _vp(s),
            _vp(g_t),
            _vp(msk2),
            _vp(w1),
            _vp(w2),
            _vp(w3),
            _vp(o),
            cu_p,
            batch_arg,
            seq_arg,
            T_val,
        )

    fn()
    torch.npu.synchronize()
    return do_bench(fn)


def main():
    torch.manual_seed(0)
    torch.npu.set_device(NPU_DEVICE)
    dev = torch.device(NPU_DEVICE)

    N_seq = int(os.getenv("GDN_BENCH_N_SEQ", "16"))
    L_seg = int(os.getenv("GDN_BENCH_L_SEG", "16384"))
    DK = DV = 128
    C = 128
    H = int(os.getenv("GDN_BENCH_H", "32"))
    HG = int(os.getenv("GDN_BENCH_HG", "16"))
    assert H % HG == 0
    T = N_seq * L_seg

    cu_seqlens = torch.arange(0, T + 1, L_seg, dtype=torch.int32, device=dev)
    tc = total_chunks(N_seq, T, C, cu_seqlens)
    bd = BLOCK_DIM
    stream = torch.npu.current_stream()._as_parameter_
    cu_p = _vp(cu_seqlens)

    lib_h = load_chunk_h_group(H, DK, C, key_heads=HG)
    lib_o = load_chunk_o_group(H, DK, C, key_heads=HG)

    k_g = F.normalize(torch.randn(1, T, HG, DK, device=dev, dtype=torch.float16), dim=-1, p=2)
    q_g = F.normalize(torch.randn(1, T, HG, DK, device=dev, dtype=torch.float16), dim=-1, p=2)
    w_g = torch.randn(1, T, H, DK, device=dev, dtype=torch.float16)
    u_g = torch.randn(1, T, H, DV, device=dev, dtype=torch.float16)
    g_sum_g = torch.randn(1, T, H, device=dev, dtype=torch.float32)
    g_t_g = _transpose_g(g_sum_g)
    ws_h = torch.zeros(bd * 4, DK, DV, device=dev, dtype=torch.float16)
    s_g = torch.zeros(tc * H, DK, DV, device=dev, dtype=torch.float16)
    nv_g = torch.empty(1, T, H, DV, device=dev, dtype=torch.float16)
    fs_g = torch.empty(N_seq * H, DK, DV, device=dev, dtype=torch.float16)

    lib_h.call_kernel(
        bd,
        stream,
        _vp(k_g),
        _vp(w_g),
        _vp(u_g),
        _vp(g_t_g),
        _vp(s_g),
        _vp(nv_g),
        _vp(fs_g),
        _vp(ws_h),
        cu_p,
        N_seq,
        T,
        T,
    )
    torch.npu.synchronize()

    msk2 = torch.tril(torch.ones(C, C, device=dev), diagonal=0).float()
    w1 = torch.zeros(bd, C, C, device=dev, dtype=torch.float16)
    w2 = torch.zeros(bd, C, DV, device=dev, dtype=torch.float16)
    w3 = torch.zeros(bd, C, C, device=dev, dtype=torch.float16)
    o_g = torch.empty(1, T, H, DV, device=dev, dtype=torch.float16)

    ms_o = bench_chunk_o(
        lib_o,
        bd,
        stream,
        (q_g, k_g, nv_g, s_g, g_t_g, msk2, w1, w2, w3, o_g),
        cu_p,
        N_seq,
        T,
        T,
    )

    # Triton Ascend JIT fails ``chunk_fwd_o`` at ``BT=128`` (UB overflow on 910B2); vendor
    # benchmarks use ``chunk_size=64`` (see ``triton_baseline/bench_triton_gdn.py``). We time
    # Triton with ``C_TRITON`` for both ``chunk_gated_delta_rule_fwd_h`` and ``chunk_fwd_o``.
    C_triton = int(os.getenv("GDN_TRITON_CHUNK_O_CHUNK", "64"))

    # Triton FLA ``chunk_fwd_o`` (``triton_baseline/fla_vendor/chunk_o.py``) — same GQA
    # indexing as ``chunk_h`` (`i_h // (H // Hg)` for Q/K). Time only ``chunk_fwd_o``;
    # run vendor ``chunk_gated_delta_rule_fwd_h`` once first so ``h`` / ``v_new`` exist.
    ms_triton_o = None
    try:
        sys.path.insert(0, os.path.join(_CHUNK_GDN, "triton_baseline"))
        from fla_vendor.chunk_delta_h import chunk_gated_delta_rule_fwd_h
        from fla_vendor.chunk_o import chunk_fwd_o
        from fla_vendor.utils import prepare_chunk_indices, prepare_chunk_offsets

        cu_long = cu_seqlens.long()
        chunk_indices = prepare_chunk_indices(cu_long, C_triton)
        chunk_offsets = prepare_chunk_offsets(cu_long, C_triton)
        scale = DK**-0.5
        q_tr = F.normalize(
            torch.randn(1, T, HG, DK, device=dev, dtype=torch.bfloat16), dim=-1, p=2
        )
        k_tr = F.normalize(
            torch.randn(1, T, HG, DK, device=dev, dtype=torch.bfloat16), dim=-1, p=2
        )
        w_tr = torch.randn(1, T, H, DK, device=dev, dtype=torch.bfloat16)
        u_tr = torch.randn(1, T, H, DV, device=dev, dtype=torch.bfloat16)
        g_tr = torch.randn(1, T, H, device=dev, dtype=torch.float32)

        h_tr, v_new_tr, _ = chunk_gated_delta_rule_fwd_h(
            k=k_tr,
            w=w_tr,
            u=u_tr,
            g=g_tr,
            initial_state=None,
            output_final_state=False,
            cu_seqlens=cu_long,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            chunk_size=C_triton,
        )
        torch.npu.synchronize()
        chunk_fwd_o(
            q=q_tr,
            k=k_tr,
            v=v_new_tr,
            h=h_tr,
            g=g_tr,
            scale=scale,
            cu_seqlens=cu_long,
            chunk_size=C_triton,
        )
        torch.npu.synchronize()

        def run_triton_o():
            chunk_fwd_o(
                q=q_tr,
                k=k_tr,
                v=v_new_tr,
                h=h_tr,
                g=g_tr,
                scale=scale,
                cu_seqlens=cu_long,
                chunk_size=C_triton,
            )

        ms_triton_o = do_bench_triton(run_triton_o)
    except Exception as e:
        msg = str(e).split("\n")[0][:240]
        print(f"[bench] Triton chunk_o skipped ({type(e).__name__}): {msg}")

    print()
    print(
        f"chunk_o group-value: N_seq={N_seq}, L_seg={L_seg}, T={T}, "
        f"H={H}, Hg={HG}, D={DK}, PTO C={C}, Triton BT={C_triton}, BLOCK_DIM={bd}"
    )
    print("| Backend | chunk_o (ms) | Notes |")
    print("| :-- | --: | :-- |")
    print(f"| PTO group-value (this dir) | {format_ms(ms_o)} | after PTO chunk_h warmup |")
    if ms_triton_o is not None:
        ratio = ms_triton_o / ms_o if ms_o > 0 else 0.0
        print(
            f"| Triton FLA vendor (`chunk_fwd_o`, BT={C_triton}) | {format_ms(ms_triton_o)} | "
            f"after Triton chunk_h warmup; vs PTO (C={C}) ×{ratio:.3f} — "
            "different chunk tile vs PTO on Ascend |",
        )


if __name__ == "__main__":
    main()
