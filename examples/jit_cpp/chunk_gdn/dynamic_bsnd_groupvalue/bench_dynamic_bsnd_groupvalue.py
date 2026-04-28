#!/usr/bin/env python3
"""
Benchmark ``chunk_h`` group-value kernel vs the original dynamic_bsndk ``chunk_h``.

Uses the same packed varlen shape as ``dynamic_bsnd/bench_dynamic_bsnd.py``
(N_seq=16, L_seg=16384, T=262144, D=128, C=128).

Compare ``chunk_h`` latency from this directory (PTO group-value layout:
``k`` is ``[B,T,Hg,D]``, ``w/u`` are ``[B,T,H,D]``) against Triton FLA when available.

To compare against the original single-head-count PTO ``chunk_h``, run
``dynamic_bsnd/bench_dynamic_bsnd.py`` in a separate process with the same ``H`` when ``H=Hg``.

Usage::
  cd .../dynamic_bsnd_groupvalue
  python3 bench_dynamic_bsnd_groupvalue.py
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

# Ensure this directory's ``pto_dynamic_common`` is used (signature includes ``key_heads``).
_pc_path = os.path.join(_HERE, "pto_dynamic_common.py")
_spec_pc = importlib.util.spec_from_file_location(
    "pto_dynamic_common_groupvalue", _pc_path,
)
_pc_mod = importlib.util.module_from_spec(_spec_pc)
assert _spec_pc.loader is not None
_spec_pc.loader.exec_module(_pc_mod)
sys.modules["pto_dynamic_common"] = _pc_mod

_lib_here = os.path.join(_HERE, "dynamic_kernel_libs.py")
_spec_g = importlib.util.spec_from_file_location("dkgv_mod", _lib_here)
dkgv_mod = importlib.util.module_from_spec(_spec_g)
assert _spec_g.loader is not None
_spec_g.loader.exec_module(dkgv_mod)
BLOCK_DIM = dkgv_mod.BLOCK_DIM
load_chunk_h_group = dkgv_mod.load_chunk_h
total_chunks = dkgv_mod.total_chunks

from gdn_bench_common import do_bench, format_ms


def _vp(t):
    return ctypes.c_void_p(t.data_ptr())


def _transpose_g(g_sum):
    return g_sum.squeeze(0).t().contiguous()


NPU_DEVICE = os.getenv("GDN_NPU_DEVICE", "npu:0")


def bench_pto(lib, bd, stream, tensors, cu_p, batch_arg, seq_arg, T):
    k, w, u, g_t, s, nv, fs, ws = tensors

    def fn():
        lib.call_kernel(
            bd,
            stream,
            _vp(k),
            _vp(w),
            _vp(u),
            _vp(g_t),
            _vp(s),
            _vp(nv),
            _vp(fs),
            _vp(ws),
            cu_p,
            batch_arg,
            seq_arg,
            T,
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

    lib_g = load_chunk_h_group(H, DK, C, key_heads=HG)
    k_g = torch.randn(1, T, HG, DK, device=dev, dtype=torch.float16)
    w_g = torch.randn(1, T, H, DK, device=dev, dtype=torch.float16)
    u_g = torch.randn(1, T, H, DV, device=dev, dtype=torch.float16)
    g_sum_g = torch.randn(1, T, H, device=dev, dtype=torch.float32)
    g_t_g = _transpose_g(g_sum_g)
    ws_g = torch.zeros(bd * 4, DK, DV, device=dev, dtype=torch.float16)
    s_g = torch.zeros(tc * H, DK, DV, device=dev, dtype=torch.float16)
    nv_g = torch.empty(1, T, H, DV, device=dev, dtype=torch.float16)
    fs_g = torch.empty(N_seq * H, DK, DV, device=dev, dtype=torch.float16)
    ms_group = bench_pto(
        lib_g,
        bd,
        stream,
        (k_g, w_g, u_g, g_t_g, s_g, nv_g, fs_g, ws_g),
        cu_p,
        N_seq,
        T,
        T,
    )

    ms_triton = None
    try:
        sys.path.insert(0, os.path.join(_CHUNK_GDN, "triton_baseline"))
        from fla_vendor.chunk_delta_h import chunk_gated_delta_rule_fwd_h
        from fla_vendor.utils import prepare_chunk_indices, prepare_chunk_offsets

        cu_long = cu_seqlens.long()
        chunk_indices = prepare_chunk_indices(cu_long, C)
        chunk_offsets = prepare_chunk_offsets(cu_long, C)
        k_tr = torch.randn(1, T, HG, DK, device=dev, dtype=torch.bfloat16)
        w_tr = torch.randn(1, T, H, DK, device=dev, dtype=torch.bfloat16)
        u_tr = torch.randn(1, T, H, DV, device=dev, dtype=torch.bfloat16)
        g_tr = torch.randn(1, T, H, device=dev, dtype=torch.float32)

        def run_triton():
            chunk_gated_delta_rule_fwd_h(
                k=k_tr,
                w=w_tr,
                u=u_tr,
                g=g_tr,
                initial_state=None,
                output_final_state=False,
                cu_seqlens=cu_long,
                chunk_indices=chunk_indices,
                chunk_offsets=chunk_offsets,
                chunk_size=C,
            )

        run_triton()
        torch.npu.synchronize()
        from gdn_bench_common import do_bench_triton

        ms_triton = do_bench_triton(run_triton)
    except Exception as e:
        print(f"[bench] Triton chunk_h skipped: {e}")

    print()
    print(
        f"Shape: N_seq={N_seq}, L_seg={L_seg}, T={T}, H={H}, Hg={HG}, "
        f"D={DK}, C={C}, BLOCK_DIM={bd}"
    )
    print("| Backend | chunk_h (ms) | Notes |")
    print("| :-- | --: | :-- |")
    print(f"| PTO group-value (this dir) | {format_ms(ms_group)} | packed varlen BSND |")
    print(
        "| Original PTO ``dynamic_bsnd/bench_dynamic_bsnd.py`` | — | "
        "run separately with matching ``H`` when ``Hg=H`` |",
    )
    if ms_triton is not None:
        sp = ms_triton / ms_group if ms_group > 0 else 0
        print(f"| Triton FLA vendor | {format_ms(ms_triton)} | vs PTO group-value ×{sp:.3f} |")


if __name__ == "__main__":
    main()
