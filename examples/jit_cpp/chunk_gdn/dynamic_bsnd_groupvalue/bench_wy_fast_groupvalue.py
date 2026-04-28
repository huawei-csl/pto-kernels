#!/usr/bin/env python3
"""
Benchmark ``wy_fast`` group-value kernel (Hg key heads, H value heads).

Same packed varlen shape as ``bench_dynamic_bsnd_groupvalue.py``. Times PTO ``wy_fast``
and FLA Triton ``recompute_w_u_fwd`` (``chunk_size=C`` for both; see parent README for
PTO vs Triton tile notes).

Usage::
  cd .../dynamic_bsnd_groupvalue
  python3 bench_wy_fast_groupvalue.py
"""
from __future__ import annotations

import ctypes
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CHUNK_GDN = os.path.abspath(os.path.join(_HERE, ".."))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _CHUNK_GDN not in sys.path:
    sys.path.insert(0, _CHUNK_GDN)

import torch

_pc_path = os.path.join(_HERE, "pto_dynamic_common.py")
_spec_pc = importlib.util.spec_from_file_location(
    "pto_dynamic_common_groupvalue_wy", _pc_path,
)
_pc_mod = importlib.util.module_from_spec(_spec_pc)
assert _spec_pc.loader is not None
_spec_pc.loader.exec_module(_pc_mod)
sys.modules["pto_dynamic_common"] = _pc_mod

_lib_here = os.path.join(_HERE, "dynamic_kernel_libs.py")
_spec_g = importlib.util.spec_from_file_location("dkgv_wy", _lib_here)
dkgv_mod = importlib.util.module_from_spec(_spec_g)
assert _spec_g.loader is not None
_spec_g.loader.exec_module(dkgv_mod)
BLOCK_DIM = dkgv_mod.BLOCK_DIM
load_wy_fast_group = dkgv_mod.load_wy_fast


def _vp(t):
    return ctypes.c_void_p(t.data_ptr())


def _transpose_g(g_sum):
    return g_sum.squeeze(0).t().contiguous()


def _transpose_beta(beta):
    return beta.squeeze(0).t().contiguous()


from gdn_bench_common import do_bench, do_bench_triton, format_ms


NPU_DEVICE = os.getenv("GDN_NPU_DEVICE", "npu:0")


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
    bd = BLOCK_DIM
    stream = torch.npu.current_stream()._as_parameter_

    lib = load_wy_fast_group(H, DK, C, key_heads=HG)
    k = torch.randn(1, T, HG, DK, device=dev, dtype=torch.float16)
    v = torch.randn(1, T, H, DV, device=dev, dtype=torch.float16)
    beta = torch.rand(1, T, H, device=dev, dtype=torch.float16)
    A = torch.randn(1, T, H, C, device=dev, dtype=torch.float16)
    g_sum = torch.randn(1, T, H, device=dev, dtype=torch.float32)
    g_t = _transpose_g(g_sum)
    beta_t = _transpose_beta(beta)
    w_out = torch.empty(1, T, H, DK, device=dev, dtype=torch.float16)
    u_out = torch.empty(1, T, H, DV, device=dev, dtype=torch.float16)
    ws1 = torch.zeros(bd, C, C, device=dev, dtype=torch.float16)
    ws2 = torch.zeros_like(ws1)

    def run_pto():
        lib.call_kernel(
            bd,
            stream,
            _vp(k),
            _vp(v),
            _vp(beta_t),
            _vp(g_t),
            _vp(A),
            _vp(ws1),
            _vp(ws2),
            _vp(w_out),
            _vp(u_out),
            _vp(cu_seqlens),
            N_seq,
            T,
            T,
        )

    run_pto()
    torch.npu.synchronize()
    ms_pto = do_bench(run_pto)

    ms_triton = None
    try:
        sys.path.insert(0, os.path.join(_CHUNK_GDN, "triton_baseline"))
        from fla_vendor.utils import prepare_chunk_indices
        from fla_vendor.wy_fast import recompute_w_u_fwd

        cu_long = cu_seqlens.long()
        chunk_indices = prepare_chunk_indices(cu_long, C)
        k_tr = torch.randn(1, T, HG, DK, device=dev, dtype=torch.bfloat16)
        v_tr = torch.randn(1, T, H, DV, device=dev, dtype=torch.bfloat16)
        beta_tr = torch.rand(1, T, H, device=dev, dtype=torch.bfloat16)
        A_tr = torch.randn(1, T, H, C, device=dev, dtype=torch.bfloat16)
        g_tr = torch.randn(1, T, H, device=dev, dtype=torch.float32)

        def run_triton():
            recompute_w_u_fwd(
                k=k_tr,
                v=v_tr,
                beta=beta_tr,
                g_cumsum=g_tr,
                A=A_tr,
                cu_seqlens=cu_long,
                chunk_indices=chunk_indices,
            )

        run_triton()
        torch.npu.synchronize()
        ms_triton = do_bench_triton(run_triton)
    except Exception as e:
        msg = str(e).split("\n")[0][:200]
        print(f"[bench] Triton wy_fast skipped ({type(e).__name__}): {msg}")

    print()
    print(
        f"wy_fast group-value: N_seq={N_seq}, L_seg={L_seg}, T={T}, "
        f"H={H}, Hg={HG}, D={DK}, C={C}, BLOCK_DIM={bd}"
    )
    print("| Backend | wy_fast (ms) | Notes |")
    print("| :-- | --: | :-- |")
    print(f"| PTO group-value (this dir) | {format_ms(ms_pto)} | packed varlen BSND |")
    if ms_triton is not None:
        ratio = ms_triton / ms_pto if ms_pto > 0 else 0.0
        print(
            f"| Triton FLA vendor (`recompute_w_u_fwd`) | {format_ms(ms_triton)} | "
            f"vs PTO ×{ratio:.3f} |",
        )


if __name__ == "__main__":
    main()
