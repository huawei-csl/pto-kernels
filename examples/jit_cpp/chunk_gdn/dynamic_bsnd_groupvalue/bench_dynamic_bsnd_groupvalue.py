#!/usr/bin/env python3
"""
Benchmark GQA group-value PTO kernels vs FLA Triton (packed varlen BSND).

Same default workload as ``dynamic_bsnd/bench_dynamic_bsnd.py``:
``N_seq=16``, ``L_seg=16384``, ``T=262144``, ``D=128``, ``C_PTO=128``.

Runs one or more stages per **value-head** count ``H`` with fixed **key-head** count ``Hg``
(``k`` / ``q`` shape ``[B,T,Hg,D]``; value tensors ``[B,T,H,D]``).

Stages:

- ``kkt`` — PTO ``scaled_dot_kkt`` vs Triton ``chunk_scaled_dot_kkt_fwd``. Triton defaults to
  ``BT=64`` (``GDN_TRITON_KKT_CHUNK``); optional ``BT=128`` only if ``GDN_TRITON_KKT_TRY128=1`` and compile succeeds
- ``chunk_h`` — PTO vs ``chunk_gated_delta_rule_fwd_h``.
- ``chunk_o`` — PTO ``chunk_o`` after PTO ``chunk_h`` warmup vs Triton ``chunk_fwd_o``
  after Triton chunk_h (``GDN_TRITON_CHUNK_O_CHUNK`` default ``64``).
- ``wy_fast`` — PTO vs ``recompute_w_u_fwd``.

Usage::

  cd chunk_gdn/dynamic_bsnd_groupvalue
  export ASCEND_TOOLKIT_HOME=... GDN_NPU_DEVICE=npu:7
  python3 bench_dynamic_bsnd_groupvalue.py
  python3 bench_dynamic_bsnd_groupvalue.py --heads 32 --hg 16 --stage kkt,chunk_h

Environment (optional): ``GDN_BENCH_HEADS``, ``GDN_BENCH_H``, ``GDN_BENCH_HG``, ``GDN_BENCH_N_SEQ``,
``GDN_BENCH_L_SEG``, ``GDN_TRITON_KKT_CHUNK``, ``GDN_TRITON_KKT_TRY128``, ``GDN_TRITON_CHUNK_O_CHUNK``.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
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
import torch.nn.functional as F

_pc_path = os.path.join(_HERE, "pto_dynamic_common.py")
_spec_pc = importlib.util.spec_from_file_location(
    "pto_dynamic_common_groupvalue_bench", _pc_path,
)
_pc_mod = importlib.util.module_from_spec(_spec_pc)
assert _spec_pc.loader is not None
_spec_pc.loader.exec_module(_pc_mod)
sys.modules["pto_dynamic_common"] = _pc_mod

_lib_here = os.path.join(_HERE, "dynamic_kernel_libs.py")
_spec_g = importlib.util.spec_from_file_location("dkgv_bench", _lib_here)
dkgv_mod = importlib.util.module_from_spec(_spec_g)
assert _spec_g.loader is not None
_spec_g.loader.exec_module(dkgv_mod)
BLOCK_DIM = dkgv_mod.BLOCK_DIM
load_scaled_dot_kkt = dkgv_mod.load_scaled_dot_kkt
load_chunk_h = dkgv_mod.load_chunk_h
load_chunk_o = dkgv_mod.load_chunk_o
load_wy_fast = dkgv_mod.load_wy_fast
total_chunks = dkgv_mod.total_chunks

from gdn_bench_common import do_bench, do_bench_triton, format_ms


def _vp(t):
    return ctypes.c_void_p(t.data_ptr())


def _transpose_g(g_sum):
    return g_sum.squeeze(0).t().contiguous()


def _transpose_beta(beta):
    return beta.squeeze(0).t().contiguous()


NPU_DEVICE = os.getenv("GDN_NPU_DEVICE", "npu:0")


def _time_triton_kkt(
    cu_seqlens: torch.Tensor,
    BT: int,
    dev: torch.device,
    T: int,
    H: int,
    HG: int,
    DK: int,
) -> float | None:
    try:
        sys.path.insert(0, os.path.join(_CHUNK_GDN, "triton_baseline"))
        from fla_vendor.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
        from fla_vendor.utils import prepare_chunk_indices

        cu_long = cu_seqlens.long()
        chunk_indices = prepare_chunk_indices(cu_long, BT)
        k_tr = torch.randn(1, T, HG, DK, device=dev, dtype=torch.bfloat16)
        beta_tr = torch.rand(1, T, H, device=dev, dtype=torch.bfloat16)
        g_tr = torch.randn(1, T, H, device=dev, dtype=torch.float32)

        def run_triton():
            chunk_scaled_dot_kkt_fwd(
                k=k_tr,
                beta=beta_tr,
                g_cumsum=g_tr,
                cu_seqlens=cu_long,
                chunk_indices=chunk_indices,
                chunk_size=BT,
                output_dtype=torch.float32,
            )

        run_triton()
        torch.npu.synchronize()
        return float(do_bench_triton(run_triton))
    except Exception as e:
        msg = str(e).split("\n")[0][:200]
        print(
            f"[bench] Triton chunk_scaled_dot_kkt BT={BT} skipped "
            f"({type(e).__name__}): {msg}",
        )
        gc.collect()
        if hasattr(torch.npu, "empty_cache"):
            torch.npu.empty_cache()
        return None


def _ratio(ms_t: float | None, ms_p: float) -> str:
    if ms_t is None or ms_p <= 0:
        return "—"
    return f"{ms_t / ms_p:.2f}×"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--heads",
        default=os.getenv("GDN_BENCH_HEADS", "16,32,48,64"),
        help="Comma-separated value head counts (overrides single GDN_BENCH_H)",
    )
    parser.add_argument(
        "--hg",
        type=int,
        default=int(os.getenv("GDN_BENCH_HG", "16")),
        help="Key / GQA head count Hg",
    )
    parser.add_argument(
        "--stage",
        default="kkt,chunk_h,chunk_o,wy_fast",
        help="Comma-separated: kkt, chunk_h, chunk_o, wy_fast",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.npu.set_device(NPU_DEVICE)
    dev = torch.device(NPU_DEVICE)

    if "PTO_LIB_PATH" not in os.environ:
        fb = "/sources/pto-isa"
        if os.path.isdir(os.path.join(fb, "include")):
            os.environ["PTO_LIB_PATH"] = fb

    N_seq = int(os.getenv("GDN_BENCH_N_SEQ", "16"))
    L_seg = int(os.getenv("GDN_BENCH_L_SEG", "16384"))
    DK = DV = 128
    C_pto = 128
    T = N_seq * L_seg
    cu_seqlens = torch.arange(0, T + 1, L_seg, dtype=torch.int32, device=dev)
    tc = total_chunks(N_seq, T, C_pto, cu_seqlens)
    bd = BLOCK_DIM
    stream = torch.npu.current_stream()._as_parameter_
    cu_p = _vp(cu_seqlens)
    batch_arg = N_seq
    seq_arg = T

    BT_kkt = int(os.getenv("GDN_TRITON_KKT_CHUNK", "64"))
    try_kkt_128 = os.getenv("GDN_TRITON_KKT_TRY128", "0") not in ("0", "false", "False")
    C_triton_o = int(os.getenv("GDN_TRITON_CHUNK_O_CHUNK", "64"))

    if os.getenv("GDN_BENCH_H"):
        heads_list = [int(os.environ["GDN_BENCH_H"])]
    else:
        heads_list = [int(x.strip()) for x in args.heads.split(",") if x.strip()]

    stages = {s.strip() for s in args.stage.split(",") if s.strip()}

    for H in heads_list:
        gc.collect()
        if hasattr(torch.npu, "empty_cache"):
            torch.npu.empty_cache()
        HG = args.hg
        assert H % HG == 0, f"H={H} must be divisible by Hg={HG}"
        print()
        print("=" * 72)
        print(
            f"GQA bench  N_seq={N_seq}  L_seg={L_seg}  T={T}  "
            f"H={H}  Hg={HG}  D={DK}  PTO_C={C_pto}  BLOCK_DIM={bd}",
        )
        print("=" * 72)

        if "kkt" in stages:
            lib_k = load_scaled_dot_kkt(H, DK, C_pto, key_heads=HG)
            k = torch.randn(1, T, HG, DK, device=dev, dtype=torch.float16)
            beta = torch.rand(1, T, H, device=dev, dtype=torch.float16)
            g_sum = torch.randn(1, T, H, device=dev, dtype=torch.float32)
            g_t = _transpose_g(g_sum)
            beta_t = _transpose_beta(beta)
            msk = torch.tril(torch.ones(C_pto, C_pto, device=dev), diagonal=-1).float()
            ws_k = torch.zeros(bd * 2, C_pto, C_pto, device=dev, dtype=torch.float16)
            A = torch.empty(1, T, H, C_pto, device=dev, dtype=torch.float16)

            def run_pto_kkt():
                lib_k.call_kernel(
                    bd,
                    stream,
                    _vp(k),
                    _vp(beta_t),
                    _vp(g_t),
                    _vp(msk),
                    _vp(ws_k),
                    _vp(A),
                    cu_p,
                    batch_arg,
                    seq_arg,
                    T,
                )

            run_pto_kkt()
            torch.npu.synchronize()
            ms_pto_k = do_bench(run_pto_kkt)
            ms_tr_k64 = _time_triton_kkt(cu_seqlens, BT_kkt, dev, T, H, HG, DK)
            ms_tr_k128 = None
            if try_kkt_128 and BT_kkt != 128:
                ms_tr_k128 = _time_triton_kkt(cu_seqlens, 128, dev, T, H, HG, DK)

            print("\n### scaled_dot_kkt")
            print("| Backend | ms | ms_triton/ms_pto (>1 ⇒ PTO faster) |")
            print("| :-- | --: | --: |")
            print(f"| PTO C={C_pto} | {format_ms(ms_pto_k)} | — |")
            if ms_tr_k64 is not None:
                print(
                    f"| Triton BT={BT_kkt} | {format_ms(ms_tr_k64)} | "
                    f"{_ratio(ms_tr_k64, ms_pto_k)} |",
                )
            if ms_tr_k128 is not None:
                print(
                    f"| Triton BT=128 (optional) | {format_ms(ms_tr_k128)} | "
                    f"{_ratio(ms_tr_k128, ms_pto_k)} |",
                )
            elif try_kkt_128 and BT_kkt != 128:
                print("| Triton BT=128 (optional) | — | — |")

            del k, beta, g_sum, g_t, beta_t, msk, ws_k, A
            gc.collect()
            if hasattr(torch.npu, "empty_cache"):
                torch.npu.empty_cache()

        if "chunk_h" in stages:
            lib_h = load_chunk_h(H, DK, C_pto, key_heads=HG)
            k_h = torch.randn(1, T, HG, DK, device=dev, dtype=torch.float16)
            w_h = torch.randn(1, T, H, DK, device=dev, dtype=torch.float16)
            u_h = torch.randn(1, T, H, DV, device=dev, dtype=torch.float16)
            g_sum_h = torch.randn(1, T, H, device=dev, dtype=torch.float32)
            g_t_h = _transpose_g(g_sum_h)
            ws_h = torch.zeros(bd * 4, DK, DV, device=dev, dtype=torch.float16)
            s_h = torch.zeros(tc * H, DK, DV, device=dev, dtype=torch.float16)
            nv_h = torch.empty(1, T, H, DV, device=dev, dtype=torch.float16)
            fs_h = torch.empty(N_seq * H, DK, DV, device=dev, dtype=torch.float16)

            def run_pto_h():
                lib_h.call_kernel(
                    bd,
                    stream,
                    _vp(k_h),
                    _vp(w_h),
                    _vp(u_h),
                    _vp(g_t_h),
                    _vp(s_h),
                    _vp(nv_h),
                    _vp(fs_h),
                    _vp(ws_h),
                    cu_p,
                    batch_arg,
                    seq_arg,
                    T,
                )

            run_pto_h()
            torch.npu.synchronize()
            ms_pto_h = do_bench(run_pto_h)

            ms_tr_h = None
            try:
                sys.path.insert(0, os.path.join(_CHUNK_GDN, "triton_baseline"))
                from fla_vendor.chunk_delta_h import chunk_gated_delta_rule_fwd_h
                from fla_vendor.utils import prepare_chunk_indices, prepare_chunk_offsets

                cu_long = cu_seqlens.long()
                chunk_indices = prepare_chunk_indices(cu_long, C_pto)
                chunk_offsets = prepare_chunk_offsets(cu_long, C_pto)
                k_tr = torch.randn(1, T, HG, DK, device=dev, dtype=torch.bfloat16)
                w_tr = torch.randn(1, T, H, DK, device=dev, dtype=torch.bfloat16)
                u_tr = torch.randn(1, T, H, DV, device=dev, dtype=torch.bfloat16)
                g_tr = torch.randn(1, T, H, device=dev, dtype=torch.float32)

                def run_triton_h():
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
                        chunk_size=C_pto,
                    )

                run_triton_h()
                torch.npu.synchronize()
                ms_tr_h = do_bench_triton(run_triton_h)
            except Exception as e:
                print(
                    f"[bench] Triton chunk_h skipped ({type(e).__name__}): "
                    f"{str(e).splitlines()[0][:200]}",
                )

            print("\n### chunk_h")
            print("| Backend | ms | ms_triton/ms_pto |")
            print("| :-- | --: | --: |")
            print(f"| PTO | {format_ms(ms_pto_h)} | — |")
            if ms_tr_h is not None:
                print(f"| Triton | {format_ms(ms_tr_h)} | {_ratio(ms_tr_h, ms_pto_h)} |")

            del lib_h, k_h, w_h, u_h, g_sum_h, g_t_h, ws_h, s_h, nv_h, fs_h
            try:
                del k_tr, w_tr, u_tr, g_tr
            except NameError:
                pass
            gc.collect()
            if hasattr(torch.npu, "empty_cache"):
                torch.npu.empty_cache()

        if "chunk_o" in stages:
            lib_h = load_chunk_h(H, DK, C_pto, key_heads=HG)
            lib_o = load_chunk_o(H, DK, C_pto, key_heads=HG)
            k_o = F.normalize(torch.randn(1, T, HG, DK, device=dev, dtype=torch.float16), dim=-1, p=2)
            q_o = F.normalize(torch.randn(1, T, HG, DK, device=dev, dtype=torch.float16), dim=-1, p=2)
            w_o = torch.randn(1, T, H, DK, device=dev, dtype=torch.float16)
            u_o = torch.randn(1, T, H, DV, device=dev, dtype=torch.float16)
            g_sum_o = torch.randn(1, T, H, device=dev, dtype=torch.float32)
            g_t_o = _transpose_g(g_sum_o)
            ws_h = torch.zeros(bd * 4, DK, DV, device=dev, dtype=torch.float16)
            s_o = torch.zeros(tc * H, DK, DV, device=dev, dtype=torch.float16)
            nv_o = torch.empty(1, T, H, DV, device=dev, dtype=torch.float16)
            fs_o = torch.empty(N_seq * H, DK, DV, device=dev, dtype=torch.float16)

            lib_h.call_kernel(
                bd,
                stream,
                _vp(k_o),
                _vp(w_o),
                _vp(u_o),
                _vp(g_t_o),
                _vp(s_o),
                _vp(nv_o),
                _vp(fs_o),
                _vp(ws_h),
                cu_p,
                batch_arg,
                seq_arg,
                T,
            )
            torch.npu.synchronize()

            msk2 = torch.tril(torch.ones(C_pto, C_pto, device=dev), diagonal=0).float()
            w1 = torch.zeros(bd, C_pto, C_pto, device=dev, dtype=torch.float16)
            w2 = torch.zeros(bd, C_pto, DV, device=dev, dtype=torch.float16)
            w3 = torch.zeros(bd, C_pto, C_pto, device=dev, dtype=torch.float16)
            o_o = torch.empty(1, T, H, DV, device=dev, dtype=torch.float16)

            def run_pto_o():
                lib_o.call_kernel(
                    bd,
                    stream,
                    _vp(q_o),
                    _vp(k_o),
                    _vp(nv_o),
                    _vp(s_o),
                    _vp(g_t_o),
                    _vp(msk2),
                    _vp(w1),
                    _vp(w2),
                    _vp(w3),
                    _vp(o_o),
                    cu_p,
                    batch_arg,
                    seq_arg,
                    T,
                )

            run_pto_o()
            torch.npu.synchronize()
            ms_pto_o = do_bench(run_pto_o)

            ms_tr_o = None
            try:
                sys.path.insert(0, os.path.join(_CHUNK_GDN, "triton_baseline"))
                from fla_vendor.chunk_delta_h import chunk_gated_delta_rule_fwd_h
                from fla_vendor.chunk_o import chunk_fwd_o
                from fla_vendor.utils import prepare_chunk_indices, prepare_chunk_offsets

                cu_long = cu_seqlens.long()
                chunk_indices = prepare_chunk_indices(cu_long, C_triton_o)
                chunk_offsets = prepare_chunk_offsets(cu_long, C_triton_o)
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
                    chunk_size=C_triton_o,
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
                        chunk_size=C_triton_o,
                    )

                run_triton_o()
                torch.npu.synchronize()
                ms_tr_o = do_bench_triton(run_triton_o)
            except Exception as e:
                msg = str(e).split("\n")[0][:240]
                print(f"[bench] Triton chunk_o skipped ({type(e).__name__}): {msg}")

            print("\n### chunk_o")
            print(
                f"(PTO C={C_pto}; Triton ``chunk_fwd_o`` BT={C_triton_o}; "
                "PTO chunk_h warmup done; Triton chunk_h warmup done before timing)\n",
            )
            print("| Backend | ms | ms_triton/ms_pto |")
            print("| :-- | --: | --: |")
            print(f"| PTO | {format_ms(ms_pto_o)} | — |")
            if ms_tr_o is not None:
                print(f"| Triton | {format_ms(ms_tr_o)} | {_ratio(ms_tr_o, ms_pto_o)} |")

            del lib_h, lib_o, k_o, q_o, w_o, u_o, g_sum_o, g_t_o, ws_h, s_o, nv_o, fs_o, msk2, w1, w2, w3, o_o
            try:
                del q_tr, k_tr, w_tr, u_tr, g_tr, h_tr, v_new_tr
            except NameError:
                pass
            gc.collect()
            if hasattr(torch.npu, "empty_cache"):
                torch.npu.empty_cache()

        if "wy_fast" in stages:
            lib_w = load_wy_fast(H, DK, C_pto, key_heads=HG)
            k_w = torch.randn(1, T, HG, DK, device=dev, dtype=torch.float16)
            v_w = torch.randn(1, T, H, DV, device=dev, dtype=torch.float16)
            beta_w = torch.rand(1, T, H, device=dev, dtype=torch.float16)
            A_w = torch.randn(1, T, H, C_pto, device=dev, dtype=torch.float16)
            g_sum_w = torch.randn(1, T, H, device=dev, dtype=torch.float32)
            g_t_w = _transpose_g(g_sum_w)
            beta_t_w = _transpose_beta(beta_w)
            w1 = torch.zeros(bd, C_pto, C_pto, device=dev, dtype=torch.float16)
            w2 = torch.zeros_like(w1)
            w_out = torch.empty(1, T, H, DK, device=dev, dtype=torch.float16)
            u_out = torch.empty(1, T, H, DV, device=dev, dtype=torch.float16)

            def run_pto_w():
                lib_w.call_kernel(
                    bd,
                    stream,
                    _vp(k_w),
                    _vp(v_w),
                    _vp(beta_t_w),
                    _vp(g_t_w),
                    _vp(A_w),
                    _vp(w1),
                    _vp(w2),
                    _vp(w_out),
                    _vp(u_out),
                    cu_p,
                    batch_arg,
                    seq_arg,
                    T,
                )

            run_pto_w()
            torch.npu.synchronize()
            ms_pto_w = do_bench(run_pto_w)

            ms_tr_w = None
            try:
                sys.path.insert(0, os.path.join(_CHUNK_GDN, "triton_baseline"))
                from fla_vendor.utils import prepare_chunk_indices
                from fla_vendor.wy_fast import recompute_w_u_fwd

                cu_long = cu_seqlens.long()
                chunk_indices = prepare_chunk_indices(cu_long, C_pto)
                k_tr = torch.randn(1, T, HG, DK, device=dev, dtype=torch.bfloat16)
                v_tr = torch.randn(1, T, H, DV, device=dev, dtype=torch.bfloat16)
                beta_tr = torch.rand(1, T, H, device=dev, dtype=torch.bfloat16)
                A_tr = torch.randn(1, T, H, C_pto, device=dev, dtype=torch.bfloat16)
                g_tr = torch.randn(1, T, H, device=dev, dtype=torch.float32)

                def run_triton_w():
                    recompute_w_u_fwd(
                        k=k_tr,
                        v=v_tr,
                        beta=beta_tr,
                        g_cumsum=g_tr,
                        A=A_tr,
                        cu_seqlens=cu_long,
                        chunk_indices=chunk_indices,
                    )

                run_triton_w()
                torch.npu.synchronize()
                ms_tr_w = do_bench_triton(run_triton_w)
            except Exception as e:
                msg = str(e).split("\n")[0][:200]
                print(f"[bench] Triton wy_fast skipped ({type(e).__name__}): {msg}")

            print("\n### wy_fast")
            print("| Backend | ms | ms_triton/ms_pto |")
            print("| :-- | --: | --: |")
            print(f"| PTO | {format_ms(ms_pto_w)} | — |")
            if ms_tr_w is not None:
                print(f"| Triton | {format_ms(ms_tr_w)} | {_ratio(ms_tr_w, ms_pto_w)} |")

            del lib_w, k_w, v_w, beta_w, A_w, g_sum_w, g_t_w, beta_t_w, w1, w2, w_out, u_out
            try:
                del k_tr, v_tr, beta_tr, A_tr, g_tr
            except NameError:
                pass
            gc.collect()
            if hasattr(torch.npu, "empty_cache"):
                torch.npu.empty_cache()

        gc.collect()
        if hasattr(torch.npu, "empty_cache"):
            torch.npu.empty_cache()


if __name__ == "__main__":
    main()
