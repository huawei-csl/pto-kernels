#!/usr/bin/env python3
"""
Benchmark ``scaled_dot_kkt`` group-value kernel (Hg key heads, H value heads).

Same packed varlen shape as ``bench_dynamic_bsnd_groupvalue.py``.

- **PTO** uses compile-time ``GDN_C=128`` (this kernel build).
- **Triton** ``chunk_scaled_dot_kkt_fwd`` defaults to **`chunk_size=64`` (BT=64)** so the MLIR
  pipeline compiles on Ascend; set ``GDN_TRITON_KKT_CHUNK`` to override the **primary** Triton tile.
- After the BT=64 timing, the script **optionally** tries **BT=128** and only prints it if compile
  and execution succeed.

Tables report **`ms_triton / ms_pto`** on Triton rows (**values > 1 ⇒ PTO is faster** than that Triton config).

Usage::
  cd .../dynamic_bsnd_groupvalue
  GDN_BENCH_H=32 GDN_BENCH_HG=16 python3 bench_scaled_dot_kkt_groupvalue.py
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
    "pto_dynamic_common_groupvalue_kkt", _pc_path,
)
_pc_mod = importlib.util.module_from_spec(_spec_pc)
assert _spec_pc.loader is not None
_spec_pc.loader.exec_module(_pc_mod)
sys.modules["pto_dynamic_common"] = _pc_mod

_lib_here = os.path.join(_HERE, "dynamic_kernel_libs.py")
_spec_g = importlib.util.spec_from_file_location("dkgv_kkt", _lib_here)
dkgv_mod = importlib.util.module_from_spec(_spec_g)
assert _spec_g.loader is not None
_spec_g.loader.exec_module(dkgv_mod)
BLOCK_DIM = dkgv_mod.BLOCK_DIM
load_scaled_dot_kkt_group = dkgv_mod.load_scaled_dot_kkt


def _vp(t):
    return ctypes.c_void_p(t.data_ptr())


def _transpose_g(g_sum):
    return g_sum.squeeze(0).t().contiguous()


def _transpose_beta(beta):
    return beta.squeeze(0).t().contiguous()


from gdn_bench_common import do_bench, do_bench_triton, format_ms


NPU_DEVICE = os.getenv("GDN_NPU_DEVICE", "npu:0")


def _time_triton_chunk_scaled_dot_kkt(
    cu_seqlens: torch.Tensor,
    BT: int,
    dev: torch.device,
    T: int,
    H: int,
    HG: int,
    DK: int,
) -> float | None:
    """Return median ms for ``chunk_scaled_dot_kkt_fwd`` or None on failure."""
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
        msg = str(e).split("\n")[0][:220]
        print(
            f"[bench] Triton chunk_scaled_dot_kkt BT={BT} skipped "
            f"({type(e).__name__}): {msg}",
        )
        return None


def main():
    torch.manual_seed(0)
    torch.npu.set_device(NPU_DEVICE)
    dev = torch.device(NPU_DEVICE)

    N_seq = int(os.getenv("GDN_BENCH_N_SEQ", "16"))
    L_seg = int(os.getenv("GDN_BENCH_L_SEG", "16384"))
    DK = 128
    C_pto = 128
    H = int(os.getenv("GDN_BENCH_H", "32"))
    HG = int(os.getenv("GDN_BENCH_HG", "16"))
    assert H % HG == 0
    T = N_seq * L_seg

    # Primary Triton tile (default 64 — compiles reliably on Ascend MLIR path)
    BT_triton = int(os.getenv("GDN_TRITON_KKT_CHUNK", "64"))
    try_triton_128 = os.getenv("GDN_TRITON_KKT_TRY128", "1") not in ("0", "false", "False")

    cu_seqlens = torch.arange(0, T + 1, L_seg, dtype=torch.int32, device=dev)
    bd = BLOCK_DIM
    stream = torch.npu.current_stream()._as_parameter_
    cu_p = _vp(cu_seqlens)

    lib = load_scaled_dot_kkt_group(H, DK, C_pto, key_heads=HG)
    k = torch.randn(1, T, HG, DK, device=dev, dtype=torch.float16)
    beta = torch.rand(1, T, H, device=dev, dtype=torch.float16)
    g_sum = torch.randn(1, T, H, device=dev, dtype=torch.float32)
    g_t = _transpose_g(g_sum)
    beta_t = _transpose_beta(beta)
    msk = torch.tril(torch.ones(C_pto, C_pto, device=dev), diagonal=-1).float()
    workspace_kkt = torch.zeros(bd * 2, C_pto, C_pto, device=dev, dtype=torch.float16)
    A = torch.empty(1, T, H, C_pto, device=dev, dtype=torch.float16)

    batch_arg = N_seq
    seq_arg = T

    def run_pto():
        lib.call_kernel(
            bd,
            stream,
            _vp(k),
            _vp(beta_t),
            _vp(g_t),
            _vp(msk),
            _vp(workspace_kkt),
            _vp(A),
            cu_p,
            batch_arg,
            seq_arg,
            T,
        )

    run_pto()
    torch.npu.synchronize()
    ms_pto = do_bench(run_pto)

    ms_triton_64 = _time_triton_chunk_scaled_dot_kkt(
        cu_seqlens, BT_triton, dev, T, H, HG, DK,
    )
    ms_triton_128 = None
    if try_triton_128 and BT_triton != 128:
        ms_triton_128 = _time_triton_chunk_scaled_dot_kkt(
            cu_seqlens, 128, dev, T, H, HG, DK,
        )

    def _ratio(ms_triton: float | None) -> str:
        if ms_triton is None or ms_pto <= 0:
            return "—"
        return f"{ms_triton / ms_pto:.2f}×"

    print()
    print(
        f"scaled_dot_kkt group-value: N_seq={N_seq}, L_seg={L_seg}, T={T}, "
        f"H={H}, Hg={HG}, D={DK}, PTO C={C_pto}, Triton primary BT={BT_triton}, "
        f"BLOCK_DIM={bd}",
    )
    print()
    print(
        "| Backend | scaled_dot_kkt (ms) | "
        "`ms_triton/ms_pto` (>1 ⇒ PTO faster) |",
    )
    print("| :-- | --: | --: |")
    print(f"| PTO (`C={C_pto}`) | {format_ms(ms_pto)} | — |")
    if ms_triton_64 is not None:
        print(
            f"| Triton `chunk_scaled_dot_kkt_fwd` (`BT={BT_triton}`) | "
            f"{format_ms(ms_triton_64)} | {_ratio(ms_triton_64)} |",
        )
    if ms_triton_128 is not None:
        print(
            "| Triton `chunk_scaled_dot_kkt_fwd` (`BT=128`, optional) | "
            f"{format_ms(ms_triton_128)} | {_ratio(ms_triton_128)} |",
        )
    elif try_triton_128 and BT_triton != 128:
        print(
            "| Triton `chunk_scaled_dot_kkt_fwd` (`BT=128`, optional) | — | — |",
        )


if __name__ == "__main__":
    main()
