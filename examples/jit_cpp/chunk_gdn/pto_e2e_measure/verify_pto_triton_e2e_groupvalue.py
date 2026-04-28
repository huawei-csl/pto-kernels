#!/usr/bin/env python3
"""
End-to-end GQA group-value GDN (``H`` value heads, ``Hg`` shared Q/K heads):
PTO chain (``C=128``) + ``fast_inverse`` vs Triton (``C=64``).

**Pass criteria:** same as ``verify_pto_triton_e2e.py`` — each backend matches its
CPU fp32 reference; PTO and Triton also agree pairwise
(``atol=1e-5``, ``rtol=1e-2``, RMSE ratios, ``R²``, Pearson ``ρ``).

Tensor layout: ``q``, ``k`` are ``[B,T,Hg,D]``; ``v``, ``beta``, gates, ``o`` use
``H`` heads (``head_g = head // (H // Hg)``, same as FLA/Triton).

Cumsum and ``solve_tril`` use the unchanged ``dynamic_bsnd`` kernels (gates and
blocks are indexed by value head ``H``). Stages ``scaled_dot_kkt``,
``wy_fast``, ``chunk_h``, ``chunk_o`` use ``dynamic_bsnd_groupvalue``.

Pipeline (both):
  cumsum → scaled_dot_kkt → solve_tril → wy_fast → chunk_h → chunk_o

Usage:
  cd examples/jit_cpp/chunk_gdn/pto_e2e_measure
  python verify_pto_triton_e2e_groupvalue.py --device npu:4 --H 32 --hg 16
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_FIG_DIR = os.path.join(_HERE, "output", "fig")
_DEFAULT_CSV_DIR = os.path.join(_HERE, "csv")
_CHUNK_GDN = os.path.abspath(os.path.join(_HERE, ".."))
_DYN_GROUP = os.path.join(_CHUNK_GDN, "dynamic_bsnd_groupvalue")
_DYN = os.path.join(_CHUNK_GDN, "dynamic_bsnd")
_JIT_CPP = os.path.abspath(os.path.join(_CHUNK_GDN, ".."))
_FAST_INV = os.path.join(_JIT_CPP, "fast_inverse")

for p in (_CHUNK_GDN, _DYN_GROUP, _DYN, _FAST_INV):
    if p not in sys.path:
        sys.path.insert(0, p)
if os.path.join(_CHUNK_GDN, "triton_baseline") not in sys.path:
    sys.path.insert(0, os.path.join(_CHUNK_GDN, "triton_baseline"))


def _import_dynamic_kernel_libs(path_dir: str, logical_name: str):
    ml = os.path.join(path_dir, "dynamic_kernel_libs.py")
    spec = importlib.util.spec_from_file_location(logical_name, ml)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_dkl_std = _import_dynamic_kernel_libs(_DYN, "pto_dkl_standard")
_dkl_gv = _import_dynamic_kernel_libs(_DYN_GROUP, "pto_dkl_groupvalue")

BLOCK_DIM = _dkl_std.BLOCK_DIM
run_chunk_cumsum = _dkl_std.run_chunk_cumsum
_transpose_g = _dkl_gv._transpose_g
_transpose_beta = _dkl_gv._transpose_beta
run_scaled_dot_kkt = _dkl_gv.run_scaled_dot_kkt
run_wy_fast = _dkl_gv.run_wy_fast
run_chunk_h = _dkl_gv.run_chunk_h
run_chunk_o = _dkl_gv.run_chunk_o
total_chunks = _dkl_gv.total_chunks

import torch
import torch.nn.functional as F

from verify_dynamic_bsnd import ref_solve_tril

from verify_dynamic_bsnd_groupvalue import (
    ref_chunk_h_group,
    ref_chunk_o_group,
    ref_cumsum,
    ref_kkt_group,
    ref_wy_group,
)

from jit_util_fast_inverse import jit_compile

from triton_baseline.fla_vendor.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from triton_baseline.fla_vendor.chunk_o import chunk_fwd_o
from triton_baseline.fla_vendor.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from triton_baseline.fla_vendor.cumsum import chunk_local_cumsum
from triton_baseline.fla_vendor.solve_tril import solve_tril
from triton_baseline.fla_vendor.utils import prepare_chunk_indices, prepare_chunk_offsets
from triton_baseline.fla_vendor.wy_fast import recompute_w_u_fwd

C_PTO = 128
C_TRITON = 64
HG_DEFAULT = int(os.getenv("GDN_HG", "16"))
H_DEFAULT = int(os.getenv("GDN_GROUPVALUE_H", "32"))
D_DEFAULT = 128

RTOL_REF = 1e-2
ATOL_REF = 1e-5
MAX_RMSE_OVER_MEAN_ABS_TRI = 0.09
MAX_RMSE_OVER_MEAN_ABS_PTO = 0.15
MIN_R2 = 0.99
MIN_PEARSON = 0.995
MIN_R2_PTO = 0.99
MIN_PEARSON_PTO = 0.995
MAX_RMSE_OVER_MEAN_ABS_CROSS = 0.02
MIN_R2_CROSS = 0.999
MIN_PEARSON_CROSS = 0.999
SCATTER_MAX_POINTS = 80_000


def _safe_exp_gate(gc_rowcol: torch.Tensor) -> torch.Tensor:
    """Match FLA ``safe_exp``: ``exp(x)`` if ``x <= 0`` else ``0`` (pairwise Δg tensor)."""
    return torch.where(gc_rowcol <= 0, torch.exp(gc_rowcol), torch.zeros_like(gc_rowcol))


def _seq_ranges(T: int, cu_seqlens):
    if cu_seqlens is None:
        return [(0, T)]
    cu = cu_seqlens.tolist() if hasattr(cu_seqlens, "tolist") else cu_seqlens
    return [(cu[i], cu[i + 1]) for i in range(len(cu) - 1)]


def ref_chunk_o_group_fla(
    q: torch.Tensor,
    k: torch.Tensor,
    v_new: torch.Tensor,
    h_states: torch.Tensor,
    g_cumsum: torch.Tensor,
    cs: int,
    cu_seqlens=None,
):
    """CPU ref matching Triton ``chunk_fwd_o`` gated attention (FLA-safe_exp), GQA indexing."""
    B, T, Hg, Dd = q.shape
    H = v_new.shape[2]
    assert H % Hg == 0
    grp = H // Hg
    qf, kf, vf, gf = q.float(), k.float(), v_new.float(), g_cumsum.float()
    o = torch.zeros(B, T, H, Dd, dtype=torch.float32)
    ranges = _seq_ranges(T, cu_seqlens)
    ci_base = 0
    for bos, eos in ranges:
        nc = (eos - bos + cs - 1) // cs
        for h in range(H):
            hg = h // grp
            for ci in range(nc):
                s, e = bos + ci * cs, min(bos + (ci + 1) * cs, eos)
                vlen = e - s
                qc = qf[0, s:e, hg, :]
                kc = kf[0, s:e, hg, :]
                vc = vf[0, s:e, h, :]
                gc = gf[0, s:e, h]
                inter = (qc @ h_states[ci_base + ci, h]) * torch.exp(gc)[:, None]
                qk = qc @ kc.T
                mask = torch.arange(vlen, device=qk.device)[:, None] >= torch.arange(
                    vlen, device=qk.device
                )[None, :]
                gate = _safe_exp_gate(gc[:, None] - gc[None, :])
                o[0, s:e, h, :] = inter + (qk * gate * mask.float()) @ vc
        ci_base += nc
    return o


def r2_score(y_ref: torch.Tensor, y: torch.Tensor) -> float:
    ref = np.asarray(y_ref.detach().cpu().numpy().ravel(), dtype=np.float64)
    pred = np.asarray(y.detach().cpu().numpy().ravel(), dtype=np.float64)
    ss_res = float(np.sum((ref - pred) ** 2))
    ss_tot = float(np.sum((ref - np.mean(ref)) ** 2))
    if ss_tot <= 1e-30 * max(ref.size, 1):
        return float("nan")
    return 1.0 - ss_res / ss_tot


def pearson_r(x: torch.Tensor, y: torch.Tensor) -> float:
    a = np.asarray(x.detach().cpu().numpy().ravel(), dtype=np.float64)
    b = np.asarray(y.detach().cpu().numpy().ravel(), dtype=np.float64)
    if a.size < 2:
        return float("nan")
    if np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(a, b)
    v = float(c[0, 1])
    return v if np.isfinite(v) else float("nan")


def _scatter_subsample(
    out: torch.Tensor, out_ref: torch.Tensor, max_n: int
) -> tuple[torch.Tensor, torch.Tensor]:
    n = out_ref.numel()
    if n <= max_n:
        return out.flatten(), out_ref.flatten()
    idx = torch.randperm(n, device=out_ref.device)[:max_n]
    return out.flatten()[idx], out_ref.flatten()[idx]


def plot_scatter_1to1(
    out: torch.Tensor,
    out_ref: torch.Tensor,
    *,
    title: str,
    path: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, y = _scatter_subsample(out, out_ref, SCATTER_MAX_POINTS)
    x_np = np.asarray(x.detach().cpu().numpy(), dtype=np.float64).ravel()
    y_np = np.asarray(y.detach().cpu().numpy(), dtype=np.float64).ravel()

    lo_d = float(min(x_np.min(), y_np.min()))
    hi_d = float(max(x_np.max(), y_np.max()))
    span = hi_d - lo_d
    pad = max(0.02 * span, 1e-6 * max(abs(lo_d), abs(hi_d), 1.0))
    lo, hi = lo_d - pad, hi_d + pad

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x_np, y_np, s=2, alpha=0.35, c="C0", rasterized=True, zorder=1)
    ax.plot([lo, hi], [lo, hi], color="C3", ls="-", lw=1.75, label="y = x", zorder=5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect(1)
    ax.set_xlabel("PTO output (flatten)")
    ax.set_ylabel("Triton output (flatten)")
    ax.set_title(title)
    ax.grid(True, alpha=0.35, linestyle=":", linewidth=0.6)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _safe_filename(label: str) -> str:
    s = re.sub(r"[^\w\-+.,=]+", "_", label)
    return s.strip("_")[:120] or "case"


def _count_varlen_chunks(cu_seqlens: torch.Tensor, chunk_size: int) -> int:
    return sum(
        (int(eos) - int(bos) + chunk_size - 1) // chunk_size
        for bos, eos in zip(
            cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=False
        )
    )


def _cu_from_seqlens(seqlens: list[int]) -> list[int]:
    cu = [0]
    for slen in seqlens:
        cu.append(cu[-1] + slen)
    return cu


def _make_minus_identity(matrix_size: int, device: torch.device) -> torch.Tensor:
    minus_identity = torch.zeros(
        (matrix_size, matrix_size),
        dtype=torch.float16,
        device=device,
    )
    minus_identity.fill_diagonal_(-1)
    return minus_identity


def pto_solve_tril(
    tri_inv_func,
    A_fp16: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_size: int,
    num_heads: int,
) -> torch.Tensor:
    """``(I+L)^{-1}`` in BSND layout; ``A`` is indexed by ``H`` value heads."""
    num_matrices = _count_varlen_chunks(cu_seqlens, chunk_size) * num_heads
    tensor_out = torch.zeros_like(A_fp16, dtype=torch.float32)
    minus_identity = _make_minus_identity(chunk_size, A_fp16.device)
    torch.npu.synchronize()
    tri_inv_func(
        tensor_out,
        A_fp16,
        minus_identity,
        chunk_size,
        num_matrices,
        num_heads,
        cu_seqlens=cu_seqlens,
        block_dim=BLOCK_DIM,
        is_lower=True,
    )
    torch.npu.synchronize()
    return tensor_out.to(torch.float16)


def run_pto_e2e(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_in: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    stream,
    tri_inv_func,
    scale: float,
    H: int,
    HG: int,
) -> torch.Tensor:
    """``q``, ``k``: NPU fp16 ``[B,T,Hg,D]``; ``v``, ``β``, gates: ``[B,T,H,...]``."""
    dev = q.device
    N_seq = len(cu_seqlens) - 1
    T = q.shape[1]
    assert q.shape[2] == HG and k.shape[2] == HG
    assert H % HG == 0
    assert v.shape[2] == H == beta.shape[2] == g_in.shape[2]

    msk_lower = torch.tril(
        torch.ones(C_PTO, C_PTO, device=dev), diagonal=-1
    ).float()
    msk_full = torch.tril(torch.ones(C_PTO, C_PTO, device=dev), diagonal=0).float()

    g_sum = torch.empty(1, T, H, device=dev, dtype=torch.float32)
    run_chunk_cumsum(
        g_in,
        g_sum,
        stream=stream,
        chunk_size=C_PTO,
        cu_seqlens=cu_seqlens,
        batch_size_override=N_seq,
    )

    g_t = _transpose_g(g_sum)
    beta_t = _transpose_beta(beta)
    torch.npu.synchronize()

    A_out = torch.zeros(1, T, H, C_PTO, device=dev, dtype=torch.float16)
    run_scaled_dot_kkt(
        k,
        beta,
        g_sum,
        msk_lower,
        None,
        A_out,
        stream=stream,
        g_t=g_t,
        beta_t=beta_t,
        chunk_size=C_PTO,
        cu_seqlens=cu_seqlens,
        batch_size_override=N_seq,
        key_heads=HG,
    )

    A_sol = pto_solve_tril(tri_inv_func, A_out, cu_seqlens, C_PTO, H)

    w_out = torch.empty_like(v)
    u_out = torch.empty_like(v)
    run_wy_fast(
        k,
        v,
        beta,
        g_sum,
        A_sol,
        w_out,
        u_out,
        stream=stream,
        g_t=g_t,
        beta_t=beta_t,
        chunk_size=C_PTO,
        cu_seqlens=cu_seqlens,
        batch_size_override=N_seq,
        key_heads=HG,
    )

    tc_n = total_chunks(N_seq, T, C_PTO, cu_seqlens)
    s_out = torch.zeros(tc_n * H, D_DEFAULT, D_DEFAULT, device=dev, dtype=torch.float16)
    v_new = torch.empty_like(v)
    fs_out = torch.zeros(N_seq * H, D_DEFAULT, D_DEFAULT, device=dev, dtype=torch.float16)
    run_chunk_h(
        k,
        w_out,
        u_out,
        g_sum,
        s_out,
        v_new,
        fs_out,
        stream=stream,
        g_t=g_t,
        chunk_size=C_PTO,
        cu_seqlens=cu_seqlens,
        batch_size_override=N_seq,
        key_heads=HG,
    )

    o_out = torch.empty_like(v)
    run_chunk_o(
        q,
        k,
        v_new,
        s_out,
        g_sum,
        msk_full,
        o_out,
        stream=stream,
        g_t=g_t,
        chunk_size=C_PTO,
        cu_seqlens=cu_seqlens,
        batch_size_override=N_seq,
        key_heads=HG,
    )
    del fs_out
    return o_out * scale


def run_triton_e2e(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_in: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.LongTensor,
    *,
    initial_state: torch.Tensor,
    scale: float,
    Hg: int,
) -> torch.Tensor:
    chunk_indices = prepare_chunk_indices(cu_seqlens, C_TRITON)
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, C_TRITON)

    g = chunk_local_cumsum(
        g_in,
        chunk_size=C_TRITON,
        cu_seqlens=cu_seqlens,
    )
    assert k.shape[2] == Hg == q.shape[2]

    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        beta=beta,
        g_cumsum=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=C_TRITON,
        output_dtype=torch.float32,
    )
    A = solve_tril(
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices_large_block=None,
        chunk_indices_bt=chunk_indices,
        output_dtype=k.dtype,
    )
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        g_cumsum=g,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=False,
        chunk_size=C_TRITON,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=C_TRITON,
    )
    return o


def _materialize_inputs(
    seed: int,
    T: int,
    H: int,
    Hg: int,
    D: int,
    cu_list: list[int],
    dev: torch.device,
):
    assert H % Hg == 0
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    q_cpu = torch.randn(1, T, Hg, D, generator=g)
    k_cpu = torch.randn(1, T, Hg, D, generator=g)
    v_cpu = torch.randn(1, T, H, D, generator=g)
    g_in_cpu = F.logsigmoid(torch.randn(1, T, H, generator=g))
    beta_cpu = torch.rand(1, T, H, generator=g)

    q_cpu, k_cpu = F.normalize(q_cpu, dim=-1, p=2), F.normalize(k_cpu, dim=-1, p=2)

    q_bf = q_cpu.to(dev, dtype=torch.bfloat16)
    k_bf = k_cpu.to(dev, dtype=torch.bfloat16)
    v_bf = v_cpu.to(dev, dtype=torch.bfloat16)
    g_bf = g_in_cpu.to(dev, dtype=torch.float32)
    beta_bf = beta_cpu.to(dev, dtype=torch.bfloat16)

    q_fp = q_cpu.to(dev, dtype=torch.float16)
    k_fp = k_cpu.to(dev, dtype=torch.float16)
    v_fp = v_cpu.to(dev, dtype=torch.float16)
    g_fp = g_in_cpu.to(dev, dtype=torch.float32)
    beta_fp = beta_cpu.to(dev, dtype=torch.float16)

    cu_long = torch.tensor(cu_list, dtype=torch.long, device=dev)
    cu32 = torch.tensor(cu_list, dtype=torch.int32, device=dev)

    N_seq = len(cu_list) - 1
    z_bf = torch.zeros(N_seq, H, D, D, device=dev, dtype=torch.bfloat16)

    scale = D**-0.5
    cpu_ref = (q_cpu, k_cpu, v_cpu, g_in_cpu, beta_cpu)
    return (q_bf, k_bf, v_bf, g_bf, beta_bf, z_bf, cu_long), (
        q_fp,
        k_fp,
        v_fp,
        g_fp,
        beta_fp,
        cu32,
    ), scale, cpu_ref


def _cpu_reference_pair(
    q_f32: torch.Tensor,
    k_f32: torch.Tensor,
    v_f32: torch.Tensor,
    g_in_f32: torch.Tensor,
    beta_f32: torch.Tensor,
    cu_list: list[int],
    *,
    scale: float,
    Hg: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU fp32 refs: PTO gated ``chunk_o`` vs FLA-gated grouped reference."""
    cu_cpu = torch.tensor(cu_list, dtype=torch.long)

    def _run(cs: int, chunk_o_fn):
        g_sum = ref_cumsum(g_in_f32, cs, cu_cpu)
        A = ref_kkt_group(k_f32, beta_f32, g_sum, cs, cu_cpu)
        A_sol = ref_solve_tril(A, cs, cu_cpu)
        w, u = ref_wy_group(k_f32, v_f32, beta_f32, A_sol, g_sum, cs, cu_cpu)
        h_st, v_new, _ = ref_chunk_h_group(k_f32, w, u, g_sum, cs, cu_cpu)
        o = chunk_o_fn(
            q_f32, k_f32, v_new, h_st, g_sum, cs, cu_cpu
        )
        return o * scale

    o_pto = _run(C_PTO, ref_chunk_o_group)
    o_tri = _run(C_TRITON, ref_chunk_o_group_fla)
    return o_pto, o_tri


def _rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.sqrt(((a - b) ** 2).mean()).item())


def _nrmse(rmse_v: float, std_ref: float) -> float:
    if std_ref <= 1e-12:
        return float("nan")
    return rmse_v / std_ref


def _mean_abs_tensor(t: torch.Tensor) -> float:
    return float(t.detach().float().abs().mean().item())


def _frac_elements_close(
    pred: torch.Tensor, ref: torch.Tensor, *, rtol: float, atol: float
) -> float:
    p = pred.detach().float().flatten()
    r = ref.detach().float().flatten()
    bound = atol + rtol * r.abs()
    return float((p.sub(r).abs() <= bound).float().mean().item())


def _quality_vs_ref(
    pred: torch.Tensor,
    ref: torch.Tensor,
    *,
    max_rmse_over_mean_abs: float,
    min_r2: float,
    min_pearson: float,
) -> tuple[bool, dict[str, float | bool | str]]:
    pred_f = pred.detach().float().cpu()
    ref_f = ref.detach().float().cpu()
    mean_abs_ref = _mean_abs_tensor(ref_f)
    rmse_v = _rmse(pred_f, ref_f)
    ratio = rmse_v / max(mean_abs_ref, 1e-15)
    std_ref = float(ref_f.std().item())
    r2 = r2_score(ref_f, pred_f)
    pr = pearson_r(pred_f, ref_f)
    frac = _frac_elements_close(pred_f, ref_f, rtol=RTOL_REF, atol=ATOL_REF)

    if mean_abs_ref < 1e-9:
        pass_ratio = rmse_v < 5e-4
        pass_r2 = True
        pass_pr = True
    else:
        pass_ratio = ratio <= max_rmse_over_mean_abs
        pass_r2 = (not np.isfinite(r2)) or std_ref < 1e-12 or r2 >= min_r2
        pass_pr = (not np.isfinite(pr)) or std_ref < 1e-12 or abs(pr) >= min_pearson

    ok = bool(pass_ratio and pass_r2 and pass_pr)
    return ok, {
        "mean_abs_ref": mean_abs_ref,
        "rmse": rmse_v,
        "rmse_over_mean_abs": ratio,
        "atol_effective": ATOL_REF,
        "r2": r2 if np.isfinite(r2) else float("nan"),
        "pearson": pr if np.isfinite(pr) else float("nan"),
        "frac_close": frac,
        "pass_rmse_ratio": pass_ratio,
        "pass_r2": pass_r2,
        "pass_pearson": pass_pr,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default=os.getenv("GDN_NPU_DEVICE", "npu:0"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--H",
        type=int,
        default=H_DEFAULT,
        help=f"Value head count (default {H_DEFAULT}; env GDN_GROUPVALUE_H)",
    )
    p.add_argument(
        "--hg",
        type=int,
        default=HG_DEFAULT,
        help=f"Shared Q/K head count Hg (default {HG_DEFAULT}; env GDN_HG)",
    )
    p.add_argument(
        "--fig-dir",
        default=None,
        help=f"Directory for scatter PNGs (default: {_DEFAULT_FIG_DIR})",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Alias for --fig-dir (deprecated)",
    )
    p.add_argument(
        "--csv-dir",
        default=None,
        help=f"Directory for error metric CSV (default: {_DEFAULT_CSV_DIR})",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib scatter figures",
    )
    args = p.parse_args()

    Hv, HG = args.H, args.hg
    if Hv % HG != 0:
        raise SystemExit(f"H={Hv} must be divisible by hg={HG}")

    fig_dir = args.fig_dir or args.out_dir or _DEFAULT_FIG_DIR
    csv_dir = args.csv_dir or _DEFAULT_CSV_DIR
    if not args.no_plots:
        os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    if "PTO_LIB_PATH" not in os.environ:
        fb = "/sources/pto-isa"
        if os.path.isdir(os.path.join(fb, "include")):
            os.environ["PTO_LIB_PATH"] = fb

    torch.manual_seed(args.seed)
    torch.npu.set_device(args.device)
    dev = torch.device(args.device)

    cpp = os.path.join(_FAST_INV, "fast_inverse.cpp")
    print(f"Compiling fast_inverse: {cpp}")
    tri_inv = jit_compile(cpp, verbose=False)
    print("Compilation OK.")

    cases: list[tuple[str, int, list[int]]] = [
        ("single seq T=128", 128, [0, 128]),
        ("single seq T=256", 256, [0, 256]),
        ("single seq T=512", 512, [0, 512]),
        ("single seq T=1024", 1024, [0, 1024]),
        ("single seq T=2048", 2048, [0, 2048]),
        ("single seq T=4096", 4096, [0, 4096]),
        ("varlen [256,256]", 512, [0, 256, 512]),
        ("varlen [128,128,128]", 384, [0, 128, 256, 384]),
        ("varlen 1×384", 384, [0, 384]),
        ("varlen [150,300] tails", 450, [0, 150, 450]),
        ("varlen [129,255] tails", 384, [0, 129, 384]),
        (
            "varlen [1,17,128,129,255] boundary mix",
            530,
            _cu_from_seqlens([1, 17, 128, 129, 255]),
        ),
        (
            "varlen [1,17,31,32,33,95,127,128,129,191,192,193,367] dense ladder",
            1536,
            _cu_from_seqlens([1, 17, 31, 32, 33, 95, 127, 128, 129, 191, 192, 193, 367]),
        ),
        (
            "varlen [128,256,384,512,768] long mix",
            2048,
            _cu_from_seqlens([128, 256, 384, 512, 768]),
        ),
        (
            "varlen [1,63,64,65,127,128,129,447,512,640,1920] long ladder",
            4096,
            _cu_from_seqlens([1, 63, 64, 65, 127, 128, 129, 447, 512, 640, 1920]),
        ),
    ]

    csv_rows: list[dict[str, object]] = []
    ok = 0
    for case_idx, (label, T, cu_list) in enumerate(cases):
        if cu_list is not None and cu_list[-1] != T:
            raise RuntimeError(f"bad case {label}")
        case_seed = args.seed + case_idx * 10_003
        tri_in, pto_in, scale, cpu_ref = _materialize_inputs(
            case_seed, T, Hv, HG, D_DEFAULT, cu_list, dev
        )
        q_bf, k_bf, v_bf, g_bf, beta_bf, z_bf, cu_long = tri_in
        q_fp, k_fp, v_fp, g_fp, beta_fp, cu32 = pto_in
        q_ref, k_ref, v_ref, g_ref, beta_ref = cpu_ref
        o_ref_pto, o_ref_tri = _cpu_reference_pair(
            q_ref, k_ref, v_ref, g_ref, beta_ref, cu_list, scale=scale, Hg=HG
        )

        torch.npu.synchronize()
        stream = torch.npu.current_stream()._as_parameter_
        o_pto = run_pto_e2e(
            q_fp,
            k_fp,
            v_fp,
            g_fp,
            beta_fp,
            cu32,
            stream=stream,
            tri_inv_func=tri_inv,
            scale=scale,
            H=Hv,
            HG=HG,
        )
        torch.npu.synchronize()
        o_tri = run_triton_e2e(
            q_bf,
            k_bf,
            v_bf,
            g_bf,
            beta_bf,
            cu_long,
            initial_state=z_bf,
            scale=scale,
            Hg=HG,
        )
        torch.npu.synchronize()

        pto_f = o_pto.float().cpu()
        tri_f = o_tri.float().cpu()
        refp = o_ref_pto.float()
        reft = o_ref_tri.float()

        qp = _quality_vs_ref(
            pto_f,
            refp,
            max_rmse_over_mean_abs=MAX_RMSE_OVER_MEAN_ABS_PTO,
            min_r2=MIN_R2_PTO,
            min_pearson=MIN_PEARSON_PTO,
        )
        ok_pto, mp = qp
        qt = _quality_vs_ref(
            tri_f,
            reft,
            max_rmse_over_mean_abs=MAX_RMSE_OVER_MEAN_ABS_TRI,
            min_r2=MIN_R2,
            min_pearson=MIN_PEARSON,
        )
        ok_tri, mt = qt
        qc = _quality_vs_ref(
            pto_f,
            tri_f,
            max_rmse_over_mean_abs=MAX_RMSE_OVER_MEAN_ABS_CROSS,
            min_r2=MIN_R2_CROSS,
            min_pearson=MIN_PEARSON_CROSS,
        )
        ok_cross, mc = qc
        rel_ok = ok_pto and ok_tri and ok_cross

        rmse_pto = float(mp["rmse"])
        rmse_tri = float(mt["rmse"])
        std_refp = float(refp.std().item())
        std_reft = float(reft.std().item())
        nrmse_pto = _nrmse(rmse_pto, std_refp)
        nrmse_tri = _nrmse(rmse_tri, std_reft)
        r2_pto = float(mp["r2"]) if np.isfinite(mp["r2"]) else float("nan")
        r2_tri = float(mt["r2"]) if np.isfinite(mt["r2"]) else float("nan")
        r_pto_tri = pearson_r(pto_f, tri_f)
        r_pto_ref = float(mp["pearson"]) if np.isfinite(mp["pearson"]) else float("nan")
        r_tri_ref = float(mt["pearson"]) if np.isfinite(mt["pearson"]) else float("nan")

        diff_cross = (pto_f - tri_f).abs()
        mx_cross = float(diff_cross.max().item())
        mean_cross = float(diff_cross.mean().item())
        rmse_cross = _rmse(pto_f, tri_f)

        r2_cross = r2_score(tri_f, pto_f)
        pr = f"{r_pto_ref:.4f}" if np.isfinite(r_pto_ref) else "nan"
        tr = f"{r_tri_ref:.4f}" if np.isfinite(r_tri_ref) else "nan"
        cr = (
            f"{float(mc['pearson']):.4f}"
            if np.isfinite(float(mc["pearson"]))
            else "nan"
        )
        hg_tag = f"H={Hv}_Hg={HG}_"
        print(
            f"{hg_tag}{label}: "
            f"PTO rmse/|ref|={mp['rmse_over_mean_abs']:.3f} r2={r2_pto:.4f} ρ={pr} "
            f"close%={100.0 * float(mp['frac_close']):.2f} ok={ok_pto} | "
            f"Tri rmse/|ref|={mt['rmse_over_mean_abs']:.4f} r2={r2_tri:.4f} ρ={tr} "
            f"close%={100.0 * float(mt['frac_close']):.2f} ok={ok_tri} | "
            f"PTO~Tri rmse/|tri|={mc['rmse_over_mean_abs']:.4f} r2={r2_cross:.4f} ρ={cr} "
            f"close%={100.0 * float(mc['frac_close']):.2f} ok={ok_cross}"
        )
        csv_rows.append(
            {
                "label": label,
                "H": Hv,
                "Hg": HG,
                "case_idx": case_idx,
                "T": T,
                "cu_seqlens": ",".join(str(x) for x in cu_list),
                "case_seed": case_seed,
                "mean_abs_ref_pto": mp["mean_abs_ref"],
                "mean_abs_ref_tri": mt["mean_abs_ref"],
                "rmse_pto_vs_ref": rmse_pto,
                "rmse_over_mean_abs_pto": mp["rmse_over_mean_abs"],
                "rmse_tri_vs_ref": rmse_tri,
                "rmse_over_mean_abs_tri": mt["rmse_over_mean_abs"],
                "nrmse_pto": nrmse_pto,
                "nrmse_tri": nrmse_tri,
                "atol_effective_pto": mp["atol_effective"],
                "atol_effective_tri": mt["atol_effective"],
                "frac_close_pto": mp["frac_close"],
                "frac_close_tri": mt["frac_close"],
                "r2_pto_vs_ref": r2_pto if np.isfinite(r2_pto) else "",
                "r2_tri_vs_ref": r2_tri if np.isfinite(r2_tri) else "",
                "ok_pto": ok_pto,
                "ok_tri": ok_tri,
                "rmse_pto_vs_tri": rmse_cross,
                "rmse_over_mean_abs_pto_vs_tri": mc["rmse_over_mean_abs"],
                "max_abs_pto_vs_tri": mx_cross,
                "mean_abs_pto_vs_tri": mean_cross,
                "frac_close_pto_vs_tri": mc["frac_close"],
                "r2_pto_vs_tri": r2_cross if np.isfinite(r2_cross) else "",
                "ok_pto_vs_tri": ok_cross,
                "pearson_pto_vs_tri": r_pto_tri if np.isfinite(r_pto_tri) else "",
                "pearson_pto_vs_ref": r_pto_ref if np.isfinite(r_pto_ref) else "",
                "pearson_tri_vs_ref": r_tri_ref if np.isfinite(r_tri_ref) else "",
                "std_ref_pto": std_refp,
                "std_ref_tri": std_reft,
                "gates_pass": rel_ok,
                "rtol": RTOL_REF,
                "atol_ref": ATOL_REF,
                "max_rmse_over_mean_abs_pto": MAX_RMSE_OVER_MEAN_ABS_PTO,
                "max_rmse_over_mean_abs_tri": MAX_RMSE_OVER_MEAN_ABS_TRI,
                "max_rmse_over_mean_abs_cross": MAX_RMSE_OVER_MEAN_ABS_CROSS,
                "device": str(dev),
                "fig_png": "",
            }
        )
        if not args.no_plots:
            png = os.path.join(fig_dir, f"{_safe_filename(hg_tag + label)}.png")
            plot_scatter_1to1(
                o_pto.detach().float().cpu(),
                o_tri.detach().float().cpu(),
                title=(
                    f"{hg_tag}{label}\nPTO rmse={rmse_pto:.4f}  Tri rmse={rmse_tri:.4f}  "
                    f"cross r²={r2_cross:.4f}"
                ),
                path=png,
            )
            print(f"  saved {png}")
            csv_rows[-1]["fig_png"] = png

        if not rel_ok:
            print("  FAIL: PTO-vs-ref, Triton-vs-ref, and/or PTO-vs-Triton gate failed")
        else:
            ok += 1

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(csv_dir, f"e2e_groupvalue_metrics_{ts}.csv")
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(csv_rows)
        latest = os.path.join(csv_dir, "e2e_groupvalue_metrics_latest.csv")
        with open(latest, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\nWrote metrics CSV: {csv_path}")
        print(f"Also: {latest}")

    print(
        f"\n{ok}/{len(cases)} cases passed "
        f"(H={Hv}, Hg={HG}; PTO-vs-ref, Triton-vs-ref, PTO-vs-Triton; "
        f"rtol={RTOL_REF}, atol={ATOL_REF}; gates: RMSE ratio, R², |ρ|)"
    )
    if not args.no_plots:
        print(f"Scatter plots: {fig_dir}")
    return 0 if ok == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
