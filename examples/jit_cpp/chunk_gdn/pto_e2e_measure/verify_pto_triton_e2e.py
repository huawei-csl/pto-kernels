#!/usr/bin/env python3
"""
End-to-end GDN: PTO chain (``C=128``) + ``fast_inverse`` vs Triton (``C=64``).

**Pass criteria:** both backends must agree with their float32 CPU references, and the
final PTO output must also agree directly with the Triton output. We use fixed
``atol=1e-5``, ``rtol=1e-2`` (see ``torch.testing.assert_close``); the primary gates are
``rmse / mean(|ref|)``, ``R²`` and Pearson ``ρ``. ``frac_close`` (share of elements
within the rtol/atol band) is reported for context but is not the primary gate.

In this end-to-end chain, the corrected PTO ``chunk_o`` gating matches Triton on the
causal domain exercised by the model, so direct PTO-vs-Triton agreement is expected.

Q/K are L2-normalized in float32 before casting to fp16/bf16.

``cu_seqlens`` is always passed explicitly so Triton ``wy_fast`` uses the varlen
path.

Pipeline (both):
  cumsum -> scaled_dot_kkt -> solve_tril -> wy_fast -> chunk_h -> chunk_o

Usage:
  cd examples/jit_cpp/chunk_gdn/pto_e2e_measure
  python verify_pto_triton_e2e.py --device npu:4

  Default outputs: ``output/fig/*.png`` (scatter), ``csv/e2e_metrics_<UTC>.csv`` and
  ``csv/e2e_metrics_latest.csv`` (metrics). Override with ``--fig-dir`` / ``--csv-dir``.
  ``--no-plots`` skips PNGs but still writes CSV.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_FIG_DIR = os.path.join(_HERE, "output", "fig")
_DEFAULT_CSV_DIR = os.path.join(_HERE, "csv")
_CHUNK_GDN = os.path.abspath(os.path.join(_HERE, ".."))
_JIT_CPP = os.path.abspath(os.path.join(_CHUNK_GDN, ".."))
_DYN = os.path.join(_CHUNK_GDN, "dynamic_bsnd")
_FAST_INV = os.path.join(_JIT_CPP, "fast_inverse")

for p in (_CHUNK_GDN, _DYN, _FAST_INV):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn.functional as F

from dynamic_kernel_libs import (
    BLOCK_DIM,
    run_chunk_cumsum,
    run_chunk_h,
    run_chunk_o,
    run_scaled_dot_kkt,
    run_wy_fast,
    total_chunks,
)
from jit_util_fast_inverse import jit_compile

from verify_dynamic_bsnd import (
    ref_chunk_h,
    ref_chunk_o,
    ref_chunk_o_fla,
    ref_cumsum,
    ref_kkt,
    ref_solve_tril,
    ref_wy,
)

from triton_baseline.fla_vendor.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from triton_baseline.fla_vendor.chunk_o import chunk_fwd_o
from triton_baseline.fla_vendor.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from triton_baseline.fla_vendor.cumsum import chunk_local_cumsum
from triton_baseline.fla_vendor.solve_tril import solve_tril
from triton_baseline.fla_vendor.utils import prepare_chunk_indices, prepare_chunk_offsets
from triton_baseline.fla_vendor.wy_fast import recompute_w_u_fwd

# PTO dynamic kernels are built and tested at C=128; Triton uses C=64 (solve_tril).
C_PTO = 128
C_TRITON = 64
H_DEFAULT, D_DEFAULT = 16, 128

# Element band for reporting only (tight atol — avoid atol ~1e-2 on ~1e-2 activations)
RTOL_REF = 1e-2
ATOL_REF = 1e-5
# rmse / mean(abs(ref)) must be < this (Triton: <0.1 ⇒ RMSE well below mean |ref|)
MAX_RMSE_OVER_MEAN_ABS_TRI = 0.09
MAX_RMSE_OVER_MEAN_ABS_PTO = 0.15
MIN_R2 = 0.99
MIN_PEARSON = 0.995
# PTO fp16 vs float32 ref: same R² target; RMSE cap may be slightly looser.
MIN_R2_PTO = 0.99
MIN_PEARSON_PTO = 0.995
# PTO vs Triton should be much tighter than either backend vs CPU fp32 ref.
MAX_RMSE_OVER_MEAN_ABS_CROSS = 0.02
MIN_R2_CROSS = 0.999
MIN_PEARSON_CROSS = 0.999

# Scatter plot: max points (random subsample if larger)
SCATTER_MAX_POINTS = 80_000


def r2_score(y_ref: torch.Tensor, y: torch.Tensor) -> float:
    """R² with ``y_ref`` as the reference: ``1 − SS_res/SS_tot`` (sklearn-style)."""
    ref = np.asarray(y_ref.detach().cpu().numpy().ravel(), dtype=np.float64)
    pred = np.asarray(y.detach().cpu().numpy().ravel(), dtype=np.float64)
    ss_res = float(np.sum((ref - pred) ** 2))
    ss_tot = float(np.sum((ref - np.mean(ref)) ** 2))
    if ss_tot <= 1e-30 * max(ref.size, 1):
        return float("nan")
    return 1.0 - ss_res / ss_tot


def pearson_r(x: torch.Tensor, y: torch.Tensor) -> float:
    """Pearson r between flattened ``x`` and ``y`` (``numpy.corrcoef``)."""
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
    """Scatter ``out`` (x) vs ``out_ref`` (y) with a visual 1:1 line (PTO vs Triton)."""
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
    # Same data range on both axes + square subplot so the diagonal is a true 45° line.
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


def _transpose_valid_chunks(
    A: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    transposed = torch.zeros_like(A)
    for bos, eos in zip(
        cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=False
    ):
        for chunk_start in range(bos, eos, chunk_size):
            actual_size = min(chunk_size, eos - chunk_start)
            chunk = A[:, chunk_start : chunk_start + actual_size, :, :actual_size]
            transposed[:, chunk_start : chunk_start + actual_size, :, :actual_size] = (
                chunk.transpose(1, 3)
            )
    return transposed


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
    """(I+L)^{-1} in BSND layout; returns fp16 same shape as ``A_fp16``."""
    A_wrk = _transpose_valid_chunks(A_fp16, cu_seqlens, chunk_size)
    num_matrices = _count_varlen_chunks(cu_seqlens, chunk_size) * num_heads
    tensor_out = torch.zeros_like(A_fp16, dtype=torch.float32)
    minus_identity = _make_minus_identity(chunk_size, A_fp16.device)
    torch.npu.synchronize()
    tri_inv_func(
        tensor_out,
        A_wrk,
        minus_identity,
        chunk_size,
        num_matrices,
        num_heads,
        cu_seqlens=cu_seqlens,
        block_dim=BLOCK_DIM,
    )
    torch.npu.synchronize()
    out = _transpose_valid_chunks(tensor_out.to(torch.float16), cu_seqlens, chunk_size)
    return out


def run_pto_e2e(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_in: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    tri_inv_func,
    scale: float,
) -> torch.Tensor:
    """q,k,v,beta,g_in on NPU fp16; cu_seqlens int32 [N+1] boundaries."""
    dev = q.device
    N_seq = len(cu_seqlens) - 1
    T = q.shape[1]

    msk_lower = torch.tril(
        torch.ones(C_PTO, C_PTO, device=dev), diagonal=-1
    ).float()
    msk_full = torch.tril(torch.ones(C_PTO, C_PTO, device=dev), diagonal=0).float()

    g_sum = torch.empty(1, T, H_DEFAULT, device=dev, dtype=torch.float32)
    run_chunk_cumsum(
        g_in,
        g_sum,
        chunk_size=C_PTO,
        cu_seqlens=cu_seqlens,
        batch_size_override=N_seq,
    )

    A_out = torch.zeros(1, T, H_DEFAULT, C_PTO, device=dev, dtype=torch.float16)
    run_scaled_dot_kkt(
        k,
        beta,
        g_sum,
        msk_lower,
        None,
        A_out,
        chunk_size=C_PTO,
        cu_seqlens=cu_seqlens,
        batch_size_override=N_seq,
    )

    A_sol = pto_solve_tril(tri_inv_func, A_out, cu_seqlens, C_PTO, H_DEFAULT)

    w_out = torch.empty_like(k)
    u_out = torch.empty_like(v)
    run_wy_fast(
        k,
        v,
        beta,
        g_sum,
        A_sol,
        w_out,
        u_out,
        chunk_size=C_PTO,
        cu_seqlens=cu_seqlens,
        batch_size_override=N_seq,
    )

    tc_n = total_chunks(N_seq, T, C_PTO, cu_seqlens)
    s_out = torch.zeros(tc_n * H_DEFAULT, D_DEFAULT, D_DEFAULT, device=dev, dtype=torch.float16)
    v_new = torch.empty_like(v)
    fs_out = torch.zeros(N_seq * H_DEFAULT, D_DEFAULT, D_DEFAULT, device=dev, dtype=torch.float16)
    run_chunk_h(
        k,
        w_out,
        u_out,
        g_sum,
        s_out,
        v_new,
        fs_out,
        chunk_size=C_PTO,
        cu_seqlens=cu_seqlens,
        batch_size_override=N_seq,
    )

    o_out = torch.empty_like(q)
    run_chunk_o(
        q,
        k,
        v_new,
        s_out,
        g_sum,
        msk_full,
        o_out,
        chunk_size=C_PTO,
        cu_seqlens=cu_seqlens,
        batch_size_override=N_seq,
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
) -> torch.Tensor:
    """Triton path: bf16 tensors, chunk size ``C_TRITON`` (FLA solve_tril)."""
    chunk_indices = prepare_chunk_indices(cu_seqlens, C_TRITON)
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, C_TRITON)

    g = chunk_local_cumsum(
        g_in,
        chunk_size=C_TRITON,
        cu_seqlens=cu_seqlens,
    )
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
    D: int,
    cu_list: list[int],
    dev: torch.device,
):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    q_cpu = torch.randn(1, T, H, D, generator=g)
    k_cpu = torch.randn(1, T, H, D, generator=g)
    v_cpu = torch.randn(1, T, H, D, generator=g)
    g_in_cpu = F.logsigmoid(torch.randn(1, T, H, generator=g))
    beta_cpu = torch.rand(1, T, H, generator=g)

    # Normalize Q/K in float32 *before* casting so fp16 and bf16 paths share the
    # same directions (normalizing per-dtype was dominating PTO–Triton error).
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Float32 CPU refs: PTO chunk_o gate vs FLA ``ref_chunk_o_fla`` (Triton)."""
    cu_cpu = torch.tensor(cu_list, dtype=torch.long)

    def _run(cs: int, chunk_o_fn):
        g_sum = ref_cumsum(g_in_f32, cs, cu_cpu)
        A = ref_kkt(k_f32, beta_f32, g_sum, cs, cu_cpu)
        A_sol = ref_solve_tril(A, cs, cu_cpu)
        w, u = ref_wy(k_f32, v_f32, beta_f32, A_sol, g_sum, cs, cu_cpu)
        h_st, v_new, _ = ref_chunk_h(k_f32, w, u, g_sum, cs, cu_cpu)
        o = chunk_o_fn(
            q_f32, k_f32, v_new, h_st, g_sum, cs, cu_cpu
        )
        return o * scale

    o_pto = _run(C_PTO, ref_chunk_o)
    o_tri = _run(C_TRITON, ref_chunk_o_fla)
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
    """Fraction of elements with ``|pred−ref| ≤ atol + rtol·|ref|``."""
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
    """Gate: RMSE ≪ mean(|ref|), R², Pearson (no required element-close fraction)."""
    pred_f = pred.detach().float().cpu()
    ref_f = ref.detach().float().cpu()
    mean_abs_ref = _mean_abs_tensor(ref_f)
    rmse_v = _rmse(pred_f, ref_f)
    ratio = rmse_v / max(mean_abs_ref, 1e-15)
    std_ref = float(ref_f.std().item())
    r2 = r2_score(ref_f, pred_f)
    pr = pearson_r(pred_f, ref_f)
    frac = _frac_elements_close(pred_f, ref_f, rtol=RTOL_REF, atol=ATOL_REF)

    # Degenerate reference (≈ constant zero): only absolute RMSE
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

    # Always pass cumulative lengths so Triton wy_fast uses IS_VARLEN (see module doc).
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
            case_seed, T, H_DEFAULT, D_DEFAULT, cu_list, dev
        )
        q_bf, k_bf, v_bf, g_bf, beta_bf, z_bf, cu_long = tri_in
        q_fp, k_fp, v_fp, g_fp, beta_fp, cu32 = pto_in
        q_ref, k_ref, v_ref, g_ref, beta_ref = cpu_ref
        o_ref_pto, o_ref_tri = _cpu_reference_pair(
            q_ref, k_ref, v_ref, g_ref, beta_ref, cu_list, scale=scale
        )

        torch.npu.synchronize()
        o_pto = run_pto_e2e(
            q_fp,
            k_fp,
            v_fp,
            g_fp,
            beta_fp,
            cu32,
            tri_inv_func=tri_inv,
            scale=scale,
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
        print(
            f"{label}: "
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
            png = os.path.join(fig_dir, f"{_safe_filename(label)}.png")
            plot_scatter_1to1(
                o_pto.detach().float().cpu(),
                o_tri.detach().float().cpu(),
                title=(
                    f"{label}\nPTO rmse={rmse_pto:.4f}  Tri rmse={rmse_tri:.4f}  "
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
    csv_path = os.path.join(csv_dir, f"e2e_metrics_{ts}.csv")
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(csv_rows)
        latest = os.path.join(csv_dir, "e2e_metrics_latest.csv")
        with open(latest, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\nWrote metrics CSV: {csv_path}")
        print(f"Also: {latest}")

    print(
        f"\n{ok}/{len(cases)} cases passed "
        f"(PTO-vs-ref, Triton-vs-ref, PTO-vs-Triton; "
        f"rtol={RTOL_REF}, atol={ATOL_REF}; gates: RMSE ratio, R², |ρ|)"
    )
    if not args.no_plots:
        print(f"Scatter plots: {fig_dir}")
    return 0 if ok == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
