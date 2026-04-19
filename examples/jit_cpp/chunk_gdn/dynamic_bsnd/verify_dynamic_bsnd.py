#!/usr/bin/env python3
"""
Numerical verification for dynamic BSND PTO kernels (H=16, D=128, C=128).

Tests each kernel stage against a PyTorch reference across many shape
combinations: fixed-length, variable-length, tail chunks, short/long
sequences, and random sequence length distributions.

All 5 stages are tested in pipeline order (each stage feeds into the
next).  A failure in an early stage will cascade to later ones.

Verifies:
  1. chunk_cumsum — chunk-local prefix sum
  2. scaled_dot_kkt — gated KK^T with mask and beta
  3. wy_fast — WY recompute (w, u) against the **same** KKT blocks as the kernel input
     (full FLA forward uses ``solve_tril`` first; see ``ref_solve_tril`` /
     ``ref_chunk_o_fla`` for CPU refs that match ``pto_e2e`` / Triton)
  4. chunk_h — chunkwise state recurrence (states, v_new, final_state)
  5. chunk_o — output; PTO uses ``exp(min(Δg,0))``; ``static_baseline/run_chunk_o_static.py``
     uses full ``exp(Δg)`` (see that script for a tiled reference)

Correctness (see ``torch.testing.assert_close`` defaults): ``rtol=1e-2`` is fine for
fp16/bf16 paths; **avoid large atol** (e.g. 1e-2) when activations are ~1e-2 — that
allows ~100% relative error. Here ``atol=1e-5`` always.

Per stage, pass if **either** (i) every element satisfies
``|a−e| ≤ atol + rtol·|e|`` with ``atol=1e-5``, ``rtol=1e-2``, **or** (ii) global
stats: ``rmse / mean(|e|)`` below a small cap **and** ``R² ≥ 0.99`` (handles a few
outliers that break strict allclose).

Regression targets:
  - Tail chunks, including ragged multi-sequence boundaries.
  - Sequential multi-case execution without subprocess isolation.

Per-stage agreement with the CPU reference is summarized by R² and Pearson ρ (see
``-v``) and optional 1:1 scatter PNGs (CPU ref on x, NPU on y) via ``--fig-dir``.
If min R² stays high for every stage but e2e PTO vs Triton is poor, the mismatch
is likely cross-backend (e.g. ``chunk_o`` gating), not PTO-vs-ref accuracy.

Usage:
  python verify_dynamic_bsnd.py --device npu:4
  python verify_dynamic_bsnd.py --device npu:4 --isolate   # each case in subprocess
  python verify_dynamic_bsnd.py --device npu:4 --quick
  python verify_dynamic_bsnd.py --device npu:4 --case 12 -v
  python verify_dynamic_bsnd.py --device npu:4 --fig-dir output/fig_stage_scatter
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field

_HERE = os.path.dirname(os.path.abspath(__file__))
_CHUNK_GDN = os.path.abspath(os.path.join(_HERE, ".."))
if _CHUNK_GDN not in sys.path:
    sys.path.insert(0, _CHUNK_GDN)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import torch
import torch.nn.functional as F

from dynamic_kernel_libs import (
    BLOCK_DIM,
    run_chunk_cumsum,
    run_chunk_o,
    run_chunk_h,
    run_scaled_dot_kkt,
    run_wy_fast,
    total_chunks,
)

C = 128
H, D = 16, 128

# Match ``torch.testing.assert_close``-style bf16 checks: tight atol, modest rtol.
RTOL_CHECK = 1e-2
ATOL_CHECK = 1e-5
# If strict elementwise bound fails (e.g. rare outliers), still pass when global fit is good:
MAX_RMSE_OVER_MEAN_ABS = 0.05  # RMSE should be ≪ typical |ref|; ~2 orders below ~0.5 scale
MIN_R2_FALLBACK = 0.99
HARD_FAIL_THRESHOLD = 1.0

# Scatter subsample size for per-stage 1:1 PNGs (CPU ref vs NPU kernel)
SCATTER_MAX_POINTS = 80_000
_DEFAULT_FIG_DIR = os.path.join(_HERE, "output", "fig_stage_scatter")


def r2_score_vs_ref(y_ref: torch.Tensor, y_pred: torch.Tensor) -> float:
    """R² with CPU reference on the ``y_ref`` axis: ``1 − SS_res/SS_tot``."""
    ref = np.asarray(y_ref.detach().cpu().numpy().ravel(), dtype=np.float64)
    pred = np.asarray(y_pred.detach().cpu().numpy().ravel(), dtype=np.float64)
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


def _scatter_subsample_pair(
    x: torch.Tensor, y: torch.Tensor, max_n: int
) -> tuple[torch.Tensor, torch.Tensor]:
    n = x.numel()
    if n <= max_n:
        return x.flatten(), y.flatten()
    idx = torch.randperm(n)[:max_n]
    return x.flatten()[idx], y.flatten()[idx]


def plot_scatter_ref_vs_kernel(
    expected: torch.Tensor,
    actual: torch.Tensor,
    *,
    title: str,
    path: str,
) -> None:
    """Scatter CPU reference (x) vs NPU kernel output (y) with a visual ``y = x`` line."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_t, y_t = _scatter_subsample_pair(
        expected.detach().float().cpu(),
        actual.detach().float().cpu(),
        SCATTER_MAX_POINTS,
    )
    x_np = np.asarray(x_t.numpy(), dtype=np.float64).ravel()
    y_np = np.asarray(y_t.numpy(), dtype=np.float64).ravel()

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
    ax.set_xlabel("CPU reference (flatten)")
    ax.set_ylabel("NPU kernel output (flatten)")
    ax.set_title(title)
    ax.grid(True, alpha=0.35, linestyle=":", linewidth=0.6)
    ax.legend(loc="lower right")
    fig.tight_layout()
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _safe_filename(label: str) -> str:
    s = re.sub(r"[^\w\-+.,=]+", "_", label)
    return s.strip("_")[:100] or "case"


# ───────────────────── Test case specification ─────────────────────────

@dataclass
class TestCase:
    label: str
    cu_seqlens_list: list[int] | None
    T: int
    known_crash: bool = False  # set True for cases that crash the NPU


def _rand_cu_seqlens(n_seq: int, total: int, rng: random.Random) -> list[int]:
    if n_seq == 1:
        return [0, total]
    bnd = sorted(rng.sample(range(1, total), n_seq - 1))
    return [0] + bnd + [total]


def _align_cu_seqlens(raw: list[int], cs: int) -> list[int]:
    aligned = [0]
    for i in range(1, len(raw) - 1):
        val = ((raw[i] + cs - 1) // cs) * cs
        if val <= aligned[-1]:
            val = aligned[-1] + cs
        aligned.append(val)
    total = max(raw[-1], aligned[-1] + cs)
    total = ((total + cs - 1) // cs) * cs
    aligned.append(total)
    return aligned


def _cu_from_seqlens(seqlens: list[int]) -> list[int]:
    cu = [0]
    for slen in seqlens:
        cu.append(cu[-1] + slen)
    return cu


def build_test_cases() -> list[TestCase]:
    c = []

    # Fixed-length (single sequence, no cu_seqlens)
    c.append(TestCase("fixed T=128 (1 chunk)", None, 128))
    c.append(TestCase("fixed T=256 (2 chunks)", None, 256))
    c.append(TestCase("fixed T=385 (tail 1)", None, 385))
    c.append(TestCase("fixed T=512 (4 chunks)", None, 512))
    c.append(TestCase("fixed T=1024 (8 chunks)", None, 1024))

    # Varlen: single sequence
    c.append(TestCase("varlen 1×128", [0, 128], 128))
    c.append(TestCase("varlen 1×256", [0, 256], 256))
    c.append(TestCase("varlen 1×384", [0, 384], 384))
    c.append(TestCase("varlen 1×512", [0, 512], 512))

    # Varlen: 2 sequences (chunk-aligned)
    c.append(TestCase("varlen [256,256]", [0, 256, 512], 512))
    c.append(TestCase("varlen [128,256]", [0, 128, 384], 384))
    c.append(TestCase("varlen [256,128]", [0, 256, 384], 384))
    c.append(TestCase("varlen [128,128]", [0, 128, 256], 256))
    c.append(TestCase("varlen [384,128]", [0, 384, 512], 512))
    c.append(TestCase("varlen [128,384]", [0, 128, 512], 512))

    # Varlen: 3+ sequences (chunk-aligned)
    c.append(TestCase("varlen [128,128,128]", [0, 128, 256, 384], 384))
    c.append(TestCase("varlen [128,256,128]", [0, 128, 384, 512], 512))
    c.append(TestCase("varlen [256,128,256,128]", [0, 256, 384, 640, 768], 768))

    # Tail chunks (seq_len not divisible by C=128)
    c.append(TestCase("varlen 1×200 (tail 72)", [0, 200], 200))
    c.append(TestCase("varlen 1×129 (tail 1)", [0, 129], 129))
    # Multi-sequence with non-aligned boundaries (previously crashing)
    c.append(TestCase("varlen [150,300] (tails)", [0, 150, 450], 450))
    c.append(TestCase("varlen [129,255] (tails)", [0, 129, 384], 384))
    c.append(TestCase(
        "varlen [1,17,128,129,255] (boundary mix)",
        _cu_from_seqlens([1, 17, 128, 129, 255]), 530,
    ))
    c.append(TestCase(
        "varlen [1,63,64,65,127,128,129,447] (ladder)",
        _cu_from_seqlens([1, 63, 64, 65, 127, 128, 129, 447]), 1024,
    ))
    c.append(TestCase(
        "varlen [1,17,31,32,33,95,127,128,129,191,192,193,367] (dense ladder)",
        _cu_from_seqlens([1, 17, 31, 32, 33, 95, 127, 128, 129, 191, 192, 193, 367]), 1536,
    ))

    # Random chunk-aligned
    rng = random.Random(42)
    for n_seq, total in [(3, 768), (7, 1792), (10, 2560)]:
        raw = _rand_cu_seqlens(n_seq, total, rng)
        aligned = _align_cu_seqlens(raw, C)
        c.append(TestCase(
            f"varlen {n_seq} seqs random T={aligned[-1]}",
            aligned, aligned[-1],
        ))

    return c


# ───────────────────── PyTorch references ──────────────────────────────

def _seq_ranges(T, cu_seqlens=None):
    if cu_seqlens is None:
        return [(0, T)]
    cu = cu_seqlens.tolist() if hasattr(cu_seqlens, 'tolist') else cu_seqlens
    return [(cu[i], cu[i + 1]) for i in range(len(cu) - 1)]


def ref_cumsum(g, cs, cu_seqlens=None):
    B, T, Hd = g.shape
    g32, out = g.float(), torch.zeros_like(g, dtype=torch.float32)
    for bos, eos in _seq_ranges(T, cu_seqlens):
        for j in range(0, eos - bos, cs):
            s, e = bos + j, min(bos + j + cs, eos)
            out[:, s:e, :] = g32[:, s:e, :].cumsum(dim=1)
    return out


def _safe_exp(x):
    return torch.where(x <= 0, torch.exp(x), torch.zeros_like(x))


def ref_kkt(k, beta, g_cumsum, cs, cu_seqlens=None):
    B, T, Hd, Dd = k.shape
    out = torch.zeros(B, T, Hd, cs, device=k.device, dtype=torch.float32)
    kf, bf, gf = k.float(), beta.float(), g_cumsum.float()
    for bos, eos in _seq_ranges(T, cu_seqlens):
        for j in range(0, eos - bos, cs):
            s, e = bos + j, min(bos + j + cs, eos)
            v = e - s
            for h in range(Hd):
                kc, gc = kf[0, s:e, h, :], gf[0, s:e, h]
                blk = (kc @ kc.T) * _safe_exp(gc[:, None] - gc[None, :]) * bf[0, s:e, h, None]
                mask = torch.arange(v, device=blk.device)[:, None] > torch.arange(v, device=blk.device)[None, :]
                out[0, s:e, h, :v] = blk * mask.float()
    return out


def ref_solve_tril(A: torch.Tensor, cs: int, cu_seqlens=None) -> torch.Tensor:
    """
    Triangular solve matching ``fast_inverse`` / ``pto_solve_tril`` layout (see
    ``fast_inverse/run_fast_inverse_varlen_like_triton.py::_reference_inverse``):
    for each chunk block ``[1, v, H, v]``, compute ``inv(transpose(block) + I)`` in
    the batched sense, then ``transpose`` back — **not** a raw ``inv(I+L)`` on the
    per-head ``[v,v]`` slice alone.
    """
    A64 = A.detach().cpu().double()
    out = torch.zeros_like(A64)
    for bos, eos in _seq_ranges(A.shape[1], cu_seqlens):
        for chunk_start in range(bos, eos, cs):
            actual_size = min(cs, eos - chunk_start)
            block = A64[
                :, chunk_start : chunk_start + actual_size, :, :actual_size
            ]
            eye = torch.eye(
                actual_size, dtype=torch.float64, device=A64.device
            )
            inv = torch.inverse(block.transpose(1, 2) + eye).transpose(1, 2)
            out[:, chunk_start : chunk_start + actual_size, :, :actual_size] = inv
    return out.to(device=A.device, dtype=A.dtype)


def ref_wy(k, v, beta, A, g_cumsum, cs, cu_seqlens=None):
    B, T, Hd, Kd = k.shape
    w = torch.zeros(B, T, Hd, Kd, device=k.device, dtype=torch.float32)
    u = torch.zeros(B, T, Hd, v.shape[-1], device=k.device, dtype=torch.float32)
    kf, vf, bf, Af, gf = k.float(), v.float(), beta.float(), A.float(), g_cumsum.float()
    for bos, eos in _seq_ranges(T, cu_seqlens):
        for j in range(0, eos - bos, cs):
            s, e = bos + j, min(bos + j + cs, eos)
            valid = e - s
            for h in range(Hd):
                Ab = Af[0, s:e, h, :valid]
                gc = gf[0, s:e, h]
                vb = vf[0, s:e, h, :] * bf[0, s:e, h, None]
                kb = kf[0, s:e, h, :] * bf[0, s:e, h, None] * torch.exp(gc)[:, None]
                u[0, s:e, h, :] = Ab @ vb
                w[0, s:e, h, :] = Ab @ kb
    return w.to(k.dtype), u.to(v.dtype)


def ref_chunk_h(k, w, u, g_cumsum, cs, cu_seqlens=None):
    B, T, Hd, Dd = k.shape
    kf, wf, uf, gf = k.float(), w.float(), u.float(), g_cumsum.float()
    ranges = _seq_ranges(T, cu_seqlens)
    N = len(ranges)
    cu_t = torch.tensor(cu_seqlens) if isinstance(cu_seqlens, list) else cu_seqlens
    tc = total_chunks(N, T, cs, cu_t)
    h_out = torch.zeros(tc, Hd, Dd, Dd, device=k.device, dtype=torch.float32)
    v_new = torch.zeros_like(uf)
    final = torch.zeros(N, Hd, Dd, Dd, device=k.device, dtype=torch.float32)
    ci_base = 0
    for si, (bos, eos) in enumerate(ranges):
        nc = (eos - bos + cs - 1) // cs
        for h in range(Hd):
            S = torch.zeros(Dd, Dd, device=k.device, dtype=torch.float32)
            for ci in range(nc):
                s, e = bos + ci * cs, min(bos + (ci + 1) * cs, eos)
                gc = gf[0, s:e, h]
                gl = gc[e - s - 1]
                h_out[ci_base + ci, h] = S.clone()
                vc = uf[0, s:e, h, :] - wf[0, s:e, h, :] @ S
                v_new[0, s:e, h, :] = vc
                kv = kf[0, s:e, h, :].T @ (vc * torch.exp(gl - gc)[:, None])
                S = torch.exp(gl) * S + kv
            final[si, h] = S
        ci_base += nc
    return h_out, v_new, final


def _qk_gate_pto(gc: torch.Tensor) -> torch.Tensor:
    """PTO dynamic ``chunk_o`` Vec: ``exp(min(g_row - g_col, 0))`` (matches device kernel)."""
    d = gc[:, None] - gc[None, :]
    return torch.exp(torch.minimum(d, torch.zeros_like(d)))


def _qk_gate_fla(gc: torch.Tensor) -> torch.Tensor:
    """Match Triton ``chunk_o`` / FLA: ``safe_exp(g_row - g_col)``."""
    return _safe_exp(gc[:, None] - gc[None, :])


def ref_chunk_o(q, k, v_new, h_states, g_cumsum, cs, cu_seqlens=None):
    """PTO NPU ``chunk_o`` gating (``exp(min(Δg,0))``); see ``static_baseline`` for full ``exp(Δg)``."""
    return _ref_chunk_o_gated(
        q, k, v_new, h_states, g_cumsum, cs, cu_seqlens, gate_fn=_qk_gate_pto
    )


def ref_chunk_o_fla(q, k, v_new, h_states, g_cumsum, cs, cu_seqlens=None):
    """Triton / FLA ``chunk_fwd_o`` semantics (``safe_exp`` on QK gate)."""
    return _ref_chunk_o_gated(
        q, k, v_new, h_states, g_cumsum, cs, cu_seqlens, gate_fn=_qk_gate_fla
    )


def _ref_chunk_o_gated(
    q, k, v_new, h_states, g_cumsum, cs, cu_seqlens, gate_fn
):
    B, T, Hd, Dd = q.shape
    qf, kf, vf, gf = q.float(), k.float(), v_new.float(), g_cumsum.float()
    o = torch.zeros_like(qf)
    ranges = _seq_ranges(T, cu_seqlens)
    ci_base = 0
    for bos, eos in ranges:
        nc = (eos - bos + cs - 1) // cs
        for h in range(Hd):
            for ci in range(nc):
                s, e = bos + ci * cs, min(bos + (ci + 1) * cs, eos)
                vlen = e - s
                qc, kc, vc, gc = (
                    qf[0, s:e, h, :],
                    kf[0, s:e, h, :],
                    vf[0, s:e, h, :],
                    gf[0, s:e, h],
                )
                inter = (qc @ h_states[ci_base + ci, h]) * torch.exp(gc)[:, None]
                qk = qc @ kc.T
                mask = torch.arange(vlen, device=qk.device)[:, None] >= torch.arange(
                    vlen, device=qk.device
                )[None, :]
                gate = gate_fn(gc)
                o[0, s:e, h, :] = inter + (qk * gate * mask.float()) @ vc
        ci_base += nc
    return o


# ───────────────────── Check result types ──────────────────────────────

@dataclass
class CheckResult:
    name: str
    passed: bool
    max_err: float
    mean_err: float
    hard_fail: bool = False
    r2: float | None = None
    pearson: float | None = None
    rmse_over_mean_abs: float | None = None
    pass_mode: str | None = None  # "allclose" | "stats" when passed; "fail" otherwise


@dataclass
class CaseResult:
    label: str
    passed: bool
    checks: list[CheckResult] = field(default_factory=list)
    error: str | None = None
    elapsed: float = 0.0

    def to_json(self) -> str:
        d = {"label": self.label, "passed": self.passed, "elapsed": self.elapsed}
        if self.error:
            d["error"] = self.error
        else:
            d["checks"] = []
            for c in self.checks:
                row = {
                    "name": c.name,
                    "passed": c.passed,
                    "max_err": c.max_err,
                    "mean_err": c.mean_err,
                    "hard_fail": c.hard_fail,
                    "r2": (
                        float(c.r2)
                        if c.r2 is not None and np.isfinite(c.r2)
                        else None
                    ),
                    "pearson": (
                        float(c.pearson)
                        if c.pearson is not None and np.isfinite(c.pearson)
                        else None
                    ),
                    "rmse_over_mean_abs": (
                        float(c.rmse_over_mean_abs)
                        if c.rmse_over_mean_abs is not None
                        and np.isfinite(c.rmse_over_mean_abs)
                        else None
                    ),
                    "pass_mode": c.pass_mode,
                }
                d["checks"].append(row)
        return json.dumps(d)

    @staticmethod
    def from_json(s: str) -> "CaseResult":
        d = json.loads(s)
        r = CaseResult(label=d["label"], passed=d["passed"], elapsed=d.get("elapsed", 0))
        if "error" in d:
            r.error = d["error"]
        else:
            checks: list[CheckResult] = []
            for c in d["checks"]:
                checks.append(
                    CheckResult(
                        name=c["name"],
                        passed=c["passed"],
                        max_err=c["max_err"],
                        mean_err=c["mean_err"],
                        hard_fail=c.get("hard_fail", False),
                        r2=c.get("r2"),
                        pearson=c.get("pearson"),
                        rmse_over_mean_abs=c.get("rmse_over_mean_abs"),
                        pass_mode=c.get("pass_mode"),
                    )
                )
            r.checks = checks
        return r


# ───────────────────── Single-case runner ──────────────────────────────

def run_single_case(
    tc: TestCase,
    dev: torch.device,
    *,
    fig_dir: str | None = None,
) -> CaseResult:
    checks: list[CheckResult] = []
    t0 = time.time()
    T = tc.T
    plot_prefix = _safe_filename(tc.label) if fig_dir else ""

    if tc.cu_seqlens_list is not None:
        cu = torch.tensor(tc.cu_seqlens_list, dtype=torch.int32, device=dev)
        N_seq = len(tc.cu_seqlens_list) - 1
    else:
        cu = None
        N_seq = 1

    torch.manual_seed(42)
    torch.npu.manual_seed(42)
    q = torch.randn(1, T, H, D, device=dev, dtype=torch.float16)
    k = torch.randn(1, T, H, D, device=dev, dtype=torch.float16)
    v = torch.randn(1, T, H, D, device=dev, dtype=torch.float16)
    q, k = F.normalize(q, dim=-1, p=2), F.normalize(k, dim=-1, p=2)
    g_in = F.logsigmoid(torch.randn(1, T, H, device=dev, dtype=torch.float32))
    beta = torch.rand(1, T, H, device=dev, dtype=torch.float16)
    cu_cpu = cu.cpu() if cu is not None else None

    def _chk(name, actual, expected):
        diff = (actual - expected).abs()
        mx, mn = diff.max().item(), diff.mean().item()
        exp_abs = expected.abs()
        bound = ATOL_CHECK + RTOL_CHECK * exp_abs
        pass_allclose = bool((diff <= bound).all().item())

        ref_1d = expected.float().flatten()
        mean_abs_ref = float(ref_1d.abs().mean().item())
        std_ref = float(ref_1d.std().item())
        rmse = float(torch.sqrt((diff.float().flatten() ** 2).mean()).item())
        ratio = rmse / max(mean_abs_ref, 1e-15)
        r2 = r2_score_vs_ref(expected, actual)
        pr = pearson_r(actual, expected)

        if mean_abs_ref < 1e-9:
            pass_stats = rmse < 5e-4
        elif std_ref < 1e-12:
            pass_stats = ratio <= MAX_RMSE_OVER_MEAN_ABS
        else:
            pass_stats = (
                ratio <= MAX_RMSE_OVER_MEAN_ABS
                and np.isfinite(r2)
                and r2 >= MIN_R2_FALLBACK
            )

        hard = mx > HARD_FAIL_THRESHOLD
        ok = (pass_allclose or pass_stats) and not hard
        if ok:
            mode = "allclose" if pass_allclose else "stats"
        else:
            mode = "fail"

        checks.append(
            CheckResult(
                name,
                ok,
                mx,
                mn,
                hard,
                r2,
                pr,
                ratio if mean_abs_ref >= 1e-9 else None,
                mode,
            )
        )
        if fig_dir and plot_prefix:
            r2s = f"{r2:.4f}" if np.isfinite(r2) else "nan"
            prs = f"{pr:.4f}" if np.isfinite(pr) else "nan"
            png = os.path.join(fig_dir, f"{plot_prefix}__{name}.png")
            plot_scatter_ref_vs_kernel(
                expected,
                actual,
                title=f"{tc.label}\n{name}  R²={r2s}  ρ={prs}",
                path=png,
            )

    def _fin(name, t):
        ok = torch.isfinite(t).all().item()
        if not ok:
            checks.append(CheckResult(name + "_finite", False, float('inf'), float('inf'), True))
        return ok

    # 1. cumsum
    g_sum = torch.empty(1, T, H, device=dev, dtype=torch.float32)
    run_chunk_cumsum(g_in, g_sum, chunk_size=C, cu_seqlens=cu, batch_size_override=N_seq)
    torch.npu.synchronize()
    _chk("cumsum", g_sum.float().cpu(), ref_cumsum(g_in.cpu(), C, cu_cpu))

    # 2. kkt
    msk = torch.tril(torch.ones(C, C, device=dev), diagonal=-1).float()
    A_out = torch.zeros(1, T, H, C, device=dev, dtype=torch.float16)
    run_scaled_dot_kkt(k, beta, g_sum, msk, None, A_out,
                       chunk_size=C, cu_seqlens=cu, batch_size_override=N_seq)
    torch.npu.synchronize()
    _chk("kkt", A_out.float().cpu(), ref_kkt(k.cpu(), beta.cpu(), g_sum.cpu(), C, cu_cpu))

    # 3. wy_fast — kernel is checked against KKT blocks (same tensor as stage 2).
    #    Full FLA / e2e uses ``solve_tril`` on ``A_out`` before this stage; see
    #    ``pto_e2e_measure/verify_pto_triton_e2e.py`` and ``ref_solve_tril``.
    w_out = torch.empty(1, T, H, D, device=dev, dtype=torch.float16)
    u_out = torch.empty(1, T, H, D, device=dev, dtype=torch.float16)
    run_wy_fast(k, v, beta, g_sum, A_out, w_out, u_out,
                chunk_size=C, cu_seqlens=cu, batch_size_override=N_seq)
    torch.npu.synchronize()
    w_ref, u_ref = ref_wy(k.cpu(), v.cpu(), beta.cpu(), A_out.cpu(), g_sum.cpu(), C, cu_cpu)
    _chk("wy_w", w_out.float().cpu(), w_ref.float())
    _chk("wy_u", u_out.float().cpu(), u_ref.float())

    # 4. chunk_h
    tc_n = total_chunks(N_seq, T, C, cu)
    s_out = torch.zeros(tc_n * H, D, D, device=dev, dtype=torch.float16)
    v_out = torch.empty(1, T, H, D, device=dev, dtype=torch.float16)
    fs_out = torch.zeros(N_seq * H, D, D, device=dev, dtype=torch.float16)
    run_chunk_h(k, w_out, u_out, g_sum, s_out, v_out, fs_out,
                chunk_size=C, cu_seqlens=cu, batch_size_override=N_seq)
    torch.npu.synchronize()
    _fin("h_states", s_out); _fin("h_vnew", v_out); _fin("h_fs", fs_out)
    h_ref, v_ref, fs_ref = ref_chunk_h(k.cpu(), w_out.cpu(), u_out.cpu(), g_sum.cpu(), C, cu_cpu)
    s_re = s_out.float().cpu().view(tc_n, H, D, D)
    _chk("h_states", s_re, h_ref.float())
    _chk("h_vnew", v_out.float().cpu(), v_ref.float())

    # 5. chunk_o
    msk2 = torch.tril(torch.ones(C, C, device=dev), diagonal=0).float()
    o_out = torch.empty(1, T, H, D, device=dev, dtype=torch.float16)
    run_chunk_o(q, k, v_out, s_out, g_sum, msk2, o_out,
                chunk_size=C, cu_seqlens=cu, batch_size_override=N_seq)
    torch.npu.synchronize()
    _fin("chunk_o", o_out)
    _chk(
        "chunk_o",
        o_out.float().cpu(),
        ref_chunk_o(q.cpu(), k.cpu(), v_out.cpu(), s_re, g_sum.cpu(), C, cu_cpu),
    )

    elapsed = time.time() - t0
    return CaseResult(label=tc.label, passed=all(c.passed for c in checks),
                      checks=checks, elapsed=elapsed)


# ───────────────────── Isolated subprocess runner ──────────────────────

def _run_isolated(
    case_idx: int,
    device: str,
    seed: int,
    fig_dir: str | None = None,
) -> CaseResult:
    """Run a single case in a fresh subprocess to avoid state leakage."""
    cmd = [
        sys.executable,
        __file__,
        "--device",
        device,
        "--seed",
        str(seed),
        "--case",
        str(case_idx),
        "--_json_output",
    ]
    if fig_dir:
        cmd.extend(["--fig-dir", fig_dir])
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                              cwd=_HERE)
        for line in proc.stdout.strip().split("\n"):
            if line.startswith("{"):
                return CaseResult.from_json(line)
        return CaseResult(label=f"case {case_idx}", passed=False,
                          error=f"no JSON output; stderr: {proc.stderr[-500:]}")
    except subprocess.TimeoutExpired:
        return CaseResult(label=f"case {case_idx}", passed=False, error="timeout")
    except Exception as e:
        return CaseResult(label=f"case {case_idx}", passed=False, error=str(e))


# ───────────────────── Main ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GDN dynamic BSND kernel verification")
    parser.add_argument("--device", default=os.getenv("GDN_NPU_DEVICE", "npu:0"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--case", type=int, default=None, help="Run only case N (1-indexed)")
    parser.add_argument("--isolate", action="store_true",
                        help="Run each case in a fresh subprocess (slower but avoids state leakage)")
    parser.add_argument("--include-crash", action="store_true",
                        help="Include cases known to crash the NPU (MTE out of range)")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fig-dir",
        default=None,
        help=(
            f"Write per-stage 1:1 scatter PNGs (CPU ref vs NPU) here; "
            f"omit to skip figures. Default suggestion: {_DEFAULT_FIG_DIR}"
        ),
    )
    parser.add_argument("--_json_output", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.npu.set_device(args.device)
    dev = torch.device(args.device)

    # JSON output mode for subprocess isolation
    if args._json_output:
        all_cases = build_test_cases()
        idx = (args.case or 1) - 1
        tc = all_cases[idx]
        try:
            result = run_single_case(tc, dev, fig_dir=args.fig_dir)
        except Exception as e:
            result = CaseResult(label=tc.label, passed=False, error=str(e))
        print(result.to_json())
        return

    fig_dir = args.fig_dir
    if fig_dir:
        os.makedirs(fig_dir, exist_ok=True)

    print(f"Device: {args.device}  H={H} D={D} C={C}  BLOCK_DIM={BLOCK_DIM}")
    print(
        f"Tolerances: rtol={RTOL_CHECK} atol={ATOL_CHECK} "
        f"(or stats: rmse/mean|ref|≤{MAX_RMSE_OVER_MEAN_ABS}, R²≥{MIN_R2_FALLBACK})"
    )
    if args.isolate:
        print("Mode: isolated subprocesses (no state leakage)")
    if fig_dir:
        print(f"Per-stage scatter PNGs (CPU ref x, NPU y): {fig_dir}")
    print()

    if args.quick:
        cases = [TestCase("quick: varlen 2×256", [0, 256, 512], 512)]
        case_indices = [None]
    elif args.case is not None:
        all_cases = build_test_cases()
        idx = args.case - 1
        if idx < 0 or idx >= len(all_cases):
            print(f"Invalid --case {args.case}, must be 1..{len(all_cases)}")
            sys.exit(1)
        cases = [all_cases[idx]]
        case_indices = [args.case]
    else:
        cases = build_test_cases()
        case_indices = list(range(1, len(cases) + 1))

    total = len(cases)
    n_pass, n_hard = 0, 0
    all_results: list[CaseResult] = []
    failed_results: list[CaseResult] = []

    print(f"Running {total} test case{'s' if total > 1 else ''}...")
    print("=" * 78)

    for i, (tc, ci) in enumerate(zip(cases, case_indices), 1):
        if tc.cu_seqlens_list is not None:
            seqlens = [tc.cu_seqlens_list[j+1] - tc.cu_seqlens_list[j]
                       for j in range(len(tc.cu_seqlens_list) - 1)]
            shape_info = f"T={tc.T} seqlens={seqlens}"
        else:
            shape_info = f"T={tc.T} (fixed-len)"
        print(f"[{i}/{total}] {tc.label}  ({shape_info})")

        if tc.known_crash and not args.include_crash:
            print(f"  SKIP  (known NPU crash — use --include-crash to run)")
            continue

        if args.isolate and ci is not None:
            result = _run_isolated(ci, args.device, args.seed, fig_dir=fig_dir)
            result.label = tc.label
        else:
            torch.npu.synchronize()
            torch.npu.empty_cache()
            try:
                result = run_single_case(tc, dev, fig_dir=fig_dir)
            except Exception as e:
                result = CaseResult(label=tc.label, passed=False, error=str(e))
                if args.verbose:
                    import traceback; traceback.print_exc()

        all_results.append(result)

        if result.error:
            print(f"  ERROR  {result.error}")
            failed_results.append(result)
            continue

        if args.verbose:
            for c in result.checks:
                tag = "PASS" if c.passed else ("HARD FAIL" if c.hard_fail else "FAIL")
                r2s = (
                    f"{c.r2:.4f}"
                    if c.r2 is not None and np.isfinite(c.r2)
                    else "nan"
                )
                prs = (
                    f"{c.pearson:.4f}"
                    if c.pearson is not None and np.isfinite(c.pearson)
                    else "nan"
                )
                rm = (
                    f"{c.rmse_over_mean_abs:.4f}"
                    if c.rmse_over_mean_abs is not None and np.isfinite(c.rmse_over_mean_abs)
                    else "n/a"
                )
                pmode = c.pass_mode or "?"
                print(
                    f"    {tag:9s} {c.name:15s}  max={c.max_err:.6f}  mean={c.mean_err:.6f}  "
                    f"R²={r2s}  ρ={prs}  rm/|ref|={rm}  [{pmode}]"
                )

        has_hard = any(c.hard_fail for c in result.checks)
        if result.passed:
            n_pass += 1
            print(f"  PASS  ({result.elapsed:.1f}s)")
        elif has_hard:
            n_hard += 1
            names = [c.name for c in result.checks if c.hard_fail]
            print(f"  HARD FAIL  ({result.elapsed:.1f}s)  kernel bug likely: {', '.join(names)}")
            failed_results.append(result)
        else:
            worst = max(result.checks, key=lambda c: c.max_err)
            print(f"  FAIL  ({result.elapsed:.1f}s)  worst: {worst.name} max={worst.max_err:.4f}")
            failed_results.append(result)

    print("=" * 78)
    print(f"\n{n_pass}/{total} passed, {n_hard} hard failures, "
          f"{len(failed_results) - n_hard} tolerance failures")

    if failed_results:
        print("\n── Failed cases ──")
        for r in failed_results:
            if r.error:
                print(f"  ERROR  {r.label}: {r.error}")
            else:
                failing = [c for c in r.checks if not c.passed]
                parts = [f"{c.name}({'HARD' if c.hard_fail else 'soft'} max={c.max_err:.4f})"
                         for c in failing]
                tag = "HARD" if any(c.hard_fail for c in failing) else "soft"
                print(f"  {tag:4s}  {r.label}: {', '.join(parts)}")

    # Max error summary across ALL results
    check_names = ["cumsum", "kkt", "wy_w", "wy_u", "h_states", "h_vnew", "chunk_o"]
    max_errs = {n: 0.0 for n in check_names}
    for r in all_results:
        for c in r.checks:
            if c.name in max_errs and not (c.max_err != c.max_err):  # skip nan
                max_errs[c.name] = max(max_errs[c.name], c.max_err)

    print("\n── Max error summary (across all cases) ──")
    for name in check_names:
        err = max_errs[name]
        if err > 0:
            flag = " *** KERNEL BUG?" if err > HARD_FAIL_THRESHOLD else ""
            print(f"  {name:15s}  max_err={err:.6f}{flag}")
        elif err == 0:
            print(f"  {name:15s}  max_err=0.000000")

    min_r2: dict[str, float] = {n: float("inf") for n in check_names}
    for r in all_results:
        if r.error:
            continue
        for c in r.checks:
            if c.name in min_r2 and c.r2 is not None and np.isfinite(c.r2):
                min_r2[c.name] = min(min_r2[c.name], c.r2)

    print("\n── Min R² vs CPU ref (across all cases; 1.0 = cloud on 1:1 line) ──")
    for name in check_names:
        v = min_r2[name]
        if v != float("inf") and v == v:
            flag = "  ** low vs ref" if v < 0.95 else ""
            print(f"  {name:15s}  min R²={v:.6f}{flag}")
        else:
            print(f"  {name:15s}  min R²=n/a")

    if n_hard > 0:
        sys.exit(2)
    elif failed_results:
        sys.exit(1)
    else:
        print("\nAll checks passed!")


if __name__ == "__main__":
    main()
