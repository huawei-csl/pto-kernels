#!/usr/bin/env python3
"""
Numerical verification for GQA group-value BSND kernels (shared key heads ``Hg``,
value heads ``H``).

Stages (each checked vs a CPU fp32 reference using FLA-style ``head_g`` indexing):

  ``kkt`` — ``scaled_dot_kkt``
  ``chunk_h`` — recurrent chunk states / ``v_new``
  ``wy_fast`` — synthetic ``A`` tiles → ``w``, ``u``
  ``chunk_o`` — ``chunk_h`` on device → ``chunk_o`` vs CPU ref

Uses the same packed-varlen case list as ``dynamic_bsnd/verify_dynamic_bsnd.py``
(extended boundary mix). Same thresholds as upstream (``rtol=1e-2``, tight ``atol``).

Usage::

  cd chunk_gdn/dynamic_bsnd_groupvalue
  python3 verify_dynamic_bsnd_groupvalue.py --device npu:7
  python3 verify_dynamic_bsnd_groupvalue.py --device npu:7 --quick --stage kkt,chunk_h
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from dataclasses import dataclass

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

HG_DEFAULT = int(os.getenv("GDN_HG", "16"))

import numpy as np
import torch
import torch.nn.functional as F

from dynamic_kernel_libs import (
    BLOCK_DIM,
    _transpose_beta,
    _transpose_g,
    run_chunk_h,
    run_chunk_o,
    run_scaled_dot_kkt,
    run_wy_fast,
    total_chunks,
)

C = 128
D = 128

RTOL_CHECK = 1e-2
ATOL_CHECK = 1e-5
MAX_RMSE_OVER_MEAN_ABS = 0.05
MIN_R2_FALLBACK = 0.99
HARD_FAIL_THRESHOLD = 1.0


def _seq_ranges(T, cu_seqlens=None):
    if cu_seqlens is None:
        return [(0, T)]
    cu = cu_seqlens.tolist() if hasattr(cu_seqlens, "tolist") else cu_seqlens
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


def ref_kkt_group(k, beta, g_cumsum, cs, cu_seqlens=None):
    B, T, Hg, Dd = k.shape
    H = beta.shape[2]
    assert H % Hg == 0
    grp = H // Hg
    out = torch.zeros(B, T, H, cs, device=k.device, dtype=torch.float32)
    kf, bf, gf = k.float(), beta.float(), g_cumsum.float()
    for bos, eos in _seq_ranges(T, cu_seqlens):
        for j in range(0, eos - bos, cs):
            s, e = bos + j, min(bos + j + cs, eos)
            v = e - s
            for h in range(H):
                hg = h // grp
                kc = kf[0, s:e, hg, :]
                gc = gf[0, s:e, h]
                blk = (
                    (kc @ kc.T)
                    * _safe_exp(gc[:, None] - gc[None, :])
                    * bf[0, s:e, h, None]
                )
                mask = torch.arange(v, device=blk.device)[:, None] > torch.arange(
                    v, device=blk.device
                )[None, :]
                out[0, s:e, h, :v] = blk * mask.float()
    return out


def ref_chunk_h_group(k, w, u, g_cumsum, cs, cu_seqlens=None):
    B, T, Hg, Dd = k.shape
    H = w.shape[2]
    assert H % Hg == 0
    grp = H // Hg
    kf, wf, uf, gf = k.float(), w.float(), u.float(), g_cumsum.float()
    ranges = _seq_ranges(T, cu_seqlens)
    N = len(ranges)
    cu_t = torch.tensor(cu_seqlens) if isinstance(cu_seqlens, list) else cu_seqlens
    tc = total_chunks(N, T, cs, cu_t)
    h_out = torch.zeros(tc, H, Dd, Dd, device=k.device, dtype=torch.float32)
    v_new = torch.zeros_like(uf)
    final = torch.zeros(N, H, Dd, Dd, device=k.device, dtype=torch.float32)
    ci_base = 0
    for si, (bos, eos) in enumerate(ranges):
        nc = (eos - bos + cs - 1) // cs
        for h in range(H):
            hg = h // grp
            S = torch.zeros(Dd, Dd, device=k.device, dtype=torch.float32)
            for ci in range(nc):
                s, e = bos + ci * cs, min(bos + (ci + 1) * cs, eos)
                gc = gf[0, s:e, h]
                gl = gc[e - s - 1]
                h_out[ci_base + ci, h] = S.clone()
                vc = uf[0, s:e, h, :] - wf[0, s:e, h, :] @ S
                v_new[0, s:e, h, :] = vc
                kv = kf[0, s:e, hg, :].T @ (vc * torch.exp(gl - gc)[:, None])
                S = torch.exp(gl) * S + kv
            final[si, h] = S
        ci_base += nc
    return h_out, v_new, final


def ref_wy_group(k, v, beta, A, g_cumsum, cs, cu_seqlens=None):
    B, T, Hg, Kd = k.shape
    H = v.shape[2]
    assert H % Hg == 0
    grp = H // Hg
    w = torch.zeros(B, T, H, Kd, device=k.device, dtype=torch.float32)
    u = torch.zeros(B, T, H, v.shape[-1], device=k.device, dtype=torch.float32)
    kf, vf, bf, Af, gf = k.float(), v.float(), beta.float(), A.float(), g_cumsum.float()
    for bos, eos in _seq_ranges(T, cu_seqlens):
        for j in range(0, eos - bos, cs):
            s, e = bos + j, min(bos + j + cs, eos)
            valid = e - s
            for h in range(H):
                hg = h // grp
                Ab = Af[0, s:e, h, :valid]
                gc = gf[0, s:e, h]
                vb = vf[0, s:e, h, :] * bf[0, s:e, h, None]
                kb = (
                    kf[0, s:e, hg, :]
                    * bf[0, s:e, h, None]
                    * torch.exp(gc)[:, None]
                )
                u[0, s:e, h, :] = Ab @ vb
                w[0, s:e, h, :] = Ab @ kb
    return w.to(k.dtype), u.to(v.dtype)


def _qk_gate_pto(gc: torch.Tensor) -> torch.Tensor:
    d = gc[:, None] - gc[None, :]
    return torch.exp(torch.minimum(d, torch.zeros_like(d)))


def ref_chunk_o_group(q, k, v_new, h_states, g_cumsum, cs, cu_seqlens=None):
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
                gate = _qk_gate_pto(gc)
                o[0, s:e, h, :] = inter + (qk * gate * mask.float()) @ vc
        ci_base += nc
    return o


def r2_score_vs_ref(y_ref: torch.Tensor, y_pred: torch.Tensor) -> float:
    ref = np.asarray(y_ref.detach().cpu().numpy().ravel(), dtype=np.float64)
    pred = np.asarray(y_pred.detach().cpu().numpy().ravel(), dtype=np.float64)
    ss_res = float(np.sum((ref - pred) ** 2))
    ss_tot = float(np.sum((ref - np.mean(ref)) ** 2))
    if ss_tot <= 1e-30 * max(ref.size, 1):
        return float("nan")
    return 1.0 - ss_res / ss_tot


def stats_ok(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    diff = (actual - expected).abs()
    mx = diff.max().item()
    exp_abs = expected.abs()
    bound = ATOL_CHECK + RTOL_CHECK * exp_abs
    pass_allclose = bool((diff <= bound).all().item())
    ref_1d = expected.float().flatten()
    mean_abs_ref = float(ref_1d.abs().mean().item())
    rmse = float(torch.sqrt((diff.float().flatten() ** 2).mean()).item())
    ratio = rmse / max(mean_abs_ref, 1e-15)
    r2 = r2_score_vs_ref(expected, actual)
    std_ref = float(ref_1d.std().item())
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
    return (pass_allclose or pass_stats) and mx <= HARD_FAIL_THRESHOLD


@dataclass
class TestCase:
    label: str
    cu_seqlens_list: list[int] | None
    T: int


def _cu_from_seqlens(seqlens: list[int]) -> list[int]:
    cu = [0]
    for slen in seqlens:
        cu.append(cu[-1] + slen)
    return cu


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


def _rand_cu_seqlens(n_seq: int, total: int, rng: random.Random) -> list[int]:
    if n_seq == 1:
        return [0, total]
    bnd = sorted(rng.sample(range(1, total), n_seq - 1))
    return [0] + bnd + [total]


def build_test_cases() -> list[TestCase]:
    c = []
    c.append(TestCase("fixed T=128 (1 chunk)", None, 128))
    c.append(TestCase("fixed T=256 (2 chunks)", None, 256))
    c.append(TestCase("fixed T=385 (tail 1)", None, 385))
    c.append(TestCase("fixed T=512 (4 chunks)", None, 512))
    c.append(TestCase("fixed T=1024 (8 chunks)", None, 1024))
    c.append(TestCase("varlen 1×128", [0, 128], 128))
    c.append(TestCase("varlen 1×256", [0, 256], 256))
    c.append(TestCase("varlen 1×384", [0, 384], 384))
    c.append(TestCase("varlen 1×512", [0, 512], 512))
    c.append(TestCase("varlen [256,256]", [0, 256, 512], 512))
    c.append(TestCase("varlen [128,256]", [0, 128, 384], 384))
    c.append(TestCase("varlen [256,128]", [0, 256, 384], 384))
    c.append(TestCase("varlen [128,128]", [0, 128, 256], 256))
    c.append(TestCase("varlen [384,128]", [0, 384, 512], 512))
    c.append(TestCase("varlen [128,384]", [0, 128, 512], 512))
    c.append(TestCase("varlen [128,128,128]", [0, 128, 256, 384], 384))
    c.append(TestCase("varlen [128,256,128]", [0, 128, 384, 512], 512))
    c.append(TestCase("varlen [256,128,256,128]", [0, 256, 384, 640, 768], 768))
    c.append(TestCase("varlen 1×200 (tail 72)", [0, 200], 200))
    c.append(TestCase("varlen 1×129 (tail 1)", [0, 129], 129))
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
        _cu_from_seqlens([1, 17, 31, 32, 33, 95, 127, 128, 129, 191, 192, 193, 367]),
        1536,
    ))
    rng = random.Random(42)
    for n_seq, total in [(3, 768), (7, 1792), (10, 2560)]:
        raw = _rand_cu_seqlens(n_seq, total, rng)
        aligned = _align_cu_seqlens(raw, C)
        c.append(TestCase(
            f"varlen {n_seq} seqs random T={aligned[-1]}",
            aligned, aligned[-1],
        ))
    return c


def run_case_kkt(tc: TestCase, dev: torch.device, H: int, HG: int) -> bool:
    T = tc.T
    if tc.cu_seqlens_list is not None:
        cu = torch.tensor(tc.cu_seqlens_list, dtype=torch.int32, device=dev)
        N_seq = len(tc.cu_seqlens_list) - 1
    else:
        cu = None
        N_seq = 1
    torch.manual_seed(42)
    torch.npu.manual_seed(42)
    k = F.normalize(torch.randn(1, T, HG, D, device=dev, dtype=torch.float16), dim=-1, p=2)
    beta = torch.rand(1, T, H, device=dev, dtype=torch.float16)
    cu_cpu = cu.cpu() if cu is not None else None
    stream = torch.npu.current_stream()._as_parameter_
    g_in = F.logsigmoid(torch.randn(1, T, H, device=dev, dtype=torch.float32))
    g_sum = ref_cumsum(g_in.cpu(), C, cu_cpu).to(device=dev)
    g_t = _transpose_g(g_sum)
    beta_t = _transpose_beta(beta)
    msk = torch.tril(torch.ones(C, C, device=dev), diagonal=-1).float()
    A_out = torch.zeros(1, T, H, C, device=dev, dtype=torch.float16)
    torch.npu.synchronize()
    run_scaled_dot_kkt(
        k, beta, g_sum, msk, None, A_out,
        stream=stream,
        g_t=g_t, beta_t=beta_t,
        chunk_size=C,
        cu_seqlens=cu,
        batch_size_override=N_seq,
        key_heads=HG,
    )
    torch.npu.synchronize()
    ref = ref_kkt_group(k.cpu(), beta.cpu(), g_sum.cpu(), C, cu_cpu)
    return stats_ok(A_out.float().cpu(), ref)


def run_case_chunk_h(tc: TestCase, dev: torch.device, H: int, HG: int) -> bool:
    if tc.cu_seqlens_list is not None:
        cu = torch.tensor(tc.cu_seqlens_list, dtype=torch.int32, device=dev)
        N_seq = len(tc.cu_seqlens_list) - 1
    else:
        cu = None
        N_seq = 1
    T = tc.T
    torch.manual_seed(42)
    torch.npu.manual_seed(42)
    k = F.normalize(torch.randn(1, T, HG, D, device=dev, dtype=torch.float16), dim=-1, p=2)
    w = torch.randn(1, T, H, D, device=dev, dtype=torch.float16)
    u = torch.randn(1, T, H, D, device=dev, dtype=torch.float16)
    cu_cpu = cu.cpu() if cu is not None else None
    g_in = F.logsigmoid(torch.randn(1, T, H, device=dev, dtype=torch.float32))
    g_sum = ref_cumsum(g_in.cpu(), C, cu_cpu).to(device=dev)
    stream = torch.npu.current_stream()._as_parameter_
    g_t = g_sum.squeeze(0).t().contiguous()
    tc_n = total_chunks(N_seq, T, C, cu)
    s_out = torch.zeros(tc_n * H, D, D, device=dev, dtype=torch.float16)
    v_out = torch.empty(1, T, H, D, device=dev, dtype=torch.float16)
    fs_out = torch.zeros(N_seq * H, D, D, device=dev, dtype=torch.float16)
    torch.npu.synchronize()
    run_chunk_h(
        k, w, u, g_sum, s_out, v_out, fs_out,
        stream=stream,
        g_t=g_t,
        chunk_size=C,
        cu_seqlens=cu,
        batch_size_override=N_seq,
        key_heads=HG,
    )
    torch.npu.synchronize()
    h_ref, v_ref, _ = ref_chunk_h_group(
        k.cpu(), w.cpu(), u.cpu(), g_sum.cpu(), C, cu_cpu,
    )
    s_re = s_out.float().cpu().view(tc_n, H, D, D)
    ok_h = stats_ok(s_re, h_ref.float())
    ok_v = stats_ok(v_out.float().cpu(), v_ref.float())
    return ok_h and ok_v


def run_case_wy(tc: TestCase, dev: torch.device, H: int, HG: int) -> bool:
    if tc.cu_seqlens_list is not None:
        cu = torch.tensor(tc.cu_seqlens_list, dtype=torch.int32, device=dev)
        N_seq = len(tc.cu_seqlens_list) - 1
    else:
        cu = None
        N_seq = 1
    T = tc.T
    torch.manual_seed(42)
    torch.npu.manual_seed(42)
    k = F.normalize(torch.randn(1, T, HG, D, device=dev, dtype=torch.float16), dim=-1, p=2)
    v = torch.randn(1, T, H, D, device=dev, dtype=torch.float16)
    beta = torch.rand(1, T, H, device=dev, dtype=torch.float16)
    A = torch.randn(1, T, H, C, device=dev, dtype=torch.float16)
    cu_cpu = cu.cpu() if cu is not None else None
    g_in = F.logsigmoid(torch.randn(1, T, H, device=dev, dtype=torch.float32))
    g32 = g_in.float().cpu()
    g_sum = torch.zeros(1, T, H, dtype=torch.float32)
    for bos, eos in _seq_ranges(T, cu_cpu):
        for j in range(0, eos - bos, C):
            s, e = bos + j, min(bos + j + C, eos)
            g_sum[0, s:e, :] = g32[0, s:e, :].cumsum(dim=1)
    g_sum = g_sum.to(device=dev)
    stream = torch.npu.current_stream()._as_parameter_
    g_t = _transpose_g(g_sum)
    beta_t = _transpose_beta(beta)
    w_out = torch.empty(1, T, H, D, device=dev, dtype=torch.float16)
    u_out = torch.empty(1, T, H, D, device=dev, dtype=torch.float16)
    torch.npu.synchronize()
    run_wy_fast(
        k, v, beta, g_sum, A, w_out, u_out,
        stream=stream,
        g_t=g_t,
        beta_t=beta_t,
        chunk_size=C,
        cu_seqlens=cu,
        batch_size_override=N_seq,
        key_heads=HG,
    )
    torch.npu.synchronize()
    w_ref, u_ref = ref_wy_group(
        k.cpu(), v.cpu(), beta.cpu(), A.cpu(), g_sum.cpu(), C, cu_cpu,
    )
    ok_w = stats_ok(w_out.float().cpu(), w_ref.float())
    ok_u = stats_ok(u_out.float().cpu(), u_ref.float())
    return ok_w and ok_u


def run_case_chunk_o(tc: TestCase, dev: torch.device, H: int, HG: int) -> bool:
    if tc.cu_seqlens_list is not None:
        cu = torch.tensor(tc.cu_seqlens_list, dtype=torch.int32, device=dev)
        N_seq = len(tc.cu_seqlens_list) - 1
    else:
        cu = None
        N_seq = 1
    T = tc.T
    torch.manual_seed(42)
    torch.npu.manual_seed(42)
    k = F.normalize(torch.randn(1, T, HG, D, device=dev, dtype=torch.float16), dim=-1, p=2)
    q = F.normalize(torch.randn(1, T, HG, D, device=dev, dtype=torch.float16), dim=-1, p=2)
    w = torch.randn(1, T, H, D, device=dev, dtype=torch.float16)
    u = torch.randn(1, T, H, D, device=dev, dtype=torch.float16)
    cu_cpu = cu.cpu() if cu is not None else None
    g_in = F.logsigmoid(torch.randn(1, T, H, device=dev, dtype=torch.float32))
    g_sum = ref_cumsum(g_in.cpu(), C, cu_cpu).to(device=dev)
    stream = torch.npu.current_stream()._as_parameter_
    g_t = g_sum.squeeze(0).t().contiguous()
    tc_n = total_chunks(N_seq, T, C, cu)
    s_out = torch.zeros(tc_n * H, D, D, device=dev, dtype=torch.float16)
    v_out = torch.empty(1, T, H, D, device=dev, dtype=torch.float16)
    fs_out = torch.zeros(N_seq * H, D, D, device=dev, dtype=torch.float16)
    torch.npu.synchronize()
    run_chunk_h(
        k, w, u, g_sum, s_out, v_out, fs_out,
        stream=stream,
        g_t=g_t,
        chunk_size=C,
        cu_seqlens=cu,
        batch_size_override=N_seq,
        key_heads=HG,
    )
    torch.npu.synchronize()
    msk2 = torch.tril(torch.ones(C, C, device=dev), diagonal=0).float()
    o_out = torch.empty(1, T, H, D, device=dev, dtype=torch.float16)
    torch.npu.synchronize()
    run_chunk_o(
        q, k, v_out, s_out, g_sum, msk2, o_out,
        stream=stream,
        g_t=g_t,
        chunk_size=C,
        cu_seqlens=cu,
        batch_size_override=N_seq,
        key_heads=HG,
    )
    torch.npu.synchronize()
    s_re = s_out.float().cpu().view(tc_n, H, D, D)
    o_ref = ref_chunk_o_group(
        q.cpu(), k.cpu(), v_out.cpu(), s_re, g_sum.cpu(), C, cu_cpu,
    )
    return stats_ok(o_out.float().cpu(), o_ref.float())


_STAGE_FUNCS = {
    "kkt": ("scaled_dot_kkt", run_case_kkt),
    "chunk_h": ("chunk_h", run_case_chunk_h),
    "wy_fast": ("wy_fast", run_case_wy),
    "chunk_o": ("chunk_o", run_case_chunk_o),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=os.getenv("GDN_NPU_DEVICE", "npu:0"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--H-list",
        default="16,32,48,64",
        help="Comma-separated value head counts",
    )
    parser.add_argument(
        "--hg",
        type=int,
        default=HG_DEFAULT,
        help="Key head count Hg (also GDN_HG)",
    )
    parser.add_argument(
        "--stage",
        default="kkt,chunk_h,wy_fast,chunk_o",
        help="Comma-separated: kkt, chunk_h, wy_fast, chunk_o",
    )
    args = parser.parse_args()

    stages = []
    for raw in args.stage.split(","):
        s = raw.strip()
        if not s:
            continue
        if s not in _STAGE_FUNCS:
            sys.stderr.write(f"Unknown stage {s!r}; choose from {list(_STAGE_FUNCS)}\n")
            sys.exit(2)
        stages.append(s)

    torch.npu.set_device(args.device)
    dev = torch.device(args.device)
    heads_list = [int(x.strip()) for x in args.H_list.split(",") if x.strip()]
    HG = args.hg

    cases = (
        [TestCase("quick fixed T=128", None, 128)]
        if args.quick
        else build_test_cases()
    )

    print(
        f"Device {args.device}  stages={stages}  H in {heads_list}  "
        f"Hg={HG}  D={D}  C={C}  BLOCK_DIM={BLOCK_DIM}",
    )
    ok_all = True
    for stage in stages:
        name, fn = _STAGE_FUNCS[stage]
        print(f"\n{'=' * 60}\nStage: {name}\n{'=' * 60}")
        for H in heads_list:
            assert H % HG == 0, f"H={H} must be divisible by Hg={HG}"
            print(f"\n--- Value heads H={H} ---")
            for i, tc in enumerate(cases):
                t0 = time.time()
                ok = fn(tc, dev, H, HG)
                dt = time.time() - t0
                status = "PASS" if ok else "FAIL"
                if not ok:
                    ok_all = False
                print(f"  [{i+1}/{len(cases)}] {status}  {tc.label}  ({dt:.2f}s)")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
