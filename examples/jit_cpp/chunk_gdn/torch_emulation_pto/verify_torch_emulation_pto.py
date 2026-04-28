#!/usr/bin/env python3
"""
Verify ``torch_emulation_pto`` against **CPU references** in ``verify_dynamic_bsnd.py``.

Compares the PTO-style emulation (explicit data-movement stand-ins in each module) to the same CPU
``ref_*`` math as ``verify_dynamic_bsnd``, via ``torch_emulation_pto.cpu_refs`` (pure PyTorch — does
**not** import ``verify_dynamic_bsnd`` or ``dynamic_kernel_libs``, which pull in kernel JIT and can
block for a long time). Each test case is bounded by ``--timeout`` (Unix) so a stuck run cannot hang
indefinitely.

For each test case we run:

- **e2e** — full emulation pipeline vs full reference chain.
- **iso** — each stage with **reference** upstream tensors so a failure isolates to one kernel.

Test cases are **diverse but modest in T** (largest packed length 448 here) so CPU stays fast;
patterns mirror ``verify_pto_triton_e2e`` (single/multi-seq, tails, boundary mix, ladders).

Pass criteria (same spirit as ``verify_dynamic_bsnd``): elementwise
``|a−e| ≤ atol + rtol·|e|`` with ``atol=1e-5``, ``rtol=1e-2``, **or** global fit
(``rmse/mean(|ref|)``, R²) when strict allclose fails on a few outliers.

Usage
-----
::

  cd examples/jit_cpp/chunk_gdn
  python torch_emulation_pto/verify_torch_emulation_pto.py
  python torch_emulation_pto/verify_torch_emulation_pto.py --quick
  python torch_emulation_pto/verify_torch_emulation_pto.py --smoke   # tiny finite-run check only
  python torch_emulation_pto/verify_torch_emulation_pto.py --quick --timeout 60
"""

from __future__ import annotations

import argparse
import contextlib
import os
import signal
import sys

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_CHUNK_GDN = os.path.abspath(os.path.join(_HERE, ".."))
_DYN = os.path.join(_CHUNK_GDN, "dynamic_bsnd")
for p in (_CHUNK_GDN, _DYN):
    if p not in sys.path:
        sys.path.insert(0, p)

from torch_emulation_pto import (  # noqa: E402
    chunk_cumsum_fwd,
    chunk_h_fwd,
    chunk_o_fwd,
    scaled_dot_kkt_fwd,
    wy_fast_fwd,
)
from torch_emulation_pto.cpu_refs import (  # noqa: E402 — avoids importing ``verify_dynamic_bsnd`` / ``dynamic_kernel_libs`` (slow JIT)
    ref_chunk_h,
    ref_chunk_o,
    ref_cumsum,
    ref_kkt,
    ref_wy,
)

C = 128
H, D = 16, 128

RTOL_CHECK = 1e-2
ATOL_CHECK = 1e-5
MAX_RMSE_OVER_MEAN_ABS = 0.05
MIN_R2_FALLBACK = 0.99
HARD_FAIL_THRESHOLD = 1.0


def _cu_from_seqlens(seqlens: list[int]) -> list[int]:
    cu = [0]
    for slen in seqlens:
        cu.append(cu[-1] + slen)
    return cu


def r2_score_vs_ref(y_ref: torch.Tensor, y: torch.Tensor) -> float:
    ref = np.asarray(y_ref.detach().cpu().numpy().ravel(), dtype=np.float64)
    pred = np.asarray(y.detach().cpu().numpy().ravel(), dtype=np.float64)
    ss_res = float(np.sum((ref - pred) ** 2))
    ss_tot = float(np.sum((ref - np.mean(ref)) ** 2))
    n = max(ref.size, 1)
    eps = 1e-30 * n
    if ss_tot <= eps:
        # ``chunk_h_states`` (and similar) can be **all zeros** when every chunk’s pre-state ``S`` is
        # zero — then total variance is 0 and the usual R² is undefined. Convention: 1.0 if no residual.
        return 1.0 if ss_res <= eps else 0.0
    return 1.0 - ss_res / ss_tot


def pearson_r(x: torch.Tensor, y: torch.Tensor) -> float:
    a = np.asarray(x.detach().cpu().numpy().ravel(), dtype=np.float64)
    b = np.asarray(y.detach().cpu().numpy().ravel(), dtype=np.float64)
    if a.size == 0:
        return float("nan")
    if a.size == 1:
        return 1.0 if np.isclose(a[0], b[0], rtol=0.0, atol=1e-12) else float("nan")
    std_a, std_b = float(np.std(a)), float(np.std(b))
    if std_a < 1e-15 and std_b < 1e-15:
        # Both constant (e.g. all-zero ``h_states``): ρ = 1 if identical, else undefined → 0.0
        return 1.0 if np.allclose(a, b, rtol=0.0, atol=1e-12) else 0.0
    if std_a < 1e-15 or std_b < 1e-15:
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(a, b)
    v = float(c[0, 1])
    return v if np.isfinite(v) else float("nan")


def check_stage(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> tuple[bool, str]:
    """``actual`` = ``torch_emulation_pto`` output; ``expected`` = ``ref_*`` from ``verify_dynamic_bsnd``."""
    diff = (actual.float() - expected.float()).abs()
    mx = float(diff.max().item())
    mn = float(diff.mean().item())
    exp_abs = expected.float().abs()
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
    mode = "allclose" if ok and pass_allclose else ("stats" if ok else "fail")
    msg = (
        f"{name}: max_err={mx:.3e} mean_err={mn:.3e} mode={mode} "
        f"rmse/mean|ref|={ratio:.3e} R2={r2:.4f} rho={pr:.4f}"
    )
    return ok, msg


def materialize_cpu(
    seed: int,
    T: int,
    cu_list: list[int],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.LongTensor | None,
    int,
]:
    """Returns ``q,k,v,g_in,beta`` on CPU (fp16 q/k/v/beta, fp32 g_in), ``cu_long``, ``N_seq``."""
    g = torch.Generator()
    g.manual_seed(seed)
    q = torch.randn(1, T, H, D, generator=g)
    k = torch.randn(1, T, H, D, generator=g)
    v = torch.randn(1, T, H, D, generator=g)
    g_in = F.logsigmoid(torch.randn(1, T, H, generator=g))
    beta = torch.rand(1, T, H, generator=g)
    q, k = F.normalize(q, dim=-1, p=2), F.normalize(k, dim=-1, p=2)
    q = q.half()
    k = k.half()
    v = v.half()
    beta = beta.half()
    g_in = g_in.float()
    N_seq = len(cu_list) - 1
    cu_long = torch.tensor(cu_list, dtype=torch.long)
    return q, k, v, g_in, beta, cu_long, N_seq


def run_emulation_cpu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_in: torch.Tensor,
    beta: torch.Tensor,
    cu_cpu: torch.LongTensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full five-kernel chain in fp32/fp16 on CPU (matches ``torch_emulation_pto``)."""
    g_sum = chunk_cumsum_fwd(g_in, C, cu_cpu)
    A = scaled_dot_kkt_fwd(k, beta, g_sum, C, cu_cpu)
    w, u = wy_fast_fwd(k, v, beta, A, g_sum, C, cu_cpu)
    h, v_new, fs = chunk_h_fwd(k, w, u, g_sum, C, cu_cpu)
    o = chunk_o_fwd(q, k, v_new, h, g_sum, C, cu_cpu)
    return g_sum, A, w, u, h, v_new, fs, o


def e2e_cases() -> list[tuple[str, int, list[int]]]:
    """Diverse ``cu_seqlens`` / tails; all ``T`` modest so CPU emulation is quick."""
    return [
        ("single seq T=128 (1 chunk)", 128, [0, 128]),
        ("single seq T=256 (2 chunks)", 256, [0, 256]),
        ("single seq T=385 (tail partial chunk)", 385, [0, 385]),
        ("varlen [128,128]", 256, [0, 128, 256]),
        ("varlen [128,128,128]", 384, [0, 128, 256, 384]),
        ("varlen 1×200 (tail 72)", 200, [0, 200]),
        ("varlen [75,150] tails", 225, [0, 75, 225]),
        ("varlen [65,128] tails", 193, [0, 65, 193]),
        (
            "varlen [1,17,64,65,127] boundary mix",
            274,
            _cu_from_seqlens([1, 17, 64, 65, 127]),
        ),
        (
            "varlen dense ladder (short)",
            370,
            _cu_from_seqlens([1, 17, 31, 32, 33, 64, 65, 127]),
        ),
        (
            "varlen multi-length mix",
            448,
            _cu_from_seqlens([64, 128, 96, 160]),
        ),
    ]


@contextlib.contextmanager
def _per_case_time_limit(seconds: float):
    """
    Wall-clock limit per test case (Unix). Uses ``SIGALRM`` / ``setitimer``; no-op on Windows or if
    ``seconds <= 0``. Prevents a stuck run from blocking forever when combined with CPU refs.
    """
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(signum, frame) -> None:  # noqa: ARG001
        raise TimeoutError(
            f"verify_torch_emulation_pto: case exceeded {seconds:g}s wall time "
            f"(raise --timeout or use --timeout 0 to disable)."
        )

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old)


def verify_one_case(
    idx: int,
    label: str,
    T: int,
    cu_list: list[int],
    seed: int,
) -> bool:
    """Single shape: e2e + iso vs ``cpu_refs`` (same math as ``verify_dynamic_bsnd``)."""
    if cu_list[-1] != T:
        raise RuntimeError(f"bad case {label}: cu[-1]={cu_list[-1]} != T={T}")
    q, k, v, g_in, beta, cu_cpu, N_seq = materialize_cpu(seed, T, cu_list)

    r_g = ref_cumsum(g_in, C, cu_cpu)
    r_A = ref_kkt(k, beta, r_g, C, cu_cpu)
    r_w, r_u = ref_wy(k, v, beta, r_A, r_g, C, cu_cpu)
    r_h, r_vn, r_fs = ref_chunk_h(k, r_w, r_u, r_g, C, cu_cpu)
    r_o = ref_chunk_o(q, k, r_vn, r_h, r_g, C, cu_cpu)

    e_g, e_A, e_w, e_u, e_h, e_vn, e_fs, e_o = run_emulation_cpu(
        q, k, v, g_in, beta, cu_cpu
    )

    print(
        f"\n=== Case {idx}: {label} (T={T}, N_seq={N_seq}) — CPU vs torch_emulation_pto.cpu_refs ==="
    )

    all_ok = True
    e2e_stages: list[tuple[str, torch.Tensor, torch.Tensor]] = [
        ("cumsum [e2e]", e_g, r_g),
        ("scaled_dot_kkt [e2e]", e_A, r_A),
        ("wy_w [e2e]", e_w, r_w),
        ("wy_u [e2e]", e_u, r_u),
        ("chunk_h_states [e2e]", e_h, r_h),
        ("chunk_h_v_new [e2e]", e_vn, r_vn),
        ("chunk_h_final [e2e]", e_fs, r_fs),
        ("chunk_o [e2e]", e_o, r_o),
    ]
    for name, a, e in e2e_stages:
        ok, msg = check_stage(name, a, e)
        all_ok = all_ok and ok
        print(("PASS" if ok else "FAIL"), msg)

    A_iso = scaled_dot_kkt_fwd(k, beta, r_g, C, cu_cpu)
    w_iso, u_iso = wy_fast_fwd(k, v, beta, r_A, r_g, C, cu_cpu)
    h_iso, vn_iso, fs_iso = chunk_h_fwd(k, r_w, r_u, r_g, C, cu_cpu)
    o_iso = chunk_o_fwd(q, k, r_vn, r_h, r_g, C, cu_cpu)

    iso_stages: list[tuple[str, torch.Tensor, torch.Tensor]] = [
        ("cumsum [iso]", e_g, r_g),
        ("scaled_dot_kkt [iso ref g]", A_iso, r_A),
        ("wy_w [iso ref A,g]", w_iso, r_w),
        ("wy_u [iso ref A,g]", u_iso, r_u),
        ("chunk_h_states [iso ref w,u,g]", h_iso, r_h),
        ("chunk_h_v_new [iso]", vn_iso, r_vn),
        ("chunk_h_final [iso]", fs_iso, r_fs),
        ("chunk_o [iso ref h,vn,g]", o_iso, r_o),
    ]
    for name, a, e in iso_stages:
        ok, msg = check_stage(name, a, e)
        all_ok = all_ok and ok
        print(("PASS" if ok else "FAIL"), msg)

    return all_ok


def verify_emulation_vs_refs(
    cases: list[tuple[str, int, list[int]]],
    seed: int,
    *,
    timeout_per_case: float,
) -> bool:
    """
    Compare ``torch_emulation_pto`` to the same CPU ``ref_*`` math as ``verify_dynamic_bsnd``,
    implemented in ``torch_emulation_pto.cpu_refs`` (no ``dynamic_kernel_libs`` import).

    For each case: **e2e** then **iso** (reference upstreams). Each case is wrapped in
    ``timeout_per_case`` seconds when > 0 (Unix).
    """
    all_ok = True
    for idx, (label, T, cu_list) in enumerate(cases):
        seed_i = seed + idx * 10_003
        try:
            with _per_case_time_limit(timeout_per_case):
                ok = verify_one_case(idx, label, T, cu_list, seed_i)
        except TimeoutError as ex:
            print(f"FAIL {label}: {ex}", file=sys.stderr)
            ok = False
        all_ok = all_ok and ok

    if all_ok:
        print("\nverify_torch_emulation_pto: all stages PASS vs CPU refs (cpu_refs).")
    else:
        print("\nverify_torch_emulation_pto: some stages FAILED vs CPU refs.", file=sys.stderr)
    return all_ok


def quick_cases() -> list[tuple[str, int, list[int]]]:
    """Minimal subset for fast iteration."""
    return [
        ("single seq T=128", 128, [0, 128]),
        ("varlen [75,150] tails", 225, [0, 75, 225]),
        (
            "varlen [1,17,64,65,127] boundary mix",
            274,
            _cu_from_seqlens([1, 17, 64, 65, 127]),
        ),
    ]


def smoke_emulation_only() -> None:
    """Sanity: emulation runs end-to-end on CPU."""
    q, k, v, g_in, beta, cu, _ns = materialize_cpu(0, 256, [0, 256])
    *_, o = run_emulation_cpu(q, k, v, g_in, beta, cu)
    assert torch.isfinite(o).all()
    print("verify_torch_emulation_pto: CPU smoke OK (emulation only).")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true", help="Run 3 representative shapes only")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Minimal finite-run smoke only (no ref_* suite)",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=None,
        metavar="SEC",
        help="Max wall seconds per test case (Unix SIGALRM). Default: 120 with --quick, 600 otherwise; 0 disables.",
    )
    args = p.parse_args()

    if args.smoke:
        smoke_emulation_only()
        return 0

    cases = quick_cases() if args.quick else e2e_cases()
    if args.timeout is None:
        timeout_per_case = 120.0 if args.quick else 600.0
    else:
        timeout_per_case = float(args.timeout)

    ok = verify_emulation_vs_refs(cases, args.seed, timeout_per_case=timeout_per_case)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
