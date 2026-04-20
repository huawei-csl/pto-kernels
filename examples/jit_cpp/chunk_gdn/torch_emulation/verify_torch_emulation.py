"""
Compare ``torch_emulation`` against Triton ``fla_vendor`` kernels (same dtypes / layouts).

For ``chunk_gated_delta_rule_fwd_h`` and ``chunk_fwd_o``, Triton bf16 matmul ordering can
differ slightly from PyTorch; we accept either ``torch.allclose`` (tight) or high :math:`R^2`
and low relative RMSE (vs Triton as reference).

Also checks that the ``cu_seqlens is None`` emulation path matches the packed layout with a
single full-length segment ``cu = [0, T]`` (see ``verify_emulation_none_vs_packed``): Triton
is not used there because the varlen Triton API requires ``cu_seqlens``.

Run from ``chunk_gdn`` with ``PYTHONPATH`` including this directory's parent (see repo README).

Uses ``npu:7`` by default (override with ``GDN_TRITON_NPU_DEVICE``).
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_CHUNK_GDN = os.path.dirname(_ROOT)
if _CHUNK_GDN not in sys.path:
    sys.path.insert(0, _CHUNK_GDN)

import torch
import torch.nn.functional as F

from torch_emulation._common import relative_rmse, tensor_r2_score
from torch_emulation.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from torch_emulation.chunk_o import chunk_fwd_o
from torch_emulation.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from torch_emulation.cumsum import chunk_local_cumsum
from torch_emulation.solve_tril import solve_tril
from torch_emulation.wy_fast import recompute_w_u_fwd

from triton_baseline.fla_vendor.chunk_delta_h import chunk_gated_delta_rule_fwd_h as chunk_gated_delta_rule_fwd_h_tr
from triton_baseline.fla_vendor.chunk_o import chunk_fwd_o as chunk_fwd_o_tr
from triton_baseline.fla_vendor.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd as chunk_scaled_dot_kkt_fwd_tr
from triton_baseline.fla_vendor.cumsum import chunk_local_cumsum as chunk_local_cumsum_tr
from triton_baseline.fla_vendor.solve_tril import solve_tril as solve_tril_tr
from triton_baseline.fla_vendor.wy_fast import recompute_w_u_fwd as recompute_w_u_fwd_tr
from triton_baseline.fla_vendor.utils import prepare_chunk_indices, prepare_chunk_offsets

NPU_DEVICE = os.getenv("GDN_TRITON_NPU_DEVICE", "npu:7")
CHUNK_SIZE = 64
RTOL, ATOL = 1e-2, 1e-5

# Emulation vs emulation (same dtype math): tight
EMU_RTOL, EMU_ATOL = 1e-5, 1e-6

# When ``allclose`` is too strict (bf16 / fused matmul), require strong agreement on these metrics
# (Triton output = reference for R² and relative RMSE).
R2_MIN = 0.9995
REL_RMSE_MAX = 0.05
# ``chunk_gated_delta_rule_fwd_h`` ``h`` can disagree on elements where Triton rounds to ~0 but
# emulation is still small-but-nonzero; global R² is then meaningless. Compare on |ref| > eps.
MASK_REF_ABS = 1e-5


def _cu_from_seqlens(seqlens: list[int]) -> list[int]:
    cu = [0]
    for slen in seqlens:
        cu.append(cu[-1] + slen)
    return cu


# (name, segment lengths) — total T = sum(segments). Inspired by ``verify_pto_triton_e2e`` cases.
#
# Every segment length must be a multiple of ``CHUNK_SIZE`` (64): the current torch
# emulation of ``chunk_scaled_dot_kkt`` / ``wy_fast`` / ``solve_tril`` truncates each
# sequence to ``length - (length % BT)``, while Triton still runs partial tail chunks via
# ``chunk_indices``. Misaligned lengths are not comparable until emulation matches that.
TRITON_VS_EMU_CASES: list[tuple[str, list[int]]] = [
    ("single seq T=128", [128]),
    ("single seq T=256", [256]),
    ("single seq T=512", [512]),
    ("single seq T=1024", [1024]),
    ("single seq T=2048", [2048]),
    ("single seq T=4096", [4096]),
    ("varlen [256,256]", [256, 256]),
    ("varlen [128,128,128]", [128, 128, 128]),
    ("varlen 1×384", [384]),
    # Aligned analogues of tail / many-segment stress (e2e-style), all lengths % 64 == 0
    ("varlen [128,320] two segments", [128, 320]),
    ("varlen [128,256] two segments", [128, 256]),
    (
        "varlen [64,64,128,128,256] boundary-style mix",
        [64, 64, 128, 128, 256],
    ),
    (
        "varlen [64,128,192,256,320] dense ladder aligned",
        [64, 128, 192, 256, 320],
    ),
    (
        "varlen [128,256,384,512,768] long mix",
        [128, 256, 384, 512, 768],
    ),
    (
        "varlen [64,128,192,256,320,384,448,512,576,640,704,768] long ladder aligned",
        [64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768],
    ),
]


def _assert_close_or_metrics(
    name: str,
    reference: torch.Tensor,
    prediction: torch.Tensor,
    *,
    rtol: float,
    atol: float,
    r2_min: float,
    rel_rmse_max: float,
    mask_if_global_r2_bad: bool = False,
) -> None:
    rf = reference.float()
    pf = prediction.float()
    if torch.allclose(rf, pf, rtol=rtol, atol=atol):
        return
    r2 = tensor_r2_score(reference, prediction)
    rr = relative_rmse(reference, prediction)
    if r2 >= r2_min and rr <= rel_rmse_max:
        print(
            f"  {name}: allclose rtol={rtol} atol={atol} failed; "
            f"R2={r2:.6f} rel_RMSE={rr:.6f} (thresholds R2>={r2_min}, rel_RMSE<={rel_rmse_max}) — OK"
        )
        return
    if mask_if_global_r2_bad:
        m = rf.abs() > MASK_REF_ABS
        if m.any():
            r2m = tensor_r2_score(reference[m], prediction[m])
            rrm = relative_rmse(reference[m], prediction[m])
            if r2m >= r2_min and rrm <= rel_rmse_max:
                print(
                    f"  {name}: allclose failed; global R2={r2:.6f} rel_RMSE={rr:.6f}; "
                    f"on |ref|>{MASK_REF_ABS}: R2={r2m:.6f} rel_RMSE={rrm:.6f} — OK"
                )
                return
    raise AssertionError(
        f"{name}: max abs={torch.max(torch.abs(rf - pf)).item():.6g}, "
        f"R2={r2:.6f} (need >={r2_min}), rel_RMSE={rr:.6f} (need <={rel_rmse_max})"
    )


def _assert_emulation_close(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    if not torch.allclose(a.float(), b.float(), rtol=EMU_RTOL, atol=EMU_ATOL):
        d = (a.float() - b.float()).abs().max().item()
        raise AssertionError(f"{name}: max abs diff={d} (emu vs emu)")


def _build_inputs(
    *,
    dev: torch.device,
    t: int,
    h: int,
    dk: int,
    dv: int,
    n_seq: int,
    seed: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    float,
]:
    torch.manual_seed(seed)
    q = torch.randn(1, t, h, dk, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, t, h, dk, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, t, h, dv, device=dev, dtype=torch.bfloat16)
    g_in = F.logsigmoid(torch.randn(1, t, h, device=dev, dtype=torch.float32))
    beta = torch.rand(1, t, h, device=dev, dtype=torch.bfloat16)
    initial_state = torch.zeros(n_seq, h, dk, dv, device=dev, dtype=torch.bfloat16)
    scale = dk**-0.5
    return q, k, v, g_in, beta, initial_state, scale


def verify_emulation_none_vs_packed(dev: torch.device) -> None:
    """
    ``cu_seqlens is None`` must match packed ``cu = [0, T]`` when ``T`` is a multiple of
    ``CHUNK_SIZE``, so segment ranges agree with the ``None`` branch
    (``0 .. t - (t % BT)`` equals ``0 .. T``).
    """
    h, dk, dv = 4, 32, 32
    t = 256
    assert t % CHUNK_SIZE == 0
    q, k, v, g_in, beta, initial_state, scale = _build_inputs(
        dev=dev, t=t, h=h, dk=dk, dv=dv, n_seq=1, seed=2026
    )

    cu = torch.tensor([0, t], dtype=torch.long, device=dev)
    ci = prepare_chunk_indices(cu, CHUNK_SIZE)
    co = prepare_chunk_offsets(cu, CHUNK_SIZE)

    g_n = chunk_local_cumsum(g_in, chunk_size=CHUNK_SIZE, cu_seqlens=None)
    g_p = chunk_local_cumsum(g_in, chunk_size=CHUNK_SIZE, cu_seqlens=cu)
    _assert_emulation_close("chunk_local_cumsum (none vs packed [0,T])", g_n, g_p)

    a_n = chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g_cumsum=g_n, cu_seqlens=None, output_dtype=torch.float32
    )
    a_p = chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g_cumsum=g_p, cu_seqlens=cu, output_dtype=torch.float32
    )
    _assert_emulation_close("chunk_scaled_dot_kkt_fwd", a_n, a_p)

    w_n, u_n = recompute_w_u_fwd(
        k=k, v=v, beta=beta, A=a_n, g_cumsum=g_n, cu_seqlens=None, chunk_indices=None
    )
    w_p, u_p = recompute_w_u_fwd(
        k=k, v=v, beta=beta, A=a_p, g_cumsum=g_p, cu_seqlens=cu, chunk_indices=ci
    )
    _assert_emulation_close("recompute_w_u w", w_n, w_p)
    _assert_emulation_close("recompute_w_u u", u_n, u_p)

    s_n = solve_tril(A=a_n, cu_seqlens=None, output_dtype=k.dtype)
    s_p = solve_tril(A=a_p, cu_seqlens=cu, output_dtype=k.dtype)
    _assert_emulation_close("solve_tril", s_n, s_p)

    w2_n, u2_n = recompute_w_u_fwd(
        k=k, v=v, beta=beta, A=s_n, g_cumsum=g_n, cu_seqlens=None, chunk_indices=None
    )
    w2_p, u2_p = recompute_w_u_fwd(
        k=k, v=v, beta=beta, A=s_p, g_cumsum=g_p, cu_seqlens=cu, chunk_indices=ci
    )
    _assert_emulation_close("recompute_w_u (solved) w", w2_n, w2_p)
    _assert_emulation_close("recompute_w_u (solved) u", u2_n, u2_p)

    h_n, vn_n, _ = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w2_n,
        u=u2_n,
        g=g_n,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=None,
        chunk_indices=None,
        chunk_offsets=None,
    )
    h_p, vn_p, _ = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w2_p,
        u=u2_p,
        g=g_p,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=cu,
        chunk_indices=ci,
        chunk_offsets=co,
    )
    _assert_emulation_close("chunk_gated_delta_rule_fwd_h h", h_n, h_p)
    _assert_emulation_close("chunk_gated_delta_rule_fwd_h v_new", vn_n, vn_p)

    o_n = chunk_fwd_o(
        q=q, k=k, v=vn_n, h=h_n, g=g_n, scale=scale, cu_seqlens=None
    )
    o_p = chunk_fwd_o(
        q=q, k=k, v=vn_p, h=h_p, g=g_p, scale=scale, cu_seqlens=cu
    )
    _assert_emulation_close("chunk_fwd_o", o_n, o_p)


def run_triton_vs_emulation_case(
    dev: torch.device,
    case_name: str,
    seqlens: list[int],
    seed: int,
) -> None:
    t = sum(seqlens)
    n_seq = len(seqlens)
    h, dk, dv = 4, 32, 32
    cu = torch.tensor(_cu_from_seqlens(seqlens), dtype=torch.long, device=dev)
    chunk_indices = prepare_chunk_indices(cu, CHUNK_SIZE)
    chunk_offsets = prepare_chunk_offsets(cu, CHUNK_SIZE)

    q, k, v, g_in, beta, initial_state, scale = _build_inputs(
        dev=dev, t=t, h=h, dk=dk, dv=dv, n_seq=n_seq, seed=seed
    )

    g_tr = chunk_local_cumsum_tr(g_in, chunk_size=CHUNK_SIZE, cu_seqlens=cu)
    g_em = chunk_local_cumsum(g_in, chunk_size=CHUNK_SIZE, cu_seqlens=cu)
    assert torch.allclose(g_tr.float(), g_em.float(), rtol=RTOL, atol=ATOL), f"{case_name}: chunk_local_cumsum"

    a_tr = chunk_scaled_dot_kkt_fwd_tr(
        k=k,
        beta=beta,
        g_cumsum=g_tr,
        cu_seqlens=cu,
        output_dtype=torch.float32,
    )
    a_em = chunk_scaled_dot_kkt_fwd(
        k=k,
        beta=beta,
        g_cumsum=g_tr,
        cu_seqlens=cu,
        output_dtype=torch.float32,
    )
    assert torch.allclose(a_tr.float(), a_em.float(), rtol=RTOL, atol=ATOL), f"{case_name}: chunk_scaled_dot_kkt_fwd"

    w_tr, u_tr = recompute_w_u_fwd_tr(
        k=k,
        v=v,
        beta=beta,
        A=a_tr,
        g_cumsum=g_tr,
        cu_seqlens=cu,
        chunk_indices=chunk_indices,
    )
    w_em, u_em = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=a_tr,
        g_cumsum=g_tr,
        cu_seqlens=cu,
        chunk_indices=chunk_indices,
    )
    assert torch.allclose(w_tr.float(), w_em.float(), rtol=RTOL, atol=ATOL), f"{case_name}: recompute_w_u w"
    assert torch.allclose(u_tr.float(), u_em.float(), rtol=RTOL, atol=ATOL), f"{case_name}: recompute_w_u u"

    a_s_tr = solve_tril_tr(A=a_tr, cu_seqlens=cu, output_dtype=k.dtype)
    a_s_em = solve_tril(A=a_tr, cu_seqlens=cu, output_dtype=k.dtype)
    _assert_close_or_metrics(
        f"{case_name} solve_tril",
        a_s_tr,
        a_s_em,
        rtol=RTOL,
        atol=ATOL,
        r2_min=R2_MIN,
        rel_rmse_max=REL_RMSE_MAX,
        mask_if_global_r2_bad=False,
    )

    w2_tr, u2_tr = recompute_w_u_fwd_tr(
        k=k,
        v=v,
        beta=beta,
        A=a_s_tr,
        g_cumsum=g_tr,
        cu_seqlens=cu,
        chunk_indices=chunk_indices,
    )
    # Use the same solved ``A`` as Triton so this step tests ``wy_fast`` emulation only;
    # tiny ``solve_tril`` diffs would otherwise dominate the matmul (see ``solve_tril`` check above).
    w2_em, u2_em = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=a_s_tr,
        g_cumsum=g_tr,
        cu_seqlens=cu,
        chunk_indices=chunk_indices,
    )
    assert torch.allclose(w2_tr.float(), w2_em.float(), rtol=RTOL, atol=ATOL), f"{case_name}: recompute_w_u (solved) w"
    assert torch.allclose(u2_tr.float(), u2_em.float(), rtol=RTOL, atol=ATOL), f"{case_name}: recompute_w_u (solved) u"

    h_m_tr, v_new_tr, _ = chunk_gated_delta_rule_fwd_h_tr(
        k=k,
        w=w2_tr,
        u=u2_tr,
        g=g_tr,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=cu,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )
    h_m_em, v_new_em, _ = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w2_tr,
        u=u2_tr,
        g=g_tr,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=cu,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )
    _assert_close_or_metrics(
        f"{case_name} chunk_gated_delta_rule_fwd_h h",
        h_m_tr,
        h_m_em,
        rtol=RTOL,
        atol=ATOL,
        r2_min=R2_MIN,
        rel_rmse_max=REL_RMSE_MAX,
        mask_if_global_r2_bad=True,
    )
    _assert_close_or_metrics(
        f"{case_name} chunk_gated_delta_rule_fwd_h v_new",
        v_new_tr,
        v_new_em,
        rtol=RTOL,
        atol=ATOL,
        r2_min=R2_MIN,
        rel_rmse_max=REL_RMSE_MAX,
        mask_if_global_r2_bad=False,
    )

    o_tr = chunk_fwd_o_tr(
        q=q,
        k=k,
        v=v_new_tr,
        h=h_m_tr,
        g=g_tr,
        scale=scale,
        cu_seqlens=cu,
    )
    o_em = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new_tr,
        h=h_m_tr,
        g=g_tr,
        scale=scale,
        cu_seqlens=cu,
    )
    _assert_close_or_metrics(
        f"{case_name} chunk_fwd_o",
        o_tr,
        o_em,
        rtol=RTOL,
        atol=ATOL,
        r2_min=R2_MIN,
        rel_rmse_max=REL_RMSE_MAX,
        mask_if_global_r2_bad=False,
    )


def main() -> None:
    torch.manual_seed(0)
    torch.npu.set_device(NPU_DEVICE)
    dev = torch.device(NPU_DEVICE)

    print("verify_torch_emulation: cu_seqlens=None vs packed [0,T] (emulation only)...")
    verify_emulation_none_vs_packed(dev)

    for i, (case_name, seqlens) in enumerate(TRITON_VS_EMU_CASES):
        seed = 1 + i * 997
        print(f"verify_torch_emulation: Triton vs emu — {case_name} (T={sum(seqlens)})...")
        run_triton_vs_emulation_case(dev, case_name, seqlens, seed=seed)

    print("verify_torch_emulation: all checks passed.")


if __name__ == "__main__":
    main()
