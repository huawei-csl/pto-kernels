"""
Compare ``torch_emulation`` against Triton ``fla_vendor`` kernels (same dtypes / layouts).

For ``chunk_gated_delta_rule_fwd_h`` and ``chunk_fwd_o``, Triton bf16 matmul ordering can
differ slightly from PyTorch; we accept either ``torch.allclose`` (tight) or high :math:`R^2`
and low relative RMSE (vs Triton as reference).

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

# When ``allclose`` is too strict (bf16 / fused matmul), require strong agreement on these metrics
# (Triton output = reference for R² and relative RMSE).
R2_MIN = 0.9995
REL_RMSE_MAX = 0.05
# ``chunk_gated_delta_rule_fwd_h`` ``h`` can disagree on elements where Triton rounds to ~0 but
# emulation is still small-but-nonzero; global R² is then meaningless. Compare on |ref| > eps.
MASK_REF_ABS = 1e-5


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


def main() -> None:
    torch.manual_seed(1)
    torch.npu.set_device(NPU_DEVICE)
    dev = torch.device(NPU_DEVICE)

    n_seq, l_seg = 2, 128
    h, dk, dv = 4, 32, 32
    t = n_seq * l_seg
    cu_seqlens = torch.arange(0, t + 1, l_seg, dtype=torch.long, device=dev)
    chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, CHUNK_SIZE)

    q = torch.randn(1, t, h, dk, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, t, h, dk, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, t, h, dv, device=dev, dtype=torch.bfloat16)
    g_in = F.logsigmoid(torch.randn(1, t, h, device=dev, dtype=torch.float32))
    beta = torch.rand(1, t, h, device=dev, dtype=torch.bfloat16)
    initial_state = torch.zeros(n_seq, h, dk, dv, device=dev, dtype=torch.bfloat16)
    scale = dk**-0.5

    g_tr = chunk_local_cumsum_tr(g_in, chunk_size=CHUNK_SIZE, cu_seqlens=cu_seqlens)
    g_em = chunk_local_cumsum(g_in, chunk_size=CHUNK_SIZE, cu_seqlens=cu_seqlens)
    assert torch.allclose(g_tr.float(), g_em.float(), rtol=RTOL, atol=ATOL), "chunk_local_cumsum"

    a_tr = chunk_scaled_dot_kkt_fwd_tr(
        k=k,
        beta=beta,
        g_cumsum=g_tr,
        cu_seqlens=cu_seqlens,
        output_dtype=torch.float32,
    )
    a_em = chunk_scaled_dot_kkt_fwd(
        k=k,
        beta=beta,
        g_cumsum=g_tr,
        cu_seqlens=cu_seqlens,
        output_dtype=torch.float32,
    )
    assert torch.allclose(a_tr.float(), a_em.float(), rtol=RTOL, atol=ATOL), "chunk_scaled_dot_kkt_fwd"

    w_tr, u_tr = recompute_w_u_fwd_tr(
        k=k,
        v=v,
        beta=beta,
        A=a_tr,
        g_cumsum=g_tr,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    w_em, u_em = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=a_tr,
        g_cumsum=g_tr,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    assert torch.allclose(w_tr.float(), w_em.float(), rtol=RTOL, atol=ATOL), "recompute_w_u w"
    assert torch.allclose(u_tr.float(), u_em.float(), rtol=RTOL, atol=ATOL), "recompute_w_u u"

    a_s_tr = solve_tril_tr(A=a_tr, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
    a_s_em = solve_tril(A=a_tr, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
    assert torch.allclose(a_s_tr.float(), a_s_em.float(), rtol=RTOL, atol=ATOL), "solve_tril"

    w2_tr, u2_tr = recompute_w_u_fwd_tr(
        k=k,
        v=v,
        beta=beta,
        A=a_s_tr,
        g_cumsum=g_tr,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    w2_em, u2_em = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=a_s_em,
        g_cumsum=g_tr,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    assert torch.allclose(w2_tr.float(), w2_em.float(), rtol=RTOL, atol=ATOL), "recompute_w_u (solved) w"
    assert torch.allclose(u2_tr.float(), u2_em.float(), rtol=RTOL, atol=ATOL), "recompute_w_u (solved) u"

    # Same w,u for Triton vs emulation so differences are only from chunk_h / chunk_o math.
    h_m_tr, v_new_tr, _ = chunk_gated_delta_rule_fwd_h_tr(
        k=k,
        w=w2_tr,
        u=u2_tr,
        g=g_tr,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
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
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )
    _assert_close_or_metrics(
        "chunk_gated_delta_rule_fwd_h h",
        h_m_tr,
        h_m_em,
        rtol=RTOL,
        atol=ATOL,
        r2_min=R2_MIN,
        rel_rmse_max=REL_RMSE_MAX,
        mask_if_global_r2_bad=True,
    )
    _assert_close_or_metrics(
        "chunk_gated_delta_rule_fwd_h v_new",
        v_new_tr,
        v_new_em,
        rtol=RTOL,
        atol=ATOL,
        r2_min=R2_MIN,
        rel_rmse_max=REL_RMSE_MAX,
        mask_if_global_r2_bad=False,
    )

    # Same v_new and h from Triton reference so chunk_o comparison is isolated.
    o_tr = chunk_fwd_o_tr(
        q=q,
        k=k,
        v=v_new_tr,
        h=h_m_tr,
        g=g_tr,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    o_em = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new_tr,
        h=h_m_tr,
        g=g_tr,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    _assert_close_or_metrics(
        "chunk_fwd_o",
        o_tr,
        o_em,
        rtol=RTOL,
        atol=ATOL,
        r2_min=R2_MIN,
        rel_rmse_max=REL_RMSE_MAX,
        mask_if_global_r2_bad=False,
    )

    print("verify_torch_emulation: all checks passed.")


if __name__ == "__main__":
    main()
