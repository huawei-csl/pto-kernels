# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------

"""Correctness tests for the depthwise causal conv1d + bias + SiLU PTO kernel.

Run:
    pytest tests/test_gdn_causal_conv1d.py -q
"""

import pytest
import torch
import torch.nn.functional as F

from pto_kernels import pto_gdn_causal_conv1d

K_VALUES = [2, 3, 4, 5, 7, 12, 13, 23, 29, 47, 64]
DTYPES = [torch.float16, torch.bfloat16]
X_SHAPES = [
    # nd, B, L, C
    (3, 2, 64, 512),
    (3, 1, 97, 256),
    (3, 2, 83, 256),
    (3, 2, 71, 512),
    (2, None, 67, 256),
    (3, 2, 89, 256),
    (3, 4, 127, 256),
    (3, 2, 61, 512),
    (2, None, 53, 256),
    (3, 1, 7, 16),
    (3, 8, 128, 2048),
    (3, 1, 1009, 256),
    (2, None, 48, 80),
    (3, 1, 384, 6144),
]

# max abs error tolerance vs the fp32 reference (dtype rounding only).
TOL = {torch.float16: 6e-3, torch.bfloat16: 6e-2}


# ---------------------------------------------------------------------------
# Reference implementation (fp32 accumulate)
# ---------------------------------------------------------------------------


def _ref(x, w, bias, activation, conv_states=None):
    """Depthwise causal conv1d (width K) + per-channel bias + optional SiLU.

    x: [B, L, C]  w: [K, C]  bias: [C] or None  conv_states: [B, K-1, C] or None
    -> [B, L, C] (fp32)
    """
    B, L, C = x.shape
    K = w.shape[0]
    xf, wf = x.float(), w.float()
    pad = (
        conv_states.float()
        if conv_states is not None
        else torch.zeros(B, K - 1, C, device=x.device, dtype=torch.float32)
    )
    xe = torch.cat([pad, xf], dim=1)  # [B, L+K-1, C]
    acc = sum(xe[:, k : k + L] * wf[k] for k in range(K))
    if bias is not None:
        acc = acc + bias.float()
    return F.silu(acc) if activation else acc


@pytest.mark.parametrize("K", K_VALUES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("has_conv_states", [True, False])
@pytest.mark.parametrize("activation", [True, False])
@pytest.mark.parametrize(
    "ndim, batch, seq, channels",
    X_SHAPES,
)
def test_gdn_causal_conv1d(
    npu_device,
    K,
    dtype,
    has_bias,
    has_conv_states,
    activation,
    ndim,
    batch,
    seq,
    channels,
):
    """Parametrized correctness sweep: all K values × all option/shape combinations."""
    if ndim == 2:
        x = 2 * torch.rand(seq, channels, device=npu_device, dtype=dtype) - 1
        x_ref = x.unsqueeze(0)
        cs = (
            2 * torch.rand(K - 1, channels, device=npu_device, dtype=dtype) - 1
            if has_conv_states
            else None
        )
        cs_ref = cs.unsqueeze(0) if cs is not None else None
    else:
        x = 2 * torch.rand(batch, seq, channels, device=npu_device, dtype=dtype) - 1
        x_ref = x
        cs = (
            2 * torch.rand(batch, K - 1, channels, device=npu_device, dtype=dtype) - 1
            if has_conv_states
            else None
        )
        cs_ref = cs

    w = torch.rand(K, channels, device=npu_device, dtype=dtype) - 0.5
    bias = (
        torch.rand(channels, device=npu_device, dtype=dtype) - 0.5 if has_bias else None
    )

    y = pto_gdn_causal_conv1d(x, w, bias, conv_states=cs, activation=activation)
    torch.npu.synchronize()

    if ndim == 2:
        assert y.shape == x.shape

    ref = _ref(x_ref, w, bias, activation, conv_states=cs_ref)
    if ndim == 2:
        ref = ref[0]

    err = (y.float() - ref).abs().max().item()
    assert err <= TOL[dtype], (
        f"K={K} {dtype} bias={has_bias} states={has_conv_states} act={activation} "
        f"ndim={ndim} shape={tuple(x.shape)} max_abs_err={err:.3e} tol={TOL[dtype]:.1e}"
    )


# ---------------------------------------------------------------------------
# Invalid-input tests (expected failures)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("K", [1, 65, 100, 128])
@pytest.mark.parametrize("dtype", DTYPES)
def test_invalid_k(npu_device, dtype, K):
    """K outside [2..64] must raise RuntimeError."""
    dim = 256
    x = torch.rand(1, 16, dim, device=npu_device, dtype=dtype)
    w = torch.rand(K, dim, device=npu_device, dtype=dtype)
    with pytest.raises(RuntimeError, match="filter width K must be in"):
        pto_gdn_causal_conv1d(x, w)


@pytest.mark.parametrize("channels", [1, 8, 9, 15, 17, 100, 130])
@pytest.mark.parametrize("dtype", DTYPES)
def test_invalid_channel_alignment(npu_device, dtype, channels):
    """channels not divisible by 16 must raise RuntimeError."""
    K = 4
    x = torch.rand(1, 16, channels, device=npu_device, dtype=dtype)
    w = torch.rand(K, channels, device=npu_device, dtype=dtype)
    with pytest.raises(RuntimeError, match="channels must be a multiple of 16"):
        pto_gdn_causal_conv1d(x, w)
