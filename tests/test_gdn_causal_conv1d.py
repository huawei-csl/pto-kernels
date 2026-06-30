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

# (batch, seq, dim): small general shapes + the GDN prefill regime
# (dim 2048 = H*D, 6144 = q+k+v; seq 128..512).
TEST_CASES = [
    (1, 16, 256),
    (2, 31, 256),
    (1, 128, 2048),
    (8, 128, 2048),
    (1, 256, 2048),
    (8, 384, 2048),
    (1, 512, 2048),
    (8, 512, 2048),
    (1, 128, 6144),
    (8, 256, 6144),
    (1, 384, 6144),
    (8, 512, 6144),
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

    If conv_states is given, it is prepended as history; otherwise zero-padding.
    """
    B, L, C = x.shape
    K = w.shape[0]
    xf = x.float()
    wf = w.float()

    if conv_states is not None:
        pad = conv_states.float()  # [B, K-1, C]
    else:
        pad = torch.zeros((B, K - 1, C), device=x.device, dtype=torch.float32)
    xe = torch.cat([pad, xf], dim=1)  # [B, L+K-1, C]

    acc = sum(xe[:, k : k + L] * wf[k] for k in range(K))
    if bias is not None:
        acc = acc + bias.float()
    return F.silu(acc) if activation else acc


# ---------------------------------------------------------------------------
# Ring-size variants — one K per RS
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("K", K_VALUES)
def test_ring_size_variants(npu_device, dtype, K):
    """Each K value exercises a distinct ring-size (RS = roundUpToPow2(K)) variant."""
    batch, seq, dim = 2, 64, 512
    x = 2 * torch.rand(batch, seq, dim, device=npu_device, dtype=dtype) - 1
    w = torch.rand(K, dim, device=npu_device, dtype=dtype) - 0.5
    bias = torch.rand(dim, device=npu_device, dtype=dtype) - 0.5

    y = pto_gdn_causal_conv1d(x, w, bias, activation=True)
    torch.npu.synchronize()

    ref = _ref(x, w, bias, activation=True)
    err = (y.float() - ref).abs().max().item()
    assert (
        err <= TOL[dtype]
    ), f"K={K} dtype={dtype} max abs err {err:.3e} > tol {TOL[dtype]:.1e}"


# ---------------------------------------------------------------------------
# Standard batched convolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("activation", [True, False])
@pytest.mark.parametrize("batch,seq,dim", TEST_CASES)
def test_batched_matches_reference(npu_device, dtype, activation, batch, seq, dim):
    K = 4
    x = 2 * torch.rand(batch, seq, dim, device=npu_device, dtype=dtype) - 1
    w = torch.rand(K, dim, device=npu_device, dtype=dtype) - 0.5
    bias = torch.rand(dim, device=npu_device, dtype=dtype) - 0.5

    y = pto_gdn_causal_conv1d(x, w, bias, activation=activation)
    torch.npu.synchronize()

    ref = _ref(x, w, bias, activation)
    err = (y.float() - ref).abs().max().item()
    assert err <= TOL[dtype], f"max abs err {err:.3e} > tol {TOL[dtype]:.1e}"


# ---------------------------------------------------------------------------
# Bias-less case
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_no_bias(npu_device, dtype):
    """Empty bias tensor → hasBias=0 path; reference omits bias term."""
    K, batch, seq, dim = 4, 2, 64, 256
    x = 2 * torch.rand(batch, seq, dim, device=npu_device, dtype=dtype) - 1
    w = torch.rand(K, dim, device=npu_device, dtype=dtype) - 0.5
    empty_bias = torch.empty(0, device=npu_device, dtype=dtype)

    y = pto_gdn_causal_conv1d(x, w, empty_bias, activation=True)
    torch.npu.synchronize()

    ref = _ref(x, w, bias=None, activation=True)
    err = (y.float() - ref).abs().max().item()
    assert err <= TOL[dtype], f"no-bias max abs err {err:.3e}"


# ---------------------------------------------------------------------------
# conv_states — history from prior tokens
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("K", [2, 4, 8])
def test_conv_states_matches_reference(npu_device, dtype, K):
    """History from conv_states is used instead of zero-padding."""
    batch, seq, dim = 2, 64, 256
    x = 2 * torch.rand(batch, seq, dim, device=npu_device, dtype=dtype) - 1
    w = torch.rand(K, dim, device=npu_device, dtype=dtype) - 0.5
    bias = torch.rand(dim, device=npu_device, dtype=dtype) - 0.5
    # conv_states: [B, K-1, C] — the K-1 history rows
    conv_states = 2 * torch.rand(batch, K - 1, dim, device=npu_device, dtype=dtype) - 1

    y = pto_gdn_causal_conv1d(
        x, w, bias, conv_states=conv_states, activation=True
    )
    torch.npu.synchronize()

    ref = _ref(x, w, bias, activation=True, conv_states=conv_states)
    err = (y.float() - ref).abs().max().item()
    assert (
        err <= TOL[dtype]
    ), f"K={K} dtype={dtype} conv_states max abs err {err:.3e} > tol {TOL[dtype]:.1e}"


# ---------------------------------------------------------------------------
# No conv_states → zero-padding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_no_conv_states_uses_zero_padding(npu_device, dtype):
    """Omitting conv_states (or passing empty tensor) must use zero-padding."""
    K, batch, seq, dim = 4, 2, 64, 256
    x = 2 * torch.rand(batch, seq, dim, device=npu_device, dtype=dtype) - 1
    w = torch.rand(K, dim, device=npu_device, dtype=dtype) - 0.5
    bias = torch.rand(dim, device=npu_device, dtype=dtype) - 0.5

    y = pto_gdn_causal_conv1d(x, w, bias, activation=True)
    torch.npu.synchronize()

    ref = _ref(x, w, bias, activation=True, conv_states=None)
    err = (y.float() - ref).abs().max().item()
    assert err <= TOL[dtype], f"no-conv_states max abs err {err:.3e}"


# ---------------------------------------------------------------------------
# Multi-chunk with conv_states (large K, small seq forces second chunk jstart<0)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_large_K_multiChunk(npu_device, dtype):
    """K=32, seq=48 forces seqChunks>1; second chunk's jstart is negative and
    must read history rows from conv_states rather than from x."""
    K, batch, seq, dim = 32, 1, 48, 512
    x = 2 * torch.rand(batch, seq, dim, device=npu_device, dtype=dtype) - 1
    w = torch.rand(K, dim, device=npu_device, dtype=dtype) - 0.5
    bias = torch.rand(dim, device=npu_device, dtype=dtype) - 0.5
    conv_states = 2 * torch.rand(batch, K - 1, dim, device=npu_device, dtype=dtype) - 1

    y = pto_gdn_causal_conv1d(
        x, w, bias, conv_states=conv_states, activation=False
    )
    torch.npu.synchronize()

    ref = _ref(x, w, bias, activation=False, conv_states=conv_states)
    err = (y.float() - ref).abs().max().item()
    assert err <= TOL[dtype], f"large-K multi-chunk max abs err {err:.3e}"


# ---------------------------------------------------------------------------
# 2D input reshape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seq,dim", [(16, 256), (128, 2048), (512, 6144)])
def test_2d_reshape_matches_reference(npu_device, dtype, seq, dim):
    """2D input [L, C] is treated as batch=1 and output has the same rank."""
    K = 4
    x = 2 * torch.rand(seq, dim, device=npu_device, dtype=dtype) - 1
    w = torch.rand(K, dim, device=npu_device, dtype=dtype) - 0.5
    bias = torch.rand(dim, device=npu_device, dtype=dtype) - 0.5

    y = pto_gdn_causal_conv1d(x, w, bias)
    torch.npu.synchronize()

    assert y.shape == x.shape
    ref = _ref(x.unsqueeze(0), w, bias, activation=True)[0]
    err = (y.float() - ref).abs().max().item()
    assert err <= TOL[dtype], f"2D reshape max abs err {err:.3e}"
