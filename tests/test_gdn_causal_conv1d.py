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

# Filter width K compiled into the wheel (matches CAUSAL_CONV_K default).
K = 4

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


def gdn_causal_conv1d_ref(x, w, bias, activation):
    """Depthwise causal conv1d (width K) + per-channel bias + optional SiLU.

    fp32 accumulate; x[:, <0] padded with zeros (no conv_states).
    x: [B, L, W]   w: [K, W]   bias: [W]   -> [B, L, W] (fp32)
    """
    B, L, W = x.shape
    pad = torch.zeros((B, K - 1, W), device=x.device, dtype=x.dtype)
    xe = torch.cat([pad, x], dim=1).float()
    wf = w.float()
    acc = sum(xe[:, k : k + L] * wf[k] for k in range(K)) + bias.float()
    return F.silu(acc) if activation else acc


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("activation", [True, False])
@pytest.mark.parametrize("batch,seq,dim", TEST_CASES)
def test_batched_matches_reference(npu_device, dtype, activation, batch, seq, dim):
    x = 2 * torch.rand(batch, seq, dim, device=npu_device, dtype=dtype) - 1
    w = torch.rand(K, dim, device=npu_device, dtype=torch.float32) - 0.5
    bias = torch.rand(dim, device=npu_device, dtype=torch.float32) - 0.5

    y = pto_gdn_causal_conv1d(x, w, bias, activation=activation)
    torch.npu.synchronize()

    ref = gdn_causal_conv1d_ref(x, w, bias, activation)
    err = (y.float() - ref).abs().max().item()
    assert err <= TOL[dtype], f"max abs err {err:.3e} > tol {TOL[dtype]:.1e}"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seq,dim", [(16, 256), (128, 2048), (512, 6144)])
def test_2d_reshape_matches_reference(npu_device, seq, dim):
    """2D input [L, W] is treated as batch=1 and output has the same rank."""
    x = 2 * torch.rand(seq, dim, device=npu_device, dtype=torch.float16) - 1
    w = torch.rand(K, dim, device=npu_device, dtype=torch.float32) - 0.5
    bias = torch.rand(dim, device=npu_device, dtype=torch.float32) - 0.5

    y = pto_gdn_causal_conv1d(x, w, bias)
    torch.npu.synchronize()

    assert y.shape == x.shape
    ref = gdn_causal_conv1d_ref(x.unsqueeze(0), w, bias, activation=True)[0]
    err = (y.float() - ref).abs().max().item()
    assert err <= TOL[torch.float16], f"max abs err {err:.3e}"
