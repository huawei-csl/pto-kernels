"""Correctness tests for the depthwise causal conv1d + bias + SiLU PTO kernel.

Run:
    pytest test_conv1d_dw.py -q --npu npu:0
"""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from jit_util_conv1d_dw import jit_compile, K

KERNEL_CPP = Path(__file__).resolve().parent / "conv1d_dw_pto.cpp"

DTYPES = [torch.float16, torch.bfloat16]
TEST_SEEDS = [0, 1]
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


def conv1d_dw_ref(x, w, bias, activation):
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


@pytest.fixture(scope="session")
def conv1d_kernel(npu_device):
    return jit_compile(str(KERNEL_CPP), verbose=False)


@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("activation", [True, False])
@pytest.mark.parametrize("batch,seq,dim", TEST_CASES)
def test_batched_matches_reference(
    conv1d_kernel, npu_device, seed, dtype, activation, batch, seq, dim
):
    torch.manual_seed(seed)
    x = 2 * torch.rand(batch, seq, dim, device=npu_device, dtype=dtype) - 1
    w = torch.rand(K, dim, device=npu_device, dtype=torch.float32) - 0.5
    bias = torch.rand(dim, device=npu_device, dtype=torch.float32) - 0.5

    y = conv1d_kernel.batched(x, w, bias, activation=activation)
    torch.npu.synchronize()

    ref = conv1d_dw_ref(x, w, bias, activation)
    err = (y.float() - ref).abs().max().item()
    assert err <= TOL[dtype], f"max abs err {err:.3e} > tol {TOL[dtype]:.1e}"


@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize("seq,dim", [(16, 256), (128, 2048), (512, 6144)])
def test_single_fp16_entry_matches_reference(conv1d_kernel, npu_device, seed, seq, dim):
    """The non-batched [L, W] fp16 entry (always applies SiLU)."""
    torch.manual_seed(seed)
    x = 2 * torch.rand(seq, dim, device=npu_device, dtype=torch.float16) - 1
    w = torch.rand(K, dim, device=npu_device, dtype=torch.float32) - 0.5
    bias = torch.rand(dim, device=npu_device, dtype=torch.float32) - 0.5

    y = conv1d_kernel(x, w, bias)
    torch.npu.synchronize()

    ref = conv1d_dw_ref(x.unsqueeze(0), w, bias, activation=True)[0]
    err = (y.float() - ref).abs().max().item()
    assert err <= TOL[torch.float16], f"max abs err {err:.3e}"
