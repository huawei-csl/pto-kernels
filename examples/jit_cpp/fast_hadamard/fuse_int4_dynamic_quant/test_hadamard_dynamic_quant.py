"""Test fused Hadamard + dynamic symmetric int4 quantization kernel."""

import math
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa

from fuse_int4_dynamic_quant.jit_util_hadamard_dynamic_quant import (
    jit_compile,
)

KERNEL_CPP = str(Path(__file__).resolve().parent / "fast_hadamard_dynamic_quant.cpp")
DEVICE = "npu:0"


def _hadamard_ref(x):
    """Python butterfly FHT reference (unnormalized)."""
    n = x.shape[-1]
    log2_n = int(math.log2(n))
    x = x.clone()
    n_half = n // 2
    for _ in range(log2_n):
        even = x[..., 0::2].clone()
        odd = x[..., 1::2].clone()
        x[..., :n_half] = even + odd
        x[..., n_half:] = even - odd
    return x


def _dynamic_quant_ref(x, hadamard_n):
    """Python reference: blockwise FHT + per-row symmetric int4 quant."""
    batch, full_n = x.shape
    num_blocks = full_n // hadamard_n
    inv_sqrt = 1.0 / math.sqrt(hadamard_n)

    # Blockwise Hadamard
    reshaped = x.reshape(batch, num_blocks, hadamard_n)
    transformed = _hadamard_ref(reshaped).reshape(batch, full_n)

    # Per-row max_abs and scale
    max_abs = transformed.abs().amax(dim=-1)
    scale = (max_abs * inv_sqrt / 7.0).clamp(min=1e-6)

    # Quantize
    inv_scale = 7.0 / max_abs.clamp(min=1e-6)
    q = torch.round(transformed * inv_scale.unsqueeze(-1)).clamp(-8, 7).to(torch.int8)

    # Pack pairs into bytes
    low = q[:, 0::2] & 0xF
    high = (q[:, 1::2] & 0xF) << 4
    packed = (low | high).to(torch.int8)

    return packed, scale


def _unpack_int4(packed, n):
    """Unpack int8 packed bytes into signed int4 values."""
    packed_u8 = packed.to(torch.int16)
    packed_u8 = torch.where(packed_u8 < 0, packed_u8 + 256, packed_u8)
    low = (packed_u8 & 0xF).to(torch.int8)
    high = ((packed_u8 >> 4) & 0xF).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    out = torch.empty(packed.shape[0], n, dtype=torch.int8)
    out[:, 0::2] = low
    out[:, 1::2] = high
    return out


@pytest.fixture(scope="module")
def kernel():
    return jit_compile(KERNEL_CPP, verbose=False, device=DEVICE)


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("full_n,hadamard_n", [(4096, 4096), (4096, 128), (8192, 128)])
def test_dynamic_quant_matches_reference(kernel, batch, full_n, hadamard_n):
    torch.manual_seed(42)
    x_npu = torch.randn(batch, full_n, device=DEVICE, dtype=torch.float16)
    y_npu = torch.empty(batch, full_n // 2, device=DEVICE, dtype=torch.int8)
    s_npu = torch.empty(batch, dtype=torch.float32, device=DEVICE)

    x_scratch = x_npu.clone()
    kernel(x_scratch, y_npu, s_npu, batch, full_n, hadamard_n)
    torch.npu.synchronize()

    ref_packed, ref_scale = _dynamic_quant_ref(x_npu.cpu().float(), hadamard_n)

    # Compare scales
    scale_diff = (s_npu.cpu() - ref_scale).abs().max().item()
    assert scale_diff < 0.01, f"scale mismatch: max diff {scale_diff}"

    # Compare quantized values
    got = _unpack_int4(y_npu.cpu(), full_n).float()
    ref = _unpack_int4(ref_packed, full_n).float()
    cosine = torch.nn.functional.cosine_similarity(
        got.reshape(1, -1), ref.reshape(1, -1)
    ).item()
    assert cosine > 0.99, f"quantized cosine {cosine:.6f}"
