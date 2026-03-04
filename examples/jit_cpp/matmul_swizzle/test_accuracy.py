"""Accuracy tests for the custom PTO matmul kernel.

Run with:
    pytest test_accuracy.py -v
"""

import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa

from jit_util_matmul import jit_compile

DEVICE = os.environ.get("NPU_DEVICE", "npu:0")
DTYPE = torch.float16

# Compile once per process.
_kernel_fn = None


def _get_kernel():
    global _kernel_fn
    if _kernel_fn is None:
        src = str(Path(__file__).resolve().parent / "matmul_custom_pto.cpp")
        _kernel_fn = jit_compile(src, verbose=True)
    return _kernel_fn


# ---------------------------------------------------------------------------
# Test shapes: (M, N, K)
# Covers tile-aligned, non-aligned, small, and large dimensions.
# ---------------------------------------------------------------------------
SHAPES = [
    # Tile-aligned shapes
    (128, 256, 512),  # minimal tile-aligned
    (256, 256, 512),  # square-ish aligned
    (256, 512, 512),  # wider N
    (384, 256, 512),  # M = 3 * M_TILE
    (512, 512, 512),  # medium square
    (128, 4096, 4096),  # single M tile, large N/K
    (256, 4096, 4096),  # two M tiles, large N/K
    (1024, 4096, 4096),  # large aligned
    # Non-aligned shapes (in-kernel valid-region masking)
    (200, 256, 512),  # M not multiple of 128
    (128, 300, 512),  # N not multiple of 256
    (128, 256, 400),  # K not multiple of 512
    (200, 300, 400),  # all non-aligned
    (1, 256, 512),  # M = 1
    (128, 1, 512),  # N = 1
    (128, 256, 64),  # K = single quarter-tile
    (128, 256, 100),  # K not multiple of 64
    (65, 129, 65),  # all just above tile boundaries
    (200, 300, 300),  # medium non-aligned
]

# ---------------------------------------------------------------------------
# Swizzle parameter combinations: (swizzle_direction, swizzle_count)
# ---------------------------------------------------------------------------
SWIZZLE_PARAMS = [
    (0, 0),  # no swizzle (baseline)
    (1, 0),  # no swizzle (direction ignored when count=0)
    (0, 1),  # Zn, count=1
    (0, 3),  # Zn, count=3
    (1, 1),  # Nz, count=1
    (1, 3),  # Nz, count=3
]

# fp16 matmul tolerance: allow up to 1e-2 mean abs error (hardware-dependent).
MEAN_ABS_TOL = 1e-2
MAX_ABS_TOL = 1.0


def _shape_id(shape):
    return f"M{shape[0]}_N{shape[1]}_K{shape[2]}"


def _swizzle_id(params):
    d, c = params
    names = {0: "Zn", 1: "Nz"}
    return f"dir{names.get(d, d)}_cnt{c}"


@pytest.fixture(scope="session", autouse=True)
def setup_device():
    torch.npu.set_device(DEVICE)


@pytest.mark.parametrize("shape", SHAPES, ids=[_shape_id(s) for s in SHAPES])
@pytest.mark.parametrize(
    "swizzle", SWIZZLE_PARAMS, ids=[_swizzle_id(p) for p in SWIZZLE_PARAMS]
)
def test_matmul_accuracy(shape, swizzle):
    m, n, k = shape
    swizzle_direction, swizzle_count = swizzle
    kernel = _get_kernel()

    torch.manual_seed(42)
    a = torch.randn(m, k, dtype=DTYPE, device=DEVICE)
    b = torch.randn(n, k, dtype=DTYPE, device=DEVICE)

    c_ref = F.linear(a, b)  # torch reference: a @ b^T
    c_custom = kernel(
        a,
        b,
        swizzle_direction=swizzle_direction,
        swizzle_count=swizzle_count,
    )

    assert (
        c_custom.shape == c_ref.shape
    ), f"Shape mismatch: got {c_custom.shape}, expected {c_ref.shape}"

    # Diagnostic: identify source of NaN if present
    if torch.isnan(c_custom).any() or torch.isnan(c_ref).any():
        nan_in_a = torch.isnan(a).any().item()
        nan_in_b = torch.isnan(b).any().item()
        nan_in_ref = torch.isnan(c_ref).any().item()
        nan_in_custom = torch.isnan(c_custom).any().item()
        nan_custom_count = torch.isnan(c_custom).sum().item() if nan_in_custom else 0
        print(
            f"\n  NaN diagnostic for shape=({m},{n},{k}), "
            f"swizzle=({swizzle_direction},{swizzle_count}):"
        )
        print(f"    NaN in a: {nan_in_a}, NaN in b: {nan_in_b}")
        print(
            f"    NaN in c_ref: {nan_in_ref}, NaN in c_custom: {nan_in_custom} "
            f"(count={nan_custom_count}/{c_custom.numel()})"
        )

    diff = torch.abs(c_custom.float() - c_ref.float())
    mean_err = float(diff.mean().cpu())
    max_err = float(diff.max().cpu())

    assert mean_err < MEAN_ABS_TOL, (
        f"Mean abs error {mean_err:.6f} exceeds tolerance {MEAN_ABS_TOL} "
        f"for shape=({m},{n},{k}), swizzle=({swizzle_direction},{swizzle_count})"
    )
    assert max_err < MAX_ABS_TOL, (
        f"Max abs error {max_err:.6f} exceeds tolerance {MAX_ABS_TOL} "
        f"for shape=({m},{n},{k}), swizzle=({swizzle_direction},{swizzle_count})"
    )
