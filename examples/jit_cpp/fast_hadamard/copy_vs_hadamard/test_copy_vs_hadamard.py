import os
from pathlib import Path

import numpy as np
import pytest
import torch

from copy_vs_hadamard.jit_util_copy_pto import jit_compile as jit_compile_copy_pto
from copy_vs_hadamard.jit_util_copy_raw_cce import (
    jit_compile as jit_compile_copy_raw_cce,
)
from copy_vs_hadamard.jit_util_copy_common import validate_copy_tensors

BASIC_TEST_SHAPES = [(1, 1), (2, 32), (7, 128), (8, 2048)]
BITWISE_TEST_SHAPES = [(1, 15), (3, 17), (5, 129), (17, 4095), (31, 8191)]
BITWISE_PATTERNS = ("randn", "arange", "special_bits")
BENCHMARK_BATCHES = [1 << exponent for exponent in range(7, 13)]
BENCHMARK_HIDDEN_DIMS = [1 << exponent for exponent in range(7, 15)]


@pytest.fixture(scope="session")
def copy_pto_kernel(npu_device):
    base = Path(__file__).resolve().parent
    return jit_compile_copy_pto(
        str(base / "copy_pto.cpp"),
        verbose=True,
        device=npu_device,
    )


@pytest.fixture(scope="session")
def copy_raw_cce_kernel(npu_device, copy_pto_kernel):
    base = Path(__file__).resolve().parent
    return jit_compile_copy_raw_cce(
        str(base / "copy_raw_cce.cpp"),
        verbose=True,
        device=npu_device,
        block_dim=copy_pto_kernel.block_dim,
    )


@pytest.fixture(scope="session")
def copy_raw_cce_static_4096_kernel(npu_device, copy_pto_kernel):
    base = Path(__file__).resolve().parent
    return jit_compile_copy_raw_cce(
        str(base / "copy_raw_cce_static_4096_4096.cpp"),
        verbose=True,
        device=npu_device,
        block_dim=copy_pto_kernel.block_dim,
    )


def _run_copy(copy_kernel, x, y):
    copy_kernel(x, y, x.shape[0], x.shape[1])
    torch.npu.synchronize()


def _make_input(batch, n, device, pattern):
    if pattern == "randn":
        return torch.randn(batch, n, device=device, dtype=torch.float16)

    if pattern == "arange":
        values = torch.arange(batch * n, dtype=torch.int32)
        values = ((values % 4096) - 2048).reshape(batch, n)
        return values.to(device=device, dtype=torch.float16)

    if pattern == "special_bits":
        bit_pattern = np.array(
            [
                0x0000,
                0x8000,
                0x3C00,
                0xBC00,
                0x3555,
                0x0001,
                0x03FF,
                0x7BFF,
                0xFBFF,
                0x7C00,
                0xFC00,
                0x7E00,
            ],
            dtype=np.uint16,
        )
        tiled = np.resize(bit_pattern, batch * n).reshape(batch, n)
        return torch.from_numpy(tiled.view(np.float16).copy()).to(device)

    raise ValueError(f"Unsupported test pattern: {pattern}")


def _assert_bitwise_equal(actual, expected):
    actual_bits = actual.detach().cpu().numpy().view(np.uint16)
    expected_bits = expected.detach().cpu().numpy().view(np.uint16)
    if np.array_equal(actual_bits, expected_bits):
        return

    flat_index = int(np.flatnonzero(actual_bits != expected_bits)[0])
    row, col = divmod(flat_index, expected.shape[1])
    raise AssertionError(
        f"bitwise mismatch at ({row}, {col}): "
        f"actual=0x{int(actual_bits.reshape(-1)[flat_index]):04x}, "
        f"expected=0x{int(expected_bits.reshape(-1)[flat_index]):04x}"
    )


def _assert_kernel_copies_exact_bits(copy_kernel, x):
    y = torch.full_like(x, -123.0)
    _run_copy(copy_kernel, x, y)
    _assert_bitwise_equal(y, x)


@pytest.mark.parametrize("batch,n", BASIC_TEST_SHAPES)
def test_copy_pto_correctness(copy_pto_kernel, npu_device, batch, n):
    x = torch.randn(batch, n, device=npu_device, dtype=torch.float16)
    _assert_kernel_copies_exact_bits(copy_pto_kernel, x)


@pytest.mark.parametrize("batch,n", BASIC_TEST_SHAPES)
def test_copy_raw_cce_correctness(copy_raw_cce_kernel, npu_device, batch, n):
    x = torch.randn(batch, n, device=npu_device, dtype=torch.float16)
    _assert_kernel_copies_exact_bits(copy_raw_cce_kernel, x)


@pytest.mark.parametrize("pattern", BITWISE_PATTERNS)
@pytest.mark.parametrize("batch,n", BITWISE_TEST_SHAPES)
def test_copy_pto_bitwise_patterns(copy_pto_kernel, npu_device, batch, n, pattern):
    x = _make_input(batch, n, npu_device, pattern)
    _assert_kernel_copies_exact_bits(copy_pto_kernel, x)


@pytest.mark.parametrize("pattern", BITWISE_PATTERNS)
@pytest.mark.parametrize("batch,n", BITWISE_TEST_SHAPES)
def test_copy_raw_cce_bitwise_patterns(
    copy_raw_cce_kernel, npu_device, batch, n, pattern
):
    x = _make_input(batch, n, npu_device, pattern)
    _assert_kernel_copies_exact_bits(copy_raw_cce_kernel, x)


def test_static_copy_wrapper_rejects_wrong_shape(npu_device):
    x = torch.randn(1, 1024, device=npu_device, dtype=torch.float16)
    y = torch.empty_like(x)
    with pytest.raises(ValueError, match="batch and n must match the input tensor shape."):
        validate_copy_tensors(x, y, batch=4096, n=4096)


@pytest.mark.skipif(
    os.environ.get("PTO_RUN_LARGE_COPY_STATIC_TEST") != "1",
    reason="set PTO_RUN_LARGE_COPY_STATIC_TEST=1 to run the 4096x4096 static copy test",
)
def test_static_copy_kernel_correctness(copy_raw_cce_static_4096_kernel, npu_device):
    for pattern in BITWISE_PATTERNS:
        x = _make_input(4096, 4096, npu_device, pattern)
        _assert_kernel_copies_exact_bits(copy_raw_cce_static_4096_kernel, x)


@pytest.mark.skipif(
    os.environ.get("PTO_RUN_COPY_BENCHMARK_GRID_TEST") != "1",
    reason=(
        "set PTO_RUN_COPY_BENCHMARK_GRID_TEST=1 to run the full benchmark-grid "
        "correctness test"
    ),
)
def test_copy_pto_benchmark_grid_correctness(copy_pto_kernel, npu_device):
    for batch in BENCHMARK_BATCHES:
        for n in BENCHMARK_HIDDEN_DIMS:
            x = _make_input(batch, n, npu_device, "randn")
            _assert_kernel_copies_exact_bits(copy_pto_kernel, x)


@pytest.mark.skipif(
    os.environ.get("PTO_RUN_COPY_BENCHMARK_GRID_TEST") != "1",
    reason=(
        "set PTO_RUN_COPY_BENCHMARK_GRID_TEST=1 to run the full benchmark-grid "
        "correctness test"
    ),
)
def test_copy_raw_cce_benchmark_grid_correctness(copy_raw_cce_kernel, npu_device):
    for batch in BENCHMARK_BATCHES:
        for n in BENCHMARK_HIDDEN_DIMS:
            x = _make_input(batch, n, npu_device, "randn")
            _assert_kernel_copies_exact_bits(copy_raw_cce_kernel, x)
