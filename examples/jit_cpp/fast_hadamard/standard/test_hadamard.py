import math

import pytest
import torch

from standard.jit_util_hadamard import (
    MAX_HADAMARD_N,
    SPECIALISED_HADAMARD_N,
    validate_hadamard_args,
)

TEST_BATCHES = [1, 7, 22, 65]
TEST_HIDDEN_DIMS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
TEST_SEEDS = [0, 1]
DTYPE = torch.float16


def hadamard_ref_inplace(x):
    """Reference FHT with the same stage layout as the PTO kernel."""
    x = x.clone()
    n = x.shape[-1]
    n_half = n // 2
    log2_n = int(math.log2(n))

    for _ in range(log2_n):
        even = x[..., 0::2].clone()
        odd = x[..., 1::2].clone()
        x[..., :n_half] = even + odd
        x[..., n_half:] = even - odd
    return x


def run_hadamard_inplace(hadamard_kernel, x, log2_n=None):
    n = x.shape[-1]
    if log2_n is None:
        log2_n = int(math.log2(n))
    hadamard_kernel(x, x.shape[0], n, log2_n)
    torch.npu.synchronize()


@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize("batch", TEST_BATCHES)
@pytest.mark.parametrize("n", TEST_HIDDEN_DIMS)
def test_fast_hadamard_correctness(hadamard_kernel, npu_device, seed, batch, n):
    torch.manual_seed(seed)
    log2_n = int(math.log2(n))
    x = torch.randn(batch, n, device=npu_device, dtype=DTYPE)

    y_ref = hadamard_ref_inplace(x)

    run_hadamard_inplace(hadamard_kernel, x, log2_n)

    assert torch.equal(
        x, y_ref
    ), f"Mismatch for seed={seed}, batch={batch}, N={n}; max_diff={(x - y_ref).abs().max().item():.6f}"


def test_batch_partition_boundaries(hadamard_kernel, npu_device):
    n = 1024
    logical_workers = hadamard_kernel.block_dim * 2
    batches = sorted({0, 1, logical_workers - 1, logical_workers, logical_workers + 1})

    for batch in batches:
        x = torch.randn(batch, n, device=npu_device, dtype=DTYPE)
        y_ref = hadamard_ref_inplace(x)
        run_hadamard_inplace(hadamard_kernel, x)
        assert torch.equal(x, y_ref), f"Mismatch for batch boundary case batch={batch}"


@pytest.mark.parametrize("n", [128, 1024, 16384])
def test_samples_per_load_boundaries(hadamard_kernel, npu_device, n):
    samples_per_load = MAX_HADAMARD_N // n
    batches = sorted(
        {1, max(0, samples_per_load - 1), samples_per_load, samples_per_load + 1}
    )

    for batch in batches:
        x = torch.randn(batch, n, device=npu_device, dtype=DTYPE)
        y_ref = hadamard_ref_inplace(x)
        run_hadamard_inplace(hadamard_kernel, x)
        assert torch.equal(
            x, y_ref
        ), f"Mismatch for samples_per_load boundary batch={batch}, N={n}"


@pytest.mark.parametrize("n", [192, MAX_HADAMARD_N + 1])
def test_invalid_n_rejected(npu_device, n):
    x = torch.randn(2, n, device=npu_device, dtype=DTYPE)

    with pytest.raises(ValueError, match="power of two|<= 16384"):
        validate_hadamard_args(x, 2, n, int(math.log2(n)))


def test_wrong_log2_rejected(npu_device):
    x = torch.randn(2, 1024, device=npu_device, dtype=DTYPE)

    with pytest.raises(ValueError, match="log2_n"):
        validate_hadamard_args(x, 2, 1024, 9)


def test_noncontiguous_input_rejected(npu_device):
    x = torch.randn(1024, 5, device=npu_device, dtype=DTYPE).transpose(0, 1)
    assert x.shape == (5, 1024)
    assert not x.is_contiguous()

    with pytest.raises(ValueError, match="contiguous"):
        validate_hadamard_args(x, 5, 1024, 10)


def test_wrong_dtype_rejected(npu_device):
    x = torch.randn(2, 1024, device=npu_device, dtype=torch.float32)

    with pytest.raises(TypeError, match="torch.float16"):
        validate_hadamard_args(x, 2, 1024, 10)


def test_shape_mismatch_rejected(npu_device):
    x = torch.randn(2, 1024, device=npu_device, dtype=DTYPE)

    with pytest.raises(ValueError, match="must match batch=1 and n=1024"):
        validate_hadamard_args(x, 1, 1024, 10)


@pytest.mark.parametrize("n", [192, MAX_HADAMARD_N + 1])
def test_kernel_wrapper_rejects_invalid_n(hadamard_kernel, npu_device, n):
    """The wrapper must apply the restrictions, not just re-export them.

    Without this the validator could sit in jit_util_hadamard.py and never be
    called, and every bad argument would still reach the kernel and fail
    silently -- which is the state this moved out of.
    """
    x = torch.randn(2, n, device=npu_device, dtype=DTYPE)
    with pytest.raises(ValueError, match="power of two|<= 16384"):
        hadamard_kernel(x)


def test_kernel_wrapper_rejects_wrong_log2(hadamard_kernel, npu_device):
    x = torch.randn(2, 1024, device=npu_device, dtype=DTYPE)
    with pytest.raises(ValueError, match="log2_n"):
        hadamard_kernel(x, 2, 1024, 9)


def test_kernel_wrapper_derives_batch_n_and_log2(hadamard_kernel, npu_device):
    """x alone is enough; the three redundant arguments are optional."""
    torch.manual_seed(0)
    x = torch.randn(7, 1024, device=npu_device, dtype=DTYPE)
    expected = hadamard_ref_inplace(x)
    hadamard_kernel(x)
    torch.npu.synchronize()
    assert torch.equal(x, expected)


def test_zero_batch_is_a_no_op(hadamard_kernel, npu_device):
    x = torch.randn(0, 1024, device=npu_device, dtype=DTYPE)
    hadamard_kernel(x)
    torch.npu.synchronize()
    assert x.shape == (0, 1024)


@pytest.mark.parametrize("n", SPECIALISED_HADAMARD_N)
def test_specialised_sizes_are_all_powers_of_two_within_the_limit(n):
    assert n & (n - 1) == 0
    assert n <= MAX_HADAMARD_N
