# pylint: disable=wrong-import-position  # imports are guarded by importorskip
"""Correctness tests for the N=256 fast Walsh-Hadamard kernel on Ascend A5.

Checks the device output against a torch reference (``x @ H``, H the +/-1
Hadamard matrix) across a range of batch sizes, deliberately including
non-power-of-2 and non-tile-multiple sizes to exercise the padding wrapper.
Requires a real A5 device with ``torch_npu`` and ``bisheng``.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from jit_util_hadamard256_a5 import N, build_and_load  # noqa: E402

TOLERANCE = 0.03  # fp16 accumulation over log2(256)=8 butterfly stages


def hadamard_matrix(n: int) -> np.ndarray:
    """Natural-order Sylvester construction of the +/-1 Hadamard matrix."""
    matrix = np.array([[1.0]], dtype=np.float64)
    while matrix.shape[0] < n:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix


@pytest.fixture(scope="module")
def hadamard256():
    return build_and_load(verbose=False)


@pytest.fixture(scope="module")
def reference_matrix():
    return hadamard_matrix(N)


# 64/256/1024 are tile-aligned powers of two; 1000/4097 are non-multiples of the
# 64-row tile (exercise padding); 1536/3200 are non-power-of-2 multiples.
@pytest.mark.parametrize("batch", [64, 256, 1000, 1024, 1536, 3200, 4097, 65536])
def test_matches_torch_reference(hadamard256, reference_matrix, batch):
    rng = np.random.default_rng(batch)
    x_np = rng.standard_normal((batch, N)).astype(np.float16)
    reference = x_np.astype(np.float32) @ reference_matrix

    x = torch.from_numpy(x_np).npu()
    hadamard256(x)
    torch.npu.synchronize()
    out = x.cpu().numpy().astype(np.float32)

    denom = float(np.abs(reference).max()) or 1.0
    rel_error = float(np.abs(out - reference).max()) / denom
    assert rel_error < TOLERANCE, f"batch={batch}: rel_error={rel_error:.4g}"
