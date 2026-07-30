# pylint: disable=wrong-import-position  # imports are guarded by importorskip
"""Correctness for the fast Walsh-Hadamard kernel on Ascend A5, over N and batch.

Compares device output against a torch reference (``x @ H``, H the +/-1 Hadamard
matrix). Requires a real A5 device with ``torch_npu`` and ``bisheng``.

Up to N=256 a stage's two half-rows each fit one 128-element fp16 vector; beyond
that the row splits into independent ``(N/2)/128`` chunks, and the unroll slots span
(row, chunk) pairs so every load in an iteration precedes every store.

That ordering is the whole ballgame, which is why N>256 is tested rather than
trusted. A stage's sums compact into the lower half of the row -- exactly where a
lower-numbered chunk still has to read from -- so a per-chunk load/store loop
aliases in place. It also *passes* at N=512 if the stores happen not to have landed
yet, then silently corrupts at larger N. These cases pin the ordering.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from jit_util_a5 import N, build_and_load, rows_for  # noqa: E402

TOLERANCE = 0.03  # fp16 accumulation over log2(N) butterfly stages


def hadamard_matrix(n: int) -> np.ndarray:
    """Natural-order Sylvester construction of the +/-1 Hadamard matrix."""
    matrix = np.array([[1.0]], dtype=np.float64)
    while matrix.shape[0] < n:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix


def assert_matches_reference(kernel, n, batch, seed, label):
    rng = np.random.default_rng(seed)
    x_np = rng.standard_normal((batch, n)).astype(np.float16)
    reference = x_np.astype(np.float32) @ hadamard_matrix(n)

    x = torch.from_numpy(x_np).npu()
    kernel(x)
    torch.npu.synchronize()
    out = x.cpu().numpy().astype(np.float32)

    denom = float(np.abs(reference).max()) or 1.0
    rel_error = float(np.abs(out - reference).max()) / denom
    assert rel_error < TOLERANCE, f"{label}: rel_error={rel_error:.4g}"


@pytest.fixture(scope="module")
def hadamard256():
    return build_and_load(verbose=False)


# 64/256/1024 are tile-aligned powers of two; 1000/4097 are non-multiples of the
# 64-row tile (exercise padding); 1536/3200 are non-power-of-2 multiples.
@pytest.mark.parametrize("batch", [64, 256, 1000, 1024, 1536, 3200, 4097, 65536])
def test_matches_torch_reference(hadamard256, batch):
    assert_matches_reference(hadamard256, N, batch, batch, f"batch={batch}")


# 64/128 use a partly-filled vector (chunks=1); 256 fills it exactly; 512/1024/2048
# are chunks=2/4/8 and are the cases the slot ordering exists for.
@pytest.mark.parametrize("n", [64, 128, 256, 512, 1024, 2048])
def test_matches_torch_reference_at_block_size(n):
    kernel = build_and_load(n=n, verbose=False)
    # a few tiles, so the chunk loop actually iterates
    assert_matches_reference(kernel, n, 4 * rows_for(n), n, f"n={n}")


def test_block_size_must_be_power_of_two():
    with pytest.raises(ValueError, match="power of two"):
        build_and_load(n=192, verbose=False)


# For N<256 the kernel packs 256/N rows into one vector-wide window, so a window can
# hold both real rows and batch padding. "Rows never mix" is what makes the packing
# correct, and this is the case where it would bite: assert it against hostile
# padding rather than trusting it.
@pytest.mark.parametrize("n", [32, 64, 128])
def test_padding_cannot_contaminate_packed_rows(n):
    kernel = build_and_load(n=n, verbose=False)
    rows = rows_for(n)
    real = rows + 1  # deliberately not a multiple of the pack factor
    padded = (real + rows - 1) // rows * rows
    rng = np.random.default_rng(1000 + n)
    base = rng.standard_normal((padded, n)).astype(np.float16)

    benign = base.copy()
    benign[real:] = 0.0
    hostile = base.copy()
    hostile[real:] = np.inf
    hostile[real::2] = np.nan

    outs = []
    for data in (benign, hostile):
        x = torch.from_numpy(data.copy()).npu()
        kernel(x)
        torch.npu.synchronize()
        outs.append(x.cpu().numpy()[:real])
    # bit-identical, not approximately equal: inf/nan bleeding across rows in a
    # shared window would show up as a bit difference long before a tolerance one
    assert np.array_equal(
        outs[0].view(np.uint16), outs[1].view(np.uint16)
    ), f"n={n}: inf/nan padding changed the real rows -- packed rows are mixing"
