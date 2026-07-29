# pylint: disable=wrong-import-position  # imports are guarded by importorskip
"""Correctness tests for the copy-floor reference kernel on Ascend A5.

The copy reference is the yardstick every ``fast_hadamard_256_a5`` number is
reported against, so it has to be exactly what it claims: a ``GM -> UB -> GM``
round trip that moves the data and changes nothing. If it silently dropped or
corrupted tiles it would still be *fast*, and the whole ``hadamard / copy``
ratio would be measured against a meaningless floor -- so assert bit-exactness,
not a tolerance.

Requires a real A5 device with ``torch_npu`` and ``bisheng``.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from jit_util_copy256_a5 import N, ROWS_PER_TILE, build_and_load  # noqa: E402


@pytest.fixture(scope="module")
def copy256():
    return build_and_load(verbose=False)


# every batch is a multiple of ROWS_PER_TILE: the copy kernel has no padding
# wrapper, since the benchmark only ever times it on aligned batches.
@pytest.mark.parametrize("batch", [64, 256, 1024, 4096, 65536])
def test_copy_is_bit_exact(copy256, batch):
    rng = np.random.default_rng(batch)
    x_np = rng.standard_normal((batch, N)).astype(np.float16)
    x = torch.from_numpy(x_np).npu()
    copy256(x)
    torch.npu.synchronize()
    out = x.cpu().numpy()
    assert np.array_equal(out, x_np), f"batch={batch}: copy reference altered the data"


def test_copy_touches_every_tile(copy256):
    """A copy that skipped tiles would still look fast, so check coverage.

    Zero the device buffer, copy a distinctively-valued host tensor over it, and
    require every row to survive -- this fails if the tile loop misses the tail.
    """
    batch = 3 * ROWS_PER_TILE
    x_np = np.arange(batch * N, dtype=np.float32).reshape(batch, N)
    x_np = (x_np % 2048).astype(np.float16)  # distinct per row, exact in fp16
    x = torch.from_numpy(x_np).npu()
    copy256(x)
    torch.npu.synchronize()
    out = x.cpu().numpy()
    bad = [r for r in range(batch) if not np.array_equal(out[r], x_np[r])]
    assert not bad, f"rows not preserved: {bad[:8]}{'...' if len(bad) > 8 else ''}"
