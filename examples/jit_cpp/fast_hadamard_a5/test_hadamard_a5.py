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

import ctypes

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from jit_a5 import entry, stream_ptr  # noqa: E402
from jit_util_a5 import (  # noqa: E402
    DISPATCH_ARGS,
    N,
    build_and_load,
    compile_kernel,
    kernel_rows_for,
    load_lib,
    rows_for,
)

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
def hadamard_default():
    return build_and_load(verbose=False)


# 64/256/1024 are tile-aligned powers of two; 1000/4097 are non-multiples of the
# 64-row tile (exercise padding); 1536/3200 are non-power-of-2 multiples.
@pytest.mark.parametrize("batch", [64, 256, 1000, 1024, 1536, 3200, 4097, 65536])
def test_matches_torch_reference(hadamard_default, batch):
    assert_matches_reference(hadamard_default, N, batch, batch, f"batch={batch}")


# 32/64/128 pack multiple rows per window; 256 fills a vector exactly; 512/1024/2048
# are chunks=2/4/8 and are the cases the slot ordering exists for.
@pytest.mark.parametrize("n", [32, 64, 128, 256, 512, 1024, 2048])
def test_matches_torch_reference_at_block_size(n):
    kernel = build_and_load(n=n, verbose=False)
    # a few tiles, so the chunk loop actually iterates
    assert_matches_reference(kernel, n, 4 * rows_for(n), n, f"n={n}")


@pytest.mark.parametrize("bad_n", [16, 192, 4096, 0])
def test_unsupported_block_size_is_rejected(bad_n):
    # Not cosmetic: the dispatching launcher's default case is a silent no-op, so
    # an unvalidated n returns the input unchanged rather than failing. Both
    # compile_kernel and load_lib must reject it.
    with pytest.raises(ValueError):
        build_and_load(n=bad_n, verbose=False)
    with pytest.raises(ValueError):
        load_lib(compile_kernel(verbose=False), n=bad_n)


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


# The kernel reads its argument as fp16 and as one flat run, and can report
# neither: a wider dtype is reinterpreted and a strided view is read as if flat,
# both silently. So the wrapper has to refuse them.
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_wrong_dtype_is_rejected(hadamard_default, dtype):
    x = torch.zeros((rows_for(N), N), dtype=dtype).npu()
    with pytest.raises(AssertionError, match="fp16"):
        hadamard_default(x)


# A batch that is an exact multiple of ROWS_PER_TILE is the dangerous case: the
# padding path is skipped, so the strided view reaches the kernel unmodified.
# Non-multiples would be copied into a contiguous buffer and quietly work, which
# is what would make this bug batch-dependent rather than deterministic.
def test_non_contiguous_is_rejected(hadamard_default):
    wide = torch.zeros((rows_for(N), 2 * N), dtype=torch.float16).npu()
    view = wide[:, :N]
    assert not view.is_contiguous(), "test needs a genuinely strided view"
    with pytest.raises(AssertionError, match="contiguous"):
        hadamard_default(view)
    # and the same data, made contiguous, is accepted
    hadamard_default(view.contiguous())


# rows_for() is stated in Python (the padding wrapper needs it before any .so
# exists) and again as RowsFor<N> in the kernel. Pin them together.
def test_rows_for_matches_kernel():
    query = kernel_rows_for(compile_kernel(verbose=False))
    mismatched = {
        n: (rows_for(n), query(n))
        for n in (32, 64, 128, 256, 512, 1024, 2048)
        if rows_for(n) != query(n)
    }
    assert not mismatched, f"host/kernel tiling disagree: {mismatched}"


# The 4-argument launcher hardcodes DEFAULT_N. Pin that it really is N, rather than
# trusting two constants in two languages to stay equal.
def test_default_launcher_matches_explicit_n():
    so = compile_kernel(verbose=False)
    default = entry(so, "call_hadamard_default")
    explicit = entry(so, "call_hadamard", DISPATCH_ARGS)

    rng = np.random.default_rng(11)
    x_np = rng.standard_normal((4 * rows_for(N), N)).astype(np.float16)
    outs = []
    for fn, extra in ((default, ()), (explicit, (N,))):
        x = torch.from_numpy(x_np.copy()).npu()
        fn(64, stream_ptr(), ctypes.c_void_p(x.data_ptr()), x.shape[0], *extra)
        torch.npu.synchronize()
        outs.append(x.cpu().numpy())
    assert np.array_equal(
        outs[0].view(np.uint16), outs[1].view(np.uint16)
    ), "call_hadamard_default does not agree with call_hadamard(n=N): DEFAULT_N != N"
