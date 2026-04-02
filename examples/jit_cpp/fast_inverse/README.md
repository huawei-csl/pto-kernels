## fast_inverse — JIT triangular matrix inverse (recursive unroll)

JIT-compiled example of `kernel_tri_inv_rec_unroll`, which inverts a batch of
upper-triangular fp16 matrices stored in a multi-dimensional tensor.

### Algorithm

Given an input tensor whose last two dimensions form an n×n upper-triangular
matrix U (off-diagonal part only; the diagonal is assumed to be all-ones), the
kernel computes the inverse of (U + I) for every matrix in the batch.

The implementation uses a two-phase recursive approach on Ascend cube cores:

1. **Inv-trick phase** – inverts each 16×16 diagonal fractal block via a
   Neumann-series expansion (`X = (I − M) + (I − M)·M + …`).
2. **Unrolled recursion phase** – assembles partial inverses of progressively
   larger sub-blocks until the full matrix is inverted.

### Files

| File | Purpose |
|------|---------|
| `fast_inverse.cpp` | Thin JIT wrapper: includes the kernel and exposes `call_kernel` |
| `jit_util_fast_inverse.py` | Compiles the kernel with `bisheng` and loads it via `ctypes` |
| `run_fast_inverse.py` | Correctness test suite, including aligned and varlen BSND coverage |
| `run_fast_inverse_varlen_like_triton.py` | Standalone varlen runner that mirrors the Triton `test_solve_tril_varlen` input generation in pure PyTorch |
| `benchmark_bsnd_fast_inverse.py` | Benchmarks fixed BSND vs varlen-uniform BSND and plots effective bandwidth |

### Usage

```bash
export PTO_LIB_PATH=/sources/pto-isa/  # need latest header, not CANN 8.5.0 default

cd examples/jit_cpp/fast_inverse
python run_fast_inverse.py
```

The script compiles `fast_inverse.cpp` on first run (takes ~60 s), then
executes correctness checks across a range of matrix sizes (16, 32, 64, 128)
and batch configurations.

To run the standalone Triton-like varlen coverage:

```bash
export PTO_LIB_PATH=/sources/pto-isa/

cd examples/jit_cpp/fast_inverse
python run_fast_inverse_varlen_like_triton.py
```

That script:

- uses the same varlen case list and input-generation structure as
  `flash-linear-attention/tests/ops/test_solve_tril.py::test_solve_tril_varlen`
- keeps PTO inputs in `float16`
- emulates `chunk_scaled_dot_kkt_fwd` in PyTorch because Triton is not available
- prints a simple pytest-like `PASS` / `FAIL` report plus a final summary

### Supported matrix sizes

`matrix_size` (last dimension of the input tensor) must be one of: **16, 32,
64, 128**.

### Layout conventions

In general, the input to the `fast_inverse` kernels is a set of `D × D` sized triangular matrices. Depending on how these matrices are stored in memory, we might have `contiguous` layout, or the so-called `BSND` layout. The main input is a batch of sequences, and each sequence is then split in "chunks" of length `chunk_size`. This `chunk_size` is the same as the matrix size `D`.

Both layouts depend on the following parameters:
- The parameter `B` denotes the batch-size (or batch-dimension). This is always the first dimension of the input tensor.
- The parameter `N` or `H` (used interchangeably) is the number of heads.
- `D` is equal to the `chunk_size`.
- `S` is the total sum of all sequence lengths combined.
`BSND` can be thought of as the "raw" input tensor. The `contiguous` layout can be obtained, for example, by transposing the `N` and `S` dimensions, and by "chunking" the `S` dimension to chunks of size `S`. The final tensor will be transformed from shape `(B,S,N,D)` to `->(B,N,S/D,D)`, where we assumed that `D` divides `S` for simplicity.

The actual kernel can verify if the input is in `BSND` layout or in `contiguous` layout by specifying the input argument `num_bsnd_heads`. If it is equal to zero, then the format is assumed to be `contiguous`

| `num_bsnd_heads` | Memory layout |
|-----------------|---------------|
| `0` (default) | Each matrix stored consecutively in row-major order (`B × … × D × D`) |
| `> 0` | BSND layout: `(B, S, N, D)` where `S` is chunked into tiles of size D and N heads are interleaved |

### Varlen BSND mode

The standalone example also supports variable-length BSND inputs with the same
external signature as the Triton reference path: callers provide packed BSND
data plus `cu_seqlens`, and the PTO kernel derives each chunk row-start and
tail size internally on NPU. The kernel still inverts dense `D x D` tiles, but
tail chunks load/store only their valid prefix.

### Benchmark

To compare the original fixed-length BSND path against the new varlen path in a
matched-size sanity check:

```bash
export PTO_LIB_PATH=/sources/pto-isa/

cd examples/jit_cpp/fast_inverse
python benchmark_bsnd_fast_inverse.py --chunk-size 64
```

The benchmark script:

- runs only the PTO-ISA BSND kernel
- compares `bsnd-fixed` against `bsnd-varlen-uniform`
- uses uniform `cu_seqlens=[0, T, 2T, ...]` so both paths process the same
  total data size
- reports numerical agreement between the two outputs
- also generates a true-varlen benchmark that plots scattered bandwidth points
  against aggregated sequence length
- writes all CSV and PNG artifacts into `examples/jit_cpp/fast_inverse/benchmark_results/`
