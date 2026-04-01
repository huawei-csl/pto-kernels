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

### Supported matrix sizes

`matrix_size` (last dimension of the input tensor) must be one of: **16, 32,
64, 128**.

### Layout conventions

| `num_bsnd_heads` | Memory layout |
|-----------------|---------------|
| `0` (default) | Each matrix stored consecutively in row-major order (`B × … × N × D × D`) |
| `> 0` | BSND layout: `(B, S, N, D)` where S is chunked into tiles of size D and N heads are interleaved |

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
