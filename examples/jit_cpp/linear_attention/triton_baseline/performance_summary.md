# Triton-Ascend `chunk_o` Performance

| Kernel | Shape `(B,H,L,D,C)` | Median ms | TFLOP/s | GiB/s |
| --- | --- | ---: | ---: | ---: |
| triton_ascend | `(8, 20, 1024, 128, 128)` | 5.323 | 4.03 | 29.35 |
| pto_cpp | `(8, 20, 1024, 128, 128)` | 0.407 | 52.72 | 383.61 |
| triton_ascend | `(16, 20, 1024, 128, 128)` | 10.353 | 4.15 | 30.18 |
| pto_cpp | `(16, 20, 1024, 128, 128)` | 0.705 | 60.89 | 443.03 |
| triton_ascend | `(24, 20, 2048, 128, 128)` | 30.728 | 4.19 | 30.51 |
| pto_cpp | `(24, 20, 2048, 128, 128)` | 1.776 | 72.55 | 527.90 |

## Comparison

| Shape `(B,H,L,D,C)` | PTO / Triton speedup | Triton - PTO TFLOP/s delta |
| --- | ---: | ---: |
| `(8, 20, 1024, 128, 128)` | 13.08x | -48.69 |
| `(16, 20, 1024, 128, 128)` | 14.69x | -56.74 |
| `(24, 20, 2048, 128, 128)` | 17.30x | -68.36 |

## Larger-Shape Sanity Check

These Triton-only runs were used to verify that the corrected benchmark no longer scales to unrealistic throughput or bandwidth values.

| Kernel | Shape `(B,H,L,D,C)` | Median ms | TFLOP/s | GiB/s |
| --- | --- | ---: | ---: | ---: |
| triton_ascend | `(24, 20, 4096, 128, 128)` | 59.518 | 4.33 | 31.50 |
| triton_ascend | `(32, 20, 4096, 128, 128)` | 80.937 | 4.25 | 30.89 |
| triton_ascend | `(24, 20, 8192, 128, 128)` | 118.080 | 4.36 | 31.76 |

Notes:
- Reported TFLOP/s and GiB/s are computed from the same algorithm-level model for both kernels.
- The Triton kernel is forward-only, head-first only, and currently omits gating and varlen support.
- `TRITON_ALL_BLOCKS_PARALLEL` is intentionally left disabled here because it produced incorrect outputs for this kernel.
