# Triton-Ascend `chunk_o` Performance

| Kernel | Shape `(B,H,L,D,C)` | Median ms | TFLOP/s | GiB/s |
| --- | --- | ---: | ---: | ---: |
| triton_ascend | `(8, 20, 1024, 128, 64)` | 6.615 | 2.43 | 23.62 |
| vllm_static_no_g | `(8, 20, 1024, 128, 64)` | 1.482 | 10.87 | 105.45 |
| vllm_static_uniform_g | `(8, 20, 1024, 128, 64)` | 1.582 | 10.18 | 98.75 |
| vllm_varlen_no_g | `(8, 20, 1024, 128, 64)` | 2.038 | 7.90 | 76.67 |
| vllm_varlen_uniform_g | `(8, 20, 1024, 128, 64)` | 2.094 | 7.69 | 74.60 |
| pto_cpp | `(8, 20, 1024, 128, 64)` | 0.580 | 27.79 | 269.62 |
| triton_ascend | `(16, 20, 1024, 128, 64)` | 13.319 | 2.42 | 23.46 |
| vllm_static_no_g | `(16, 20, 1024, 128, 64)` | 2.686 | 11.99 | 116.34 |
| vllm_static_uniform_g | `(16, 20, 1024, 128, 64)` | 2.760 | 11.67 | 113.24 |
| vllm_varlen_no_g | `(16, 20, 1024, 128, 64)` | 3.193 | 10.09 | 97.86 |
| vllm_varlen_uniform_g | `(16, 20, 1024, 128, 64)` | 3.254 | 9.90 | 96.02 |
| pto_cpp | `(16, 20, 1024, 128, 64)` | 1.109 | 29.04 | 281.70 |
| triton_ascend | `(24, 20, 2048, 128, 64)` | 38.034 | 2.54 | 24.65 |
| vllm_static_no_g | `(24, 20, 2048, 128, 64)` | 7.465 | 12.95 | 125.59 |
| vllm_static_uniform_g | `(24, 20, 2048, 128, 64)` | 7.538 | 12.82 | 124.37 |
| vllm_varlen_no_g | `(24, 20, 2048, 128, 64)` | 7.964 | 12.13 | 117.71 |
| vllm_varlen_uniform_g | `(24, 20, 2048, 128, 64)` | 8.010 | 12.06 | 117.04 |
| pto_cpp | `(24, 20, 2048, 128, 64)` | 3.064 | 31.54 | 306.00 |

## PTO / Kernel Speedup

| Shape `(B,H,L,D,C)` | Kernel | PTO / Kernel speedup | Kernel - PTO TFLOP/s delta |
| --- | --- | ---: | ---: |
| `(8, 20, 1024, 128, 64)` | `triton_ascend` | 11.42x | -25.36 |
| `(8, 20, 1024, 128, 64)` | `vllm_static_no_g` | 2.56x | -16.92 |
| `(8, 20, 1024, 128, 64)` | `vllm_static_uniform_g` | 2.73x | -17.61 |
| `(8, 20, 1024, 128, 64)` | `vllm_varlen_no_g` | 3.52x | -19.89 |
| `(8, 20, 1024, 128, 64)` | `vllm_varlen_uniform_g` | 3.61x | -20.10 |
| `(16, 20, 1024, 128, 64)` | `triton_ascend` | 12.01x | -26.62 |
| `(16, 20, 1024, 128, 64)` | `vllm_static_no_g` | 2.42x | -17.04 |
| `(16, 20, 1024, 128, 64)` | `vllm_static_uniform_g` | 2.49x | -17.36 |
| `(16, 20, 1024, 128, 64)` | `vllm_varlen_no_g` | 2.88x | -18.95 |
| `(16, 20, 1024, 128, 64)` | `vllm_varlen_uniform_g` | 2.93x | -19.14 |
| `(24, 20, 2048, 128, 64)` | `triton_ascend` | 12.41x | -29.00 |
| `(24, 20, 2048, 128, 64)` | `vllm_static_no_g` | 2.44x | -18.60 |
| `(24, 20, 2048, 128, 64)` | `vllm_static_uniform_g` | 2.46x | -18.72 |
| `(24, 20, 2048, 128, 64)` | `vllm_varlen_no_g` | 2.60x | -19.41 |
| `(24, 20, 2048, 128, 64)` | `vllm_varlen_uniform_g` | 2.61x | -19.48 |

Notes:
- Reported TFLOP/s and GiB/s are computed from the same algorithm-level model for both kernels.
- The Triton kernel is forward-only, head-first only, and currently omits gating and varlen support.
- The copied vLLM-style kernel is benchmarked with precomputed `h` state and pre-transposed inputs, so transpose/setup cost is excluded as requested.
- On this device, the copied vLLM-style kernel compiled and ran for `C=64`, but the unmodified `BT=C=128` configuration overflowed UB and was not benchmarked.
- `TRITON_ALL_BLOCKS_PARALLEL` is intentionally left disabled here because it produced incorrect outputs for this kernel.
