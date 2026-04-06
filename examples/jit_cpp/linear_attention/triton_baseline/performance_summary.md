# Triton-Ascend `chunk_o` Performance

| Kernel | Shape `(B,H,L,D,C)` | Median ms | TFLOP/s | GiB/s |
| --- | --- | ---: | ---: | ---: |
| triton_ascend | `(8, 20, 1024, 128, 64)` | 1.260 | 12.78 | 124.02 |
| vllm_static_no_g | `(8, 20, 1024, 128, 64)` | 1.461 | 11.02 | 106.95 |
| vllm_static_uniform_g | `(8, 20, 1024, 128, 64)` | 1.543 | 10.44 | 101.24 |
| vllm_varlen_no_g | `(8, 20, 1024, 128, 64)` | 1.944 | 8.29 | 80.39 |
| vllm_varlen_uniform_g | `(8, 20, 1024, 128, 64)` | 2.011 | 8.01 | 77.70 |
| pto_cpp | `(8, 20, 1024, 128, 64)` | 0.583 | 27.60 | 267.79 |
| pto_cpp | `(8, 20, 1024, 128, 128)` | 0.407 | 52.72 | 383.61 |
| triton_ascend | `(16, 20, 1024, 128, 64)` | 2.215 | 14.54 | 141.07 |
| vllm_static_no_g | `(16, 20, 1024, 128, 64)` | 2.669 | 12.07 | 117.08 |
| vllm_static_uniform_g | `(16, 20, 1024, 128, 64)` | 2.732 | 11.79 | 114.39 |
| vllm_varlen_no_g | `(16, 20, 1024, 128, 64)` | 3.119 | 10.33 | 100.19 |
| vllm_varlen_uniform_g | `(16, 20, 1024, 128, 64)` | 3.172 | 10.15 | 98.51 |
| pto_cpp | `(16, 20, 1024, 128, 64)` | 1.062 | 30.33 | 294.25 |
| pto_cpp | `(16, 20, 1024, 128, 128)` | 0.705 | 60.89 | 443.03 |
| triton_ascend | `(24, 20, 2048, 128, 64)` | 6.053 | 15.97 | 154.89 |
| vllm_static_no_g | `(24, 20, 2048, 128, 64)` | 7.450 | 12.97 | 125.84 |
| vllm_static_uniform_g | `(24, 20, 2048, 128, 64)` | 7.503 | 12.88 | 124.95 |
| vllm_varlen_no_g | `(24, 20, 2048, 128, 64)` | 7.885 | 12.26 | 118.89 |
| vllm_varlen_uniform_g | `(24, 20, 2048, 128, 64)` | 7.930 | 12.19 | 118.22 |
| pto_cpp | `(24, 20, 2048, 128, 64)` | 3.051 | 31.67 | 307.26 |
| pto_cpp | `(24, 20, 2048, 128, 128)` | 1.776 | 72.55 | 527.90 |

## PTO / Kernel Speedup

| Shape `(B,H,L,D,C)` | Kernel | PTO / Kernel speedup | Kernel - PTO TFLOP/s delta |
| --- | --- | ---: | ---: |
| `(8, 20, 1024, 128, 64)` | `triton_ascend` | 2.16x | -14.82 |
| `(8, 20, 1024, 128, 64)` | `vllm_static_no_g` | 2.50x | -16.58 |
| `(8, 20, 1024, 128, 64)` | `vllm_static_uniform_g` | 2.65x | -17.17 |
| `(8, 20, 1024, 128, 64)` | `vllm_varlen_no_g` | 3.33x | -19.32 |
| `(8, 20, 1024, 128, 64)` | `vllm_varlen_uniform_g` | 3.45x | -19.59 |
| `(8, 20, 1024, 128, 128)` | `triton_ascend` | 3.10x | -39.94 |
| `(8, 20, 1024, 128, 128)` | `vllm_static_no_g` | 3.59x | -41.70 |
| `(8, 20, 1024, 128, 128)` | `vllm_static_uniform_g` | 3.79x | -42.28 |
| `(8, 20, 1024, 128, 128)` | `vllm_varlen_no_g` | 4.78x | -44.43 |
| `(8, 20, 1024, 128, 128)` | `vllm_varlen_uniform_g` | 4.94x | -44.71 |
| `(16, 20, 1024, 128, 64)` | `triton_ascend` | 2.09x | -15.79 |
| `(16, 20, 1024, 128, 64)` | `vllm_static_no_g` | 2.51x | -18.26 |
| `(16, 20, 1024, 128, 64)` | `vllm_static_uniform_g` | 2.57x | -18.54 |
| `(16, 20, 1024, 128, 64)` | `vllm_varlen_no_g` | 2.94x | -20.00 |
| `(16, 20, 1024, 128, 64)` | `vllm_varlen_uniform_g` | 2.99x | -20.18 |
| `(16, 20, 1024, 128, 128)` | `triton_ascend` | 3.14x | -46.35 |
| `(16, 20, 1024, 128, 128)` | `vllm_static_no_g` | 3.79x | -48.82 |
| `(16, 20, 1024, 128, 128)` | `vllm_static_uniform_g` | 3.88x | -49.10 |
| `(16, 20, 1024, 128, 128)` | `vllm_varlen_no_g` | 4.42x | -50.56 |
| `(16, 20, 1024, 128, 128)` | `vllm_varlen_uniform_g` | 4.50x | -50.74 |
| `(24, 20, 2048, 128, 64)` | `triton_ascend` | 1.98x | -15.71 |
| `(24, 20, 2048, 128, 64)` | `vllm_static_no_g` | 2.44x | -18.70 |
| `(24, 20, 2048, 128, 64)` | `vllm_static_uniform_g` | 2.46x | -18.79 |
| `(24, 20, 2048, 128, 64)` | `vllm_varlen_no_g` | 2.58x | -19.42 |
| `(24, 20, 2048, 128, 64)` | `vllm_varlen_uniform_g` | 2.60x | -19.49 |
| `(24, 20, 2048, 128, 128)` | `triton_ascend` | 3.41x | -56.58 |
| `(24, 20, 2048, 128, 128)` | `vllm_static_no_g` | 4.20x | -59.58 |
| `(24, 20, 2048, 128, 128)` | `vllm_static_uniform_g` | 4.22x | -59.67 |
| `(24, 20, 2048, 128, 128)` | `vllm_varlen_no_g` | 4.44x | -60.29 |
| `(24, 20, 2048, 128, 128)` | `vllm_varlen_uniform_g` | 4.46x | -60.36 |

Notes:
- Reported TFLOP/s and GiB/s are computed from the same algorithm-level model for both kernels.
- The Triton kernel is forward-only, head-first only, and currently omits gating and varlen support.
- The updated custom Triton kernel is benchmarked with precomputed chunk states `h`, so state construction is excluded from timed measurements.
- The copied vLLM-style kernel is benchmarked with precomputed `h` state and pre-transposed inputs, so transpose/setup cost is excluded as requested.
- The `pto_cpp` `C=128` rows are included as fair references for the same `(B, H, L, D)` workloads because `C` is an internal algorithm parameter.
- On this device, the Triton kernels in this baseline could not be compiled/benchmarked for `C=128`; the copied vLLM-style kernel's unmodified `BT=C=128` configuration overflowed UB.
- `TRITON_ALL_BLOCKS_PARALLEL` is intentionally left disabled here because it produced incorrect outputs for this kernel.
