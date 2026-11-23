# Triton-Ascend `chunk_o` Performance

| Kernel | Shape `(B,H,L,D,C)` | Median ms | TFLOP/s | GiB/s |
| --- | --- | ---: | ---: | ---: |
| triton_mask_onthefly | `(8, 20, 1024, 128, 64)` | 1.305 | 12.34 | 119.69 |
| triton_mask_cached | `(8, 20, 1024, 128, 64)` | 1.323 | 12.17 | 118.10 |
| vllm_static_no_g | `(8, 20, 1024, 128, 64)` | 1.470 | 10.96 | 106.29 |
| vllm_static_uniform_g | `(8, 20, 1024, 128, 64)` | 1.565 | 10.29 | 99.82 |
| vllm_varlen_no_g | `(8, 20, 1024, 128, 64)` | 2.030 | 7.93 | 76.97 |
| vllm_varlen_uniform_g | `(8, 20, 1024, 128, 64)` | 2.090 | 7.71 | 74.78 |
| pto_cpp | `(8, 20, 1024, 128, 64)` | 0.583 | 27.61 | 267.87 |
| triton_mask_onthefly | `(16, 20, 1024, 128, 64)` | 2.260 | 14.25 | 138.28 |
| triton_mask_cached | `(16, 20, 1024, 128, 64)` | 2.312 | 13.93 | 135.14 |
| vllm_static_no_g | `(16, 20, 1024, 128, 64)` | 2.681 | 12.01 | 116.56 |
| vllm_static_uniform_g | `(16, 20, 1024, 128, 64)` | 2.758 | 11.68 | 113.31 |
| vllm_varlen_no_g | `(16, 20, 1024, 128, 64)` | 3.200 | 10.07 | 97.66 |
| vllm_varlen_uniform_g | `(16, 20, 1024, 128, 64)` | 3.252 | 9.90 | 96.08 |
| pto_cpp | `(16, 20, 1024, 128, 64)` | 1.121 | 28.74 | 278.82 |
| triton_mask_onthefly | `(24, 20, 2048, 128, 64)` | 6.096 | 15.85 | 153.79 |
| triton_mask_cached | `(24, 20, 2048, 128, 64)` | 6.264 | 15.43 | 149.67 |
| vllm_static_no_g | `(24, 20, 2048, 128, 64)` | 7.458 | 12.96 | 125.71 |
| vllm_static_uniform_g | `(24, 20, 2048, 128, 64)` | 7.529 | 12.83 | 124.51 |
| vllm_varlen_no_g | `(24, 20, 2048, 128, 64)` | 7.978 | 12.11 | 117.51 |
| vllm_varlen_uniform_g | `(24, 20, 2048, 128, 64)` | 8.007 | 12.07 | 117.08 |
| pto_cpp | `(24, 20, 2048, 128, 64)` | 3.072 | 31.46 | 305.18 |

## PTO / Kernel Speedup

| Shape `(B,H,L,D,C)` | Kernel | PTO / Kernel speedup | Kernel - PTO TFLOP/s delta |
| --- | --- | ---: | ---: |
| `(8, 20, 1024, 128, 64)` | `triton_mask_onthefly` | 2.24x | -15.27 |
| `(8, 20, 1024, 128, 64)` | `triton_mask_cached` | 2.27x | -15.44 |
| `(8, 20, 1024, 128, 64)` | `vllm_static_no_g` | 2.52x | -16.66 |
| `(8, 20, 1024, 128, 64)` | `vllm_static_uniform_g` | 2.68x | -17.32 |
| `(8, 20, 1024, 128, 64)` | `vllm_varlen_no_g` | 3.48x | -19.68 |
| `(8, 20, 1024, 128, 64)` | `vllm_varlen_uniform_g` | 3.58x | -19.90 |
| `(16, 20, 1024, 128, 64)` | `triton_mask_onthefly` | 2.02x | -14.49 |
| `(16, 20, 1024, 128, 64)` | `triton_mask_cached` | 2.06x | -14.81 |
| `(16, 20, 1024, 128, 64)` | `vllm_static_no_g` | 2.39x | -16.73 |
| `(16, 20, 1024, 128, 64)` | `vllm_static_uniform_g` | 2.46x | -17.06 |
| `(16, 20, 1024, 128, 64)` | `vllm_varlen_no_g` | 2.85x | -18.67 |
| `(16, 20, 1024, 128, 64)` | `vllm_varlen_uniform_g` | 2.90x | -18.84 |
| `(24, 20, 2048, 128, 64)` | `triton_mask_onthefly` | 1.98x | -15.61 |
| `(24, 20, 2048, 128, 64)` | `triton_mask_cached` | 2.04x | -16.03 |
| `(24, 20, 2048, 128, 64)` | `vllm_static_no_g` | 2.43x | -18.50 |
| `(24, 20, 2048, 128, 64)` | `vllm_static_uniform_g` | 2.45x | -18.62 |
| `(24, 20, 2048, 128, 64)` | `vllm_varlen_no_g` | 2.60x | -19.34 |
| `(24, 20, 2048, 128, 64)` | `vllm_varlen_uniform_g` | 2.61x | -19.39 |

## Triton Cached vs On-The-Fly

| Shape `(B,H,L,D,C)` | Cached ms | On-the-fly ms | On-the-fly / Cached | Cached TFLOP/s delta |
| --- | ---: | ---: | ---: | ---: |
| `(8, 20, 1024, 128, 64)` | 1.323 | 1.305 | 0.99x | -0.16 |
| `(16, 20, 1024, 128, 64)` | 2.312 | 2.260 | 0.98x | -0.32 |
| `(24, 20, 2048, 128, 64)` | 6.264 | 6.096 | 0.97x | -0.43 |

Notes:
- Reported TFLOP/s and GiB/s are computed from the same algorithm-level model for both kernels.
- The Triton kernel is forward-only, head-first only, and currently omits gating and varlen support.
- `triton_mask_onthefly` computes the causal mask inside the Triton kernel but still uses precomputed chunk states `h`.
- `triton_mask_cached` is benchmarked with precomputed chunk states `h` and a cached causal mask, so state construction and mask setup are excluded from timed measurements.
- The copied vLLM-style kernel is benchmarked with precomputed `h` state and pre-transposed inputs, so transpose/setup cost is excluded as requested.
- On this device, the copied vLLM-style kernel compiled and ran for `C=64`, but the unmodified `BT=C=128` configuration overflowed UB and was not benchmarked.
- `TRITON_ALL_BLOCKS_PARALLEL` is intentionally left disabled here because it produced incorrect outputs for this kernel.
