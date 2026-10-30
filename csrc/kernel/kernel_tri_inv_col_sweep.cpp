/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/

#include "kernel_tri_inv_col_sweep.h"

extern "C" __global__ AICORE void triv_inv_col_sweep_fp16(
    GM_ADDR x, GM_ADDR z, uint32_t total_tiles, uint32_t matrix_size,
    uint32_t is_lower, uint32_t num_bsnd_heads, GM_ADDR cu_seqlens) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if defined(__DAV_VEC__)

  // Input/output dtype
  using fp16_t = __gm__ half*;
  __gm__ int32_t* _cu_seqlens = (__gm__ int32_t*)cu_seqlens;

  if (matrix_size == 16) {
    dispatchTriInv<half, 16>((fp16_t)x, (fp16_t)z, total_tiles, is_lower,
                             num_bsnd_heads, _cu_seqlens);
  } else if (matrix_size == 32) {
    dispatchTriInv<half, 32>((fp16_t)x, (fp16_t)z, total_tiles, is_lower,
                             num_bsnd_heads, _cu_seqlens);
  } else if (matrix_size == 64) {
    dispatchTriInv<half, 64>((fp16_t)x, (fp16_t)z, total_tiles, is_lower,
                             num_bsnd_heads, _cu_seqlens);
  } else if (matrix_size == 128) {
    dispatchTriInv<half, 128>((fp16_t)x, (fp16_t)z, total_tiles, is_lower,
                              num_bsnd_heads, _cu_seqlens);
  }
#endif
}

extern "C" __global__ AICORE void triv_inv_col_sweep_fp32(
    GM_ADDR x, GM_ADDR z, uint32_t total_tiles, uint32_t matrix_size,
    uint32_t is_lower, uint32_t num_bsnd_heads, GM_ADDR cu_seqlens) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if defined(__DAV_VEC__)

  // Input/output dtype
  using fp32_t = __gm__ float*;
  __gm__ int32_t* _cu_seqlens = (__gm__ int32_t*)cu_seqlens;

  if (matrix_size == 16) {
    dispatchTriInv<float, 16>((fp32_t)x, (fp32_t)z, total_tiles, is_lower,
                              num_bsnd_heads, _cu_seqlens);
  } else if (matrix_size == 32) {
    dispatchTriInv<float, 32>((fp32_t)x, (fp32_t)z, total_tiles, is_lower,
                              num_bsnd_heads, _cu_seqlens);
  } else if (matrix_size == 64) {
    dispatchTriInv<float, 64>((fp32_t)x, (fp32_t)z, total_tiles, is_lower,
                              num_bsnd_heads, _cu_seqlens);
  } else if (matrix_size == 128) {
    dispatchTriInv<float, 128>((fp32_t)x, (fp32_t)z, total_tiles, is_lower,
                               num_bsnd_heads, _cu_seqlens);
  }
#endif
}

extern "C" void pto_launch_triv_inv_col_sweep_fp16(
    uint32_t blockDim, void* stream, void* x, void* z, uint32_t total_tiles,
    uint32_t matrix_size, uint32_t is_lower, uint32_t num_bsnd_heads,
    void* cu_seqlens) {
  triv_inv_col_sweep_fp16<<<blockDim, nullptr, stream>>>(
      (GM_ADDR)x, (GM_ADDR)z, total_tiles, matrix_size, is_lower,
      num_bsnd_heads, (GM_ADDR)cu_seqlens);
}

extern "C" void pto_launch_triv_inv_col_sweep_fp32(
    uint32_t blockDim, void* stream, void* x, void* z, uint32_t total_tiles,
    uint32_t matrix_size, uint32_t is_lower, uint32_t num_bsnd_heads,
    void* cu_seqlens) {
  triv_inv_col_sweep_fp32<<<blockDim, nullptr, stream>>>(
      (GM_ADDR)x, (GM_ADDR)z, total_tiles, matrix_size, is_lower,
      num_bsnd_heads, (GM_ADDR)cu_seqlens);
}
