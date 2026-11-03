/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/

#include "kernel_tri_inv_rec_unroll.h"

/*
 * @brief: Wrapper of kernel for input/output dtype bf16 .
 *
 */
extern "C" __global__ AICORE void tri_inv_rec_unroll_bf16(
    __gm__ void* tensor_out, __gm__ void* tensor_in, __gm__ void* minus_eye_in,
    uint32_t matrix_size, uint32_t num_matrices, uint32_t num_bsnd_heads,
    uint32_t is_lower, __gm__ void* cu_seqlens,
    uint32_t max_doubling_block_size) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
  __gm__ bfloat16_t* _tensor_out = (__gm__ bfloat16_t*)tensor_out;
  __gm__ bfloat16_t* _tensor_in = (__gm__ bfloat16_t*)tensor_in;
  __gm__ bfloat16_t* _minus_eye_in = (__gm__ bfloat16_t*)minus_eye_in;
  __gm__ int32_t* _cu_seqlens = (__gm__ int32_t*)cu_seqlens;

  run_tri_inv_rec_unroll_per_num_matrices<
      bfloat16_t, bfloat16_t, 1 /* NumTilesPerCubeIter */, false /* IsBSND */>(
      _tensor_out, _tensor_in, _minus_eye_in, matrix_size, num_matrices,
      num_bsnd_heads, is_lower, _cu_seqlens, max_doubling_block_size);
}

/*
 * @brief: Wrapper of kernel for input/output dtype fp16.
 *
 */
extern "C" __global__ AICORE void tri_inv_rec_unroll_fp16(
    __gm__ void* tensor_out, __gm__ void* tensor_in, __gm__ void* minus_eye_in,
    uint32_t matrix_size, uint32_t num_matrices, uint32_t num_bsnd_heads,
    uint32_t is_lower, __gm__ void* cu_seqlens,
    uint32_t max_doubling_block_size) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
  __gm__ half* _tensor_out = (__gm__ half*)tensor_out;
  __gm__ half* _tensor_in = (__gm__ half*)tensor_in;
  __gm__ half* _minus_eye_in = (__gm__ half*)minus_eye_in;
  __gm__ int32_t* _cu_seqlens = (__gm__ int32_t*)cu_seqlens;

  run_tri_inv_rec_unroll_per_num_matrices<
      half, half, 1 /* NumTilesPerCubeIter */, false /* IsBSND */>(
      _tensor_out, _tensor_in, _minus_eye_in, matrix_size, num_matrices,
      num_bsnd_heads, is_lower, _cu_seqlens, max_doubling_block_size);
}

// Host-callable launch shims: the `<<<>>>` syntax is only
// understood by the kernel compiler, so the launch lives here
// rather than in the host wrappers under csrc/host/.
extern "C" void pto_launch_tri_inv_rec_unroll_bf16(
    uint32_t blockDim, void* stream, void* tensor_out, void* tensor_in,
    void* minus_eye_in, uint32_t matrix_size, uint32_t num_matrices,
    uint32_t num_bsnd_heads, uint32_t is_lower, void* cu_seqlens,
    uint32_t max_doubling_block_size) {
  tri_inv_rec_unroll_bf16<<<blockDim, nullptr, stream>>>(
      (GM_ADDR)tensor_out, (GM_ADDR)tensor_in, (GM_ADDR)minus_eye_in,
      matrix_size, num_matrices, num_bsnd_heads, is_lower, (GM_ADDR)cu_seqlens,
      max_doubling_block_size);
}

extern "C" void pto_launch_tri_inv_rec_unroll_fp16(
    uint32_t blockDim, void* stream, void* tensor_out, void* tensor_in,
    void* minus_eye_in, uint32_t matrix_size, uint32_t num_matrices,
    uint32_t num_bsnd_heads, uint32_t is_lower, void* cu_seqlens,
    uint32_t max_doubling_block_size) {
  tri_inv_rec_unroll_fp16<<<blockDim, nullptr, stream>>>(
      (__gm__ void*)tensor_out, (__gm__ void*)tensor_in,
      (__gm__ void*)minus_eye_in, matrix_size, num_matrices, num_bsnd_heads,
      is_lower, (__gm__ void*)cu_seqlens, max_doubling_block_size);
}
