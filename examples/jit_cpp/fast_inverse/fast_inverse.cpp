/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/

// Include the triangular inverse kernel implementation.
// The build script adds csrc/kernel/ to the include path so that
// kernel_utils.h (included by kernel_tri_inv_rec_unroll.cpp) is found.
#include "kernel_tri_inv_rec_unroll.cpp"

/**
 * @brief JIT entry point for the triangular inverse (recursive unroll) kernel.
 *
 * @param blockDim   Number of AI-Core blocks to launch.
 * @param stream     NPU stream handle.
 * @param tensor_out fp32 output buffer (same element count as tensor_in).
 * @param tensor_in  fp16 input buffer holding the upper-triangular matrices
 *                   (diagonal is assumed to be all-ones).
 * @param minus_identity_in  fp16 buffer of size matrix_size×matrix_size
 *                           pre-filled with -I (negative identity).
 * @param matrix_size   Side length of each square matrix (16 / 32 / 64 / 128).
 * @param num_matrices  Total number of matrices to invert.
 * @param num_bsnd_heads  0 for standard (B…ND) layout;
 *                        N (number of heads) for BSND layout.
 * @param cu_seqlens  Optional int32 pointer used only for varlen BSND when the
 *                    device kernel derives chunk metadata itself.
 * @param chunk_indices Optional int32 pointer containing per-chunk row starts
 *                      for the host-precomputed varlen path.
 * @param chunk_valid_sizes Optional int32 pointer containing each chunk's
 *                          runtime size for the host-precomputed varlen path.
 */
extern "C" void call_kernel(uint32_t blockDim, void* stream, void* tensor_out,
                             void* tensor_in, void* minus_identity_in,
                             uint32_t matrix_size, uint32_t num_matrices,
                             uint32_t num_bsnd_heads, void* cu_seqlens,
                             void* chunk_indices, void* chunk_valid_sizes) {
  tri_inv_rec_unroll_fp16<<<blockDim, nullptr, stream>>>(
      tensor_out, tensor_in, minus_identity_in, matrix_size, num_matrices,
      num_bsnd_heads, cu_seqlens, chunk_indices, chunk_valid_sizes);
}
