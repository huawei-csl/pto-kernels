/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/

#include "kernel_tri_inv_col_sweep.h"
#include "kernel_tri_inv_rec_unroll.h"

/**
 * @brief Inverts the tiles of a tensor on both engines at once.
 *
 * The tensor is cut in two halves along the tile index. The first half goes
 * to the cube units, which invert it with the recursive-doubling / unrolled
 * recursion of tri_inv_rec_unroll; the second half goes to the vector units,
 * which invert it with the rank-1 column sweep of tri_inv_col_sweep. Both
 * algorithms already own a disjoint set of tiles -- and therefore disjoint
 * rows of the output -- so the two halves need no cross-core synchronization
 * and simply run concurrently for the length of the kernel.
 *
 * Splitting the work this way is what the AIC_1_2 mix mode buys: a cube-only
 * kernel leaves the 2 * blockDim vector cores idle and a vector-only kernel
 * leaves the blockDim cube cores idle, while here each engine works on its
 * own half of the tiles with the algorithm that suits it.
 *
 * The halves are equal in tile count, not in runtime: the engines are not
 * equally fast per tile, so for a given shape one half can finish before the
 * other and the kernel ends with the slower engine.
 *
 * @tparam T Input and output data type. Only fp16 is supported: the cube
 * half also accepts bf16, but the rank-1 sweep needs TSUB and TROWEXPANDMUL,
 * which have no bf16 form, so bf16 is out for the mix kernel as a whole.
 *
 * @param tensor_out Pointer to global memory to store the inverses.
 * @param tensor_in Pointer to the input tensor in global memory.
 * @param minus_eye_in Pointer to the negative identity matrix, used by the
 * cube half only.
 * @param matrix_size Size of each individual matrix / tile: 16, 32, 64 or 128.
 * @param num_matrices Total number of matrices / tiles in the tensor.
 * @param num_bsnd_heads Number of heads N for BSND input, or 0 when each tile
 * is stored contiguously and row-wise.
 * @param is_lower Whether the input matrices are lower (1) or upper (0)
 * triangular.
 * @param cu_seqlens Cumulative sequence lengths for variable-length BSND
 * input, or nullptr when every chunk is full.
 * @param max_doubling_block_size Block size at which the cube half crosses
 * over from recursive doubling to the unrolled recursion.
 */
template <typename T>
AICORE inline void runTriInvMix(__gm__ T* tensor_out, __gm__ T* tensor_in,
                                __gm__ T* minus_eye_in, uint32_t matrix_size,
                                uint32_t num_matrices, uint32_t num_bsnd_heads,
                                uint32_t is_lower, __gm__ int32_t* cu_seqlens,
                                uint32_t max_doubling_block_size) {
  // Tiles [0, num_cube_tiles) belong to the cube half, the rest to the
  // vector half. The odd tile goes to the vector half, which has twice as
  // many cores to spread it over.
  const uint32_t num_cube_tiles = num_matrices / 2;
  const uint32_t num_vector_tiles = num_matrices - num_cube_tiles;

#if defined(__DAV_CUBE__)
  (void)num_vector_tiles;
  run_tri_inv_rec_unroll_per_num_matrices<T, T, 1 /* NumTilesPerCubeIter */,
                                          false /* IsBSND */>(
      tensor_out, tensor_in, minus_eye_in, matrix_size, num_cube_tiles,
      num_bsnd_heads, is_lower, cu_seqlens, max_doubling_block_size,
      0 /* tile_id_base */);
#elif defined(__DAV_VEC__)
  (void)minus_eye_in;
  (void)max_doubling_block_size;
  // The sweep takes (in, out); the recursion above takes (out, in).
  if (matrix_size == 16) {
    dispatchTriInv<T, 16>(tensor_in, tensor_out, num_vector_tiles, is_lower,
                          num_bsnd_heads, cu_seqlens, num_cube_tiles);
  } else if (matrix_size == 32) {
    dispatchTriInv<T, 32>(tensor_in, tensor_out, num_vector_tiles, is_lower,
                          num_bsnd_heads, cu_seqlens, num_cube_tiles);
  } else if (matrix_size == 64) {
    dispatchTriInv<T, 64>(tensor_in, tensor_out, num_vector_tiles, is_lower,
                          num_bsnd_heads, cu_seqlens, num_cube_tiles);
  } else if (matrix_size == 128) {
    dispatchTriInv<T, 128>(tensor_in, tensor_out, num_vector_tiles, is_lower,
                           num_bsnd_heads, cu_seqlens, num_cube_tiles);
  }
#endif
}

/*
 * @brief: Wrapper of kernel for input/output dtype fp16.
 */
extern "C" __global__ AICORE void tri_inv_mix_fp16(
    __gm__ void* tensor_out, __gm__ void* tensor_in, __gm__ void* minus_eye_in,
    uint32_t matrix_size, uint32_t num_matrices, uint32_t num_bsnd_heads,
    uint32_t is_lower, __gm__ void* cu_seqlens,
    uint32_t max_doubling_block_size) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  runTriInvMix<half>((__gm__ half*)tensor_out, (__gm__ half*)tensor_in,
                     (__gm__ half*)minus_eye_in, matrix_size, num_matrices,
                     num_bsnd_heads, is_lower, (__gm__ int32_t*)cu_seqlens,
                     max_doubling_block_size);
}

// Host-callable launch shims: the `<<<>>>` syntax is only
// understood by the kernel compiler, so the launch lives here
// rather than in the host wrappers under csrc/host/.
extern "C" void pto_launch_tri_inv_mix_fp16(
    uint32_t blockDim, void* stream, void* tensor_out, void* tensor_in,
    void* minus_eye_in, uint32_t matrix_size, uint32_t num_matrices,
    uint32_t num_bsnd_heads, uint32_t is_lower, void* cu_seqlens,
    uint32_t max_doubling_block_size) {
  tri_inv_mix_fp16<<<blockDim, nullptr, stream>>>(
      (__gm__ void*)tensor_out, (__gm__ void*)tensor_in,
      (__gm__ void*)minus_eye_in, matrix_size, num_matrices, num_bsnd_heads,
      is_lower, (__gm__ void*)cu_seqlens, max_doubling_block_size);
}
