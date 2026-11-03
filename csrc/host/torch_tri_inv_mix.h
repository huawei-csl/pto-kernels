/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/
#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

#include <algorithm>
#include <cmath>
#include <optional>

#include "utils.h"

extern "C" {

void pto_launch_tri_inv_mix_fp16(uint32_t blockDim, void* stream,
                                 void* tensor_out, void* tensor_in,
                                 void* minus_eye_in, uint32_t matrix_size,
                                 uint32_t num_matrices, uint32_t num_cube_tiles,
                                 uint32_t num_bsnd_heads, uint32_t is_lower,
                                 void* cu_seqlens,
                                 uint32_t max_doubling_block_size);

}  // extern "C"

namespace pto_isa_ops {

/**
 * @brief Default share of the tiles that tri_inv_mix inverts on the cube units.
 *
 * Measured on Ascend910B4 by sweeping the fraction in steps of 0.05 over 1536
 * to 24576 tiles, dense and BSND: the value with the lowest geometric-mean
 * runtime for each matrix size. The cube kernel's lead over the vector kernel
 * peaks at matrix_size 64, hence the larger cube share there.
 *
 * @param matrix_size The dimension of the square matrices.
 * @return double The default fraction of tiles assigned to cube units.
 */
constexpr double DefaultTriInvMixCubeFraction(uint32_t matrix_size) {
  switch (matrix_size) {
    case 16:
      return 0.50;
    case 32:
      return 0.60;
    case 64:
      return 0.65;
    case 128:
      return 0.55;
    default:
      return 0.60;
  }
}

/**
 * @brief Triangular inverse on the cube and the vector units at once.
 *
 * The matrices are split in two parts along the tile index. The first
 * round(cube_fraction * num_tiles) tiles are inverted on the cube units with
 * the "recursive unroll" method of run_tri_inv_rec_unroll, the rest on the
 * vector units with the rank-1 column sweep of run_tri_inv_col_sweep. The
 * parts are disjoint, so the two engines run concurrently without
 * synchronizing.
 *
 * Note: supports the fp16 input dtype only. The cube half would take bf16 as
 * run_tri_inv_rec_unroll does, but the vector half's rank-1 update has no
 * bf16 form.
 *
 * Input range: the cube half carries over the input range of
 * run_tri_inv_rec_unroll -- it forms the powers A^(2^j) of each strictly
 * triangular block in the input dtype, so inputs whose powers overflow fp16
 * come back as NaN. The vector half has no such restriction, so for a given
 * tensor the cube part of the tiles is the one that constrains the input.
 *
 * @param M Input tensor containing square matrices on the last two
 * dimensions. As with run_tri_inv_rec_unroll, the matrices are the strictly
 * triangular A and the kernel returns the inverse of I + A: the diagonal of
 * the input is not read by either half.
 * @param cu_seqlens A 1-dimensional torch tensor that contains the lengths
 * of each input sequence (it is the cummulative sum of the lengths)
 * @param is_bsnd_format A boolean flag indicating if the matrix is in BSND
 * format. If false, then each matrix / tile is stored in consecutive positions
 * in memory, and thus we define num_bsnd_heads=0. If true, then the matrices
 * are stored in "strided mode". In this case we define:
 * num_bsnd_heads=M.size(-2), which is used to do strided load / store ops.
 * @param is_lower If input matrices are lower-triangular (is_lower == true) or
 * upper-triangular (is_lower == false). Default is upper triangular.
 * @param max_doubling_block_size The block-size (<= matrix_size) at which the
 * cube half changes from the recursive doubling to the unrolled algorithm.
 * Defaults to 16. Values higher than 16 can increase performance but can raise
 * numerical instabilities. The vector half ignores it.
 * @param cube_fraction_opt Optional fraction of the tiles, in [0, 1], inverted
 * on the cube units. The split is balanced when it matches the cube units'
 * share of the combined throughput, which depends on the matrix size; when
 * unset, the measured per-size value of DefaultTriInvMixCubeFraction is used.
 * 0 runs the vector algorithm only, 1 the cube algorithm only. Ignored when
 * there are fewer tiles than cube cores: then all of them run on the cube
 * units.
 * @return at::Tensor Tensor containing inverses of input matrices having same
 * dtype as input.
 */
at::Tensor run_tri_inv_mix(
    const at::Tensor& M, const at::Tensor& cu_seqlens = at::zeros({1}),
    const bool is_bsnd_format = false, const bool is_lower = false,
    const uint32_t max_doubling_block_size = 16,
    const std::optional<double> cube_fraction_opt = std::nullopt) {
  const at::Device device = M.options().device();
  const auto dtype = M.options().dtype();

  TORCH_CHECK(device.type() == DEVICE_TYPE,
              "tri_inv_mix: tensor must be on NPU, got ", device);
  TORCH_CHECK(dtype == at::kHalf, "tri_inv_mix: input dtype must be fp16, got ",
              dtype);
  TORCH_CHECK(M.dim() >= 2,
              "tri_inv_mix: input tensor must have at least 2 dimensions, got ",
              M.dim());

  const uint32_t matrix_size = static_cast<uint32_t>(M.size(-1));
  const double cube_fraction =
      cube_fraction_opt.value_or(DefaultTriInvMixCubeFraction(matrix_size));
  TORCH_CHECK(cube_fraction >= 0.0 && cube_fraction <= 1.0,
              "tri_inv_mix: cube_fraction must be in [0, 1], got ",
              cube_fraction);
  const uint32_t num_bsnd_heads =
      is_bsnd_format ? static_cast<uint32_t>(M.size(-2)) : 0;

  const uint32_t num_elems = static_cast<uint32_t>(M.numel());
  uint32_t total_tiles = 0;
  if (is_bsnd_format && (cu_seqlens.numel() > 1)) {
    for (int j = 1; j < cu_seqlens.size(0); ++j) {
      const uint32_t this_seq_len = static_cast<uint32_t>(
          cu_seqlens[j].item<int>() - cu_seqlens[j - 1].item<int>());
      total_tiles +=
          static_cast<uint32_t>((this_seq_len + matrix_size - 1) / matrix_size);
    }
    total_tiles = total_tiles * num_bsnd_heads;
  } else {
    total_tiles =
        static_cast<uint32_t>(num_elems / (matrix_size * matrix_size));
  }

  // Leading tiles on the cube units, the rest on the vector units, in
  // proportion to the two engines' throughput. With fewer tiles than cube
  // cores, every tile gets a cube core of its own and the faster engine takes
  // them all: splitting would only make the vector cores the critical path.
  const uint32_t num_cube_cores = GetNumCubeCores();
  const uint32_t num_cube_tiles =
      total_tiles < num_cube_cores
          ? total_tiles
          : std::min(total_tiles, static_cast<uint32_t>(std::lround(
                                      cube_fraction * total_tiles)));
  const uint32_t num_vector_tiles = total_tiles - num_cube_tiles;

  // blockDim counts cube cores; the mix launch gives each of them two vector
  // cores. Cap it at what either half can keep busy, so small inputs do not
  // start cores that return immediately.
  const uint32_t vector_blocks = (num_vector_tiles + 1) / 2;
  uint32_t block_dim = std::max({num_cube_tiles, vector_blocks, 1u});
  block_dim = std::min(block_dim, num_cube_cores);

  const at::Tensor M_inv = at::zeros_like(M);

  const at::Tensor I_neg =
      at::zeros({matrix_size, matrix_size},
                at::TensorOptions().dtype(dtype).device(device));
  I_neg.fill_diagonal_(-1);

  void* cu_seqlens_ptr = nullptr;
  if (cu_seqlens.numel() != 1) {
    cu_seqlens_ptr = ConvertType(cu_seqlens);
  }

  EXEC_KERNEL_CMD(tri_inv_mix_fp16, block_dim, M_inv, M, I_neg, matrix_size,
                  total_tiles, num_cube_tiles, num_bsnd_heads, is_lower,
                  cu_seqlens_ptr, max_doubling_block_size);

  return M_inv;
}
}  // namespace pto_isa_ops
