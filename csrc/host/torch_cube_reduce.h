/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/
#pragma once

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>

#include <algorithm>
#include <limits>
#include <vector>

#include "utils.h"

extern "C" {
void pto_launch_cube_reduce_fp16(uint32_t blockDim, void* stream, void* input,
                                 void* partials, void* output,
                                 uint32_t num_tiles, uint32_t blocks,
                                 uint64_t tail_offset, uint32_t tail_len,
                                 uint32_t size);
void pto_launch_cube_reduce_fp32(uint32_t blockDim, void* stream, void* input,
                                 void* partials, void* output,
                                 uint32_t num_tiles, uint32_t blocks,
                                 uint64_t tail_offset, uint32_t tail_len,
                                 uint32_t size);
void pto_launch_cube_reduce_int8(uint32_t blockDim, void* stream, void* input,
                                 void* partials, void* output,
                                 uint32_t num_tiles, uint32_t blocks,
                                 uint64_t tail_offset, uint32_t tail_len,
                                 uint32_t size);
void pto_launch_cube_row_reduce_fp16(uint32_t blockDim, void* stream,
                                     void* input, void* dest, void* output,
                                     uint32_t rows, uint32_t cols,
                                     uint32_t splits, uint32_t row_tile,
                                     uint32_t col_tile, uint32_t vector_blocks);
void pto_launch_cube_row_reduce_fp32(uint32_t blockDim, void* stream,
                                     void* input, void* dest, void* output,
                                     uint32_t rows, uint32_t cols,
                                     uint32_t splits, uint32_t row_tile,
                                     uint32_t col_tile, uint32_t vector_blocks);
void pto_launch_cube_row_reduce_int8(uint32_t blockDim, void* stream,
                                     void* input, void* dest, void* output,
                                     uint32_t rows, uint32_t cols,
                                     uint32_t splits, uint32_t row_tile,
                                     uint32_t col_tile, uint32_t vector_blocks);
}

namespace pto_isa_ops {

namespace {

// The nd2nz load that stages a tile in L1 takes the source row stride in a
// 16-bit field, so a row of the input may be at most this long.
constexpr int64_t kMaxReduceCols = 65535;

// Accumulator dtype for a given input dtype: int8 sums exactly in int32,
// everything else in fp32.
inline at::TensorOptions ReduceOutputOptions(const at::Tensor& x) {
  return x.options().dtype(x.scalar_type() == at::kChar ? at::kInt
                                                        : at::kFloat);
}

// Shared argument checks for both reductions.
inline void CheckReduceInput(const at::Tensor& x) {
  TORCH_CHECK(x.device().type() == DEVICE_TYPE,
              "cube_reduce: tensor must be on NPU, got ", x.device());
  const auto dtype = x.scalar_type();
  TORCH_CHECK(
      dtype == at::kHalf || dtype == at::kFloat || dtype == at::kChar,
      "cube_reduce: only float16, float32 and int8 inputs are supported, "
      "got ",
      dtype);
  TORCH_CHECK(x.numel() <= std::numeric_limits<uint32_t>::max(),
              "cube_reduce: input length exceeds uint32 range");
}

at::Tensor RunCubeReduceAll(const at::Tensor& x) {
  const auto dtype = x.scalar_type();
  const c10::DeviceGuard device_guard(x.device());
  const auto output_options =
      x.options().dtype(dtype == at::kChar ? at::kInt : at::kFloat);
  // An empty input has no kernel launch to produce its sum, so that one case
  // is zeroed on the host. Everywhere else the Vector stage writes this
  // element unconditionally, so it does not need initializing -- and zeroing
  // it would cost an extra kernel launch, which is a third of the floor on
  // small inputs.
  if (x.numel() == 0) return at::zeros({1}, output_options);
  const at::Tensor output = at::empty({1}, output_options);

  const int64_t size =
      x.numel() >= 128 * 128 ? 128 : (x.numel() >= 64 * 64 ? 64 : 32);
  const int64_t tile_elems = size * size;
  const uint32_t num_tiles = static_cast<uint32_t>(x.numel() / tile_elems);
  const uint64_t tail_offset = static_cast<uint64_t>(num_tiles) * tile_elems;
  const uint32_t tail_len =
      static_cast<uint32_t>(x.numel() - static_cast<int64_t>(tail_offset));
  const uint32_t matrix_size = size;
  // `blocks` is what the Vector stage folds together, so it must be zero when
  // the Cube stage was not given a single whole tile and wrote no partials.
  const uint32_t blocks = std::min(num_tiles, std::max(1u, GetNumCubeCores()));
  const uint32_t block_dim = std::max(1u, blocks);

  // Materialize strides. The kernel is handed data_ptr() and takes its length
  // from the scalars above, so it never reads the shape -- and data_ptr()
  // already points past a view's storage offset. Flattening would only cost a
  // view (about 1.2us of the host path) to produce the same pointer.
  const at::Tensor input = x.contiguous();
  // The Cube stage leaves one row of column sums per block.
  const at::Tensor partials =
      at::empty({static_cast<int64_t>(block_dim), size}, output_options);
  if (dtype == at::kHalf) {
    EXEC_KERNEL_CMD(cube_reduce_fp16, block_dim, input, partials, output,
                    num_tiles, blocks, tail_offset, tail_len, matrix_size);
  } else if (dtype == at::kFloat) {
    EXEC_KERNEL_CMD(cube_reduce_fp32, block_dim, input, partials, output,
                    num_tiles, blocks, tail_offset, tail_len, matrix_size);
  } else {
    EXEC_KERNEL_CMD(cube_reduce_int8, block_dim, input, partials, output,
                    num_tiles, blocks, tail_offset, tail_len, matrix_size);
  }
  return output;
}

at::Tensor RunCubeReduceLastDim(const at::Tensor& x) {
  const auto dtype = x.scalar_type();
  const c10::DeviceGuard device_guard(x.device());
  const auto output_options = ReduceOutputOptions(x);

  // A contiguous tensor of shape (d0, ..., dk, N) is a (rows x N) matrix, and
  // the result keeps every axis but the last. A 0-d input behaves the way
  // torch.sum(x, -1) does on one: a single row holding a single element.
  const std::vector<int64_t> out_sizes(x.sizes().begin(),
                                       x.sizes().end() - (x.dim() > 0 ? 1 : 0));
  const int64_t cols_i64 = x.dim() > 0 ? x.size(-1) : 1;
  int64_t rows_i64 = 1;
  for (const int64_t extent : out_sizes) rows_i64 *= extent;

  // No row to reduce, or every row empty: neither needs a kernel, and an
  // empty row sums to zero by definition rather than by accumulation.
  if (rows_i64 == 0) return at::empty(out_sizes, output_options);
  if (cols_i64 == 0) return at::zeros(out_sizes, output_options);
  TORCH_CHECK(cols_i64 <= kMaxReduceCols,
              "cube_reduce: dim=-1 supports a last dimension of at most ",
              kMaxReduceCols, ", got ", cols_i64);

  const at::Tensor input = x.contiguous();
  const at::Tensor output = at::empty(out_sizes, output_options);
  const uint32_t rows = static_cast<uint32_t>(rows_i64);
  const uint32_t cols = static_cast<uint32_t>(cols_i64);
  const uint32_t cores = std::max(1u, GetNumCubeCores());

  // A mad rounds m up to 16 either way, so a 16-row tile costs the Cube unit
  // nothing extra over a 128-row one and yields eight times as many tiles to
  // spread over the cores. It does pay eight times as many Fixpipe stores,
  // so the larger tile is used once there are rows enough to fill it.
  const uint32_t row_tile = rows >= 128 ? 128 : 16;
  const int64_t element_size = input.element_size();
  // L0A holds row_tile x col_tile input elements and L0B holds col_tile x 16
  // ones; both scratchpads are 64 KB.
  const uint32_t col_tile_cap = static_cast<uint32_t>(
      std::min(65536 / (row_tile * element_size), 65536 / (16 * element_size)));
  // Wider column tiles make longer MTE2 bursts, and a tile narrower than
  // 512 bytes leaves most of a burst's fixed cost unamortized. They are
  // capped at the row length because the final tile of a row mads through
  // its padding.
  uint32_t col_tile = 64;
  for (const uint32_t candidate : {128u, 256u, 512u, 1024u}) {
    if (candidate <= col_tile_cap && candidate <= cols) col_tile = candidate;
  }

  const uint32_t row_tiles = (rows + row_tile - 1) / row_tile;
  const uint32_t col_tiles = (cols + col_tile - 1) / col_tile;
  // Few long rows leave most cores idle, so the column range is split as well
  // and a Vector pass folds the pieces. Never more splits than column tiles:
  // an empty split would leave its partials unwritten.
  uint32_t splits = 1;
  if (row_tiles < cores) {
    splits = std::min((cores + row_tiles - 1) / row_tiles, col_tiles);
  }
  const uint32_t block_dim = std::min(cores, row_tiles * splits);
  const uint32_t vector_blocks =
      std::min(std::max(1u, GetNumVectorCores()), (rows + 511) / 512);

  // With one column range the Cube stage writes the final sums itself and the
  // Vector pass is not launched, so no workspace is allocated for it.
  const at::Tensor dest =
      splits > 1
          ? at::empty({static_cast<int64_t>(splits), rows_i64}, output_options)
          : output;
  if (dtype == at::kHalf) {
    EXEC_KERNEL_CMD(cube_row_reduce_fp16, block_dim, input, dest, output, rows,
                    cols, splits, row_tile, col_tile, vector_blocks);
  } else if (dtype == at::kFloat) {
    EXEC_KERNEL_CMD(cube_row_reduce_fp32, block_dim, input, dest, output, rows,
                    cols, splits, row_tile, col_tile, vector_blocks);
  } else {
    EXEC_KERNEL_CMD(cube_row_reduce_int8, block_dim, input, dest, output, rows,
                    cols, splits, row_tile, col_tile, vector_blocks);
  }
  return output;
}
}  // namespace

/**
 * @brief Sums an input tensor, either entirely or along its last dimension.
 *
 * Reducing the last axis stages each tile through L1 with an nd2nz load,
 * which costs one burst per row of the tile. Rows shorter than about 32
 * elements leave that burst mostly empty and the kernel becomes burst-bound,
 * at which point torch.sum is the faster choice; from 32 elements up it wins.
 *
 * @param [in] x Input tensor on the NPU with dtype fp16, fp32 or int8. Any
 * layout is accepted; a non-contiguous input is materialized first.
 * @param [in] dim Axis to reduce. Omitted sums every element into a
 * single-element tensor. Only the last axis (-1, or `x.dim() - 1`) is
 * supported otherwise, and the result drops that axis. A last dimension of
 * at most 65535 is supported, the widest row the nd2nz source stride holds.
 * @return at::Tensor Sums as fp32 for fp16 and fp32 input, int32 for int8.
 */
at::Tensor run_cube_reduce(const at::Tensor& x,
                           c10::optional<int64_t> dim = c10::nullopt) {
  CheckReduceInput(x);
  if (!dim.has_value()) return RunCubeReduceAll(x);
  // A 0-d tensor is indexed as if it were 1-d, matching torch.sum.
  const int64_t rank = std::max<int64_t>(x.dim(), 1);
  const int64_t axis = dim.value() < 0 ? dim.value() + rank : dim.value();
  TORCH_CHECK(axis == rank - 1,
              "cube_reduce: only the last dimension can be reduced, got dim=",
              dim.value(), " for a ", x.dim(), "D tensor");
  return RunCubeReduceLastDim(x);
}
}  // namespace pto_isa_ops
