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
#include <tuple>
#include <vector>

#include "utils.h"

extern "C" {
void pto_launch_cube_reduce_dim_fp16(uint32_t blockDim, void* stream,
                                     void* input, void* pattern, void* output,
                                     uint32_t outer, uint32_t mid,
                                     uint32_t inner, uint32_t tile_cols,
                                     uint32_t mode);
void pto_launch_cube_reduce_dim_fp32(uint32_t blockDim, void* stream,
                                     void* input, void* pattern, void* output,
                                     uint32_t outer, uint32_t mid,
                                     uint32_t inner, uint32_t tile_cols,
                                     uint32_t mode);
void pto_launch_cube_reduce_dim_int8(uint32_t blockDim, void* stream,
                                     void* input, void* pattern, void* output,
                                     uint32_t outer, uint32_t mid,
                                     uint32_t inner, uint32_t tile_cols,
                                     uint32_t mode);
}

namespace pto_isa_ops {

/// The GM -> L1 ND2NZ load carries the source row pitch in a 16-bit field, so
/// the axis a tile is cut across cannot be longer than this. Both reduction
/// modes read tiles out of a matrix whose row pitch is one of the input's own
/// extents, which is therefore what the limit applies to.
constexpr int64_t kMaxRowPitch = 65535;

/// Elements in one input tile, which must match kTileElems in the kernel: the
/// host picks only the tile's width and the kernel derives its height as this
/// divided by that width. Holding the volume fixed means a wider tile is a
/// shorter one, and that the mad volume a choice costs is simply the tile count
/// times this.
constexpr int64_t kTileElems = 16384;
/// Smallest tile width supported by the kernel dispatch.
/// The three decompositions, matching the kernel's own constants.
constexpr uint32_t kModeStrided = 0;
constexpr uint32_t kModeContiguous = 1;
constexpr uint32_t kModeFolded = 2;
/// Widest kept extent the folded mode will take. Its mad spends one column of
/// n per kept element, so the arithmetic it wastes grows with this while the
/// strided mode's shrinks; past a fractal the strided mode is the better tool.
constexpr int64_t kMaxFoldedInner = 32;
/// Declared width of the pattern operand, matching kFractalCols in the kernel.
constexpr int64_t kFractalCols = 32;
constexpr int64_t kTileColsMin = 32;
/// Largest tile width supported by the kernel dispatch.
constexpr int64_t kTileColsMax = 512;

/// Highest input rank the host folds for the kernel. The fold itself is
/// rank-agnostic, so this bound only marks how far the contract is tested.
constexpr int64_t kMaxRank = 5;

/**
 * @brief Sums a contiguous tensor of rank 1 to 5 along one axis, like
 * torch.sum(x, dim).
 *
 * The reduced axis is removed from the result, so a (D0, D1, D2) input reduces
 * to (D1, D2), (D0, D2) or (D0, D1) for dim 0, 1 and 2 respectively, and a 1D
 * input reduces to a scalar tensor.
 *
 * Rank reaches the kernel only as the three extents the axes fold into, so
 * every rank costs what its own (outer, mid, inner) view costs: reducing dim 1
 * of (D0, D1, D2, D3) is the same work as reducing dim 1 of
 * (D0, D1, D2 * D3).
 *
 * @param [in] x Input tensor on the NPU with dtype fp16, fp32 or int8. A
 * non-contiguous input is materialized first.
 * @param [in] dim Axis to reduce, in [-rank, rank - 1].
 * @return at::Tensor The sums with the reduced axis dropped, fp32 for fp16 and
 * fp32 input and int32 for int8 input.
 */
at::Tensor run_cube_reduce_dim(const at::Tensor& x, int64_t dim) {
  TORCH_CHECK(x.device().type() == DEVICE_TYPE,
              "cube_reduce_dim: tensor must be on NPU, got ", x.device());
  const auto dtype = x.scalar_type();
  TORCH_CHECK(
      dtype == at::kHalf || dtype == at::kFloat || dtype == at::kChar,
      "cube_reduce_dim: only float16, float32 and int8 inputs are supported, "
      "got ",
      dtype);
  TORCH_CHECK(x.dim() >= 1 && x.dim() <= kMaxRank,
              "cube_reduce_dim: expects an input tensor of rank 1 to ",
              kMaxRank, ", got ", x.dim(), " dimensions");
  const int64_t axis = dim < 0 ? dim + x.dim() : dim;
  TORCH_CHECK(axis >= 0 && axis < x.dim(),
              "cube_reduce_dim: dim out of range, expected it to be in [",
              -x.dim(), ", ", x.dim() - 1, "], but got ", dim);

  // Fold the axes around the reduced one into the (outer, mid, inner) view the
  // kernel reduces: element (o, i) of the result is the sum over m of
  // x[o, m, i]. Only the reduced extent and the two folds reach the kernel, so
  // past this point neither the rank nor which axis was named still matters --
  // a contiguous tensor's axes before the reduced one are one contiguous run
  // of rows, and those after it one contiguous run of columns, whatever their
  // number.
  int64_t outer = 1;
  int64_t inner = 1;
  for (int64_t index = 0; index < axis; ++index) outer *= x.sizes()[index];
  for (int64_t index = axis + 1; index < x.dim(); ++index) {
    inner *= x.sizes()[index];
  }
  const int64_t mid = x.sizes()[axis];
  TORCH_CHECK(x.numel() <= std::numeric_limits<uint32_t>::max(),
              "cube_reduce_dim: input length exceeds uint32 range");

  std::vector<int64_t> output_sizes;
  for (int64_t index = 0; index < x.dim(); ++index) {
    if (index != axis) output_sizes.push_back(x.sizes()[index]);
  }
  const c10::DeviceGuard device_guard(x.device());
  const auto output_options =
      x.options().dtype(dtype == at::kChar ? at::kInt : at::kFloat);
  // Reducing a zero-length axis sums nothing per output element, which the
  // kernel has no tile to express, so those zeros are produced on the host.
  if (mid == 0) return at::zeros(output_sizes, output_options);
  at::Tensor output = at::empty(output_sizes, output_options);
  // With a zero-length kept axis there is no output element to write.
  if (output.numel() == 0) return output;

  // The tiles are cut across the kept inner axis when that axis is the
  // contiguous one, and across the reduced axis otherwise, so that is the
  // pitch the load has to carry.
  const int64_t row_pitch = inner == 1 ? mid : inner;
  TORCH_CHECK(row_pitch <= kMaxRowPitch, "cube_reduce_dim: reducing dim ", dim,
              " of a ", x.sizes(), " tensor needs a row pitch of ", row_pitch,
              ", which exceeds the ", kMaxRowPitch,
              " the tile load supports; reduce an axis whose ",
              inner == 1 ? "own extent is" : "trailing extents multiply to",
              " at most ", kMaxRowPitch);

  // Pick the tile shape. The mad runs at a fixed rows x cols whatever a tile
  // actually holds, so each direction pays for whatever the padding adds, and
  // because the volume is fixed the padded area is exactly the tile count times
  // that volume -- one number standing for both the arithmetic and the loads.
  //
  // Minimizing it alone is not enough, though: every output element belongs to
  // exactly one work item, so a shape that splits into fewer items than there
  // are cores leaves cores idle whatever its tiles cost. Occupancy therefore
  // comes first, the padded area second, and a wider tile breaks the ties --
  // at equal cost the wider tile reads longer contiguous runs out of the input.
  const int64_t tiled_rows = inner == 1 ? outer : mid;
  const int64_t tiled_cols = inner == 1 ? mid : inner;
  const int64_t cores = std::max<int64_t>(1, GetNumCubeCores());
  int64_t tile_cols = kTileColsMin;
  int64_t work = 0;
  std::tuple<int, int64_t, int64_t, int64_t, int64_t, int64_t> best;
  for (int64_t cols = kTileColsMin; cols <= kTileColsMax; cols *= 2) {
    const int64_t rows = kTileElems / cols;
    const int64_t padded_rows = ((tiled_rows + rows - 1) / rows) * rows;
    const int64_t padded_cols = ((tiled_cols + cols - 1) / cols) * cols;
    const int64_t items = inner == 1 ? (outer + rows - 1) / rows
                                     : outer * ((inner + cols - 1) / cols);
    const bool fills_cores = items >= cores;
    // How far the worse of the two directions overshoots its extent. Dividing
    // by tiled_rows * tiled_cols would give the factor itself, and that divisor
    // is the same for every candidate, so the numerators order them as it does.
    // Two tiles of equal area can still pad very differently -- 256x64 and
    // 64x256 both cover a 64x64 extent at four times the mad width of 128x128
    // -- and the balanced one measures fastest.
    const int64_t overshoot =
        std::max(padded_rows * tiled_cols, padded_cols * tiled_rows);
    // Lexicographic and compared as one key: fills the cores, else as many as
    // possible; then the smaller padded area; then the longest run the tile can
    // actually fetch, which a width past the extent cannot lengthen; then the
    // balanced tile; then the narrowest one, since the rest is padding.
    const std::tuple<int, int64_t, int64_t, int64_t, int64_t, int64_t> key(
        fills_cores ? 1 : 0, fills_cores ? 0 : items,
        -(padded_rows * padded_cols), std::min(cols, tiled_cols), -overshoot,
        -cols);
    if (cols == kTileColsMin || key > best) {
      best = key;
      tile_cols = cols;
      work = items;
    }
  }
  const int64_t strided_tile_cols = tile_cols;
  const int64_t strided_work = work;
  uint32_t mode = inner == 1 ? kModeContiguous : kModeStrided;

  // A small kept extent is the strided mode's worst case: it pays one work
  // item per output row and pads the mad out to a fractal in both directions,
  // so the tiles end up tiny and numerous. The folded mode reads the same
  // input as R whole slabs of a contiguous (outer, mid * inner) matrix and
  // reduces them in one mad against the pattern, which costs `inner` columns
  // of n per kept element instead. Take it when it saves enough to pay for
  // the pattern, which costs a couple of launches to build.
  const int64_t folded_width = mid * inner;
  int64_t pattern_chunk = 0;
  if (inner > 1 && inner <= kMaxFoldedInner && folded_width <= kMaxRowPitch) {
    const int64_t strided_cost = work * static_cast<int64_t>(kTileElems);
    int64_t best_folded = 0;
    for (int64_t cols = kTileColsMin; cols <= kTileColsMax; cols *= 2) {
      const int64_t rows = kTileElems / cols;
      const int64_t chunk = (cols / inner) * inner;
      if (chunk <= 0) continue;
      const int64_t items = (outer + rows - 1) / rows;
      if (items < cores && items < work) continue;
      const int64_t tiles = items * ((folded_width + chunk - 1) / chunk);
      const int64_t cost = tiles * kTileElems;
      if (best_folded == 0 || cost < best_folded) {
        best_folded = cost;
        pattern_chunk = chunk;
        tile_cols = cols;
        work = items;
      }
    }
    // Four times is where the saving clears the pattern's own cost at the
    // sizes this mode is for; below it the two decompositions are close
    // enough that the extra launches would eat the difference.
    if (best_folded != 0 && best_folded * 4 <= strided_cost) {
      mode = kModeFolded;
    } else {
      pattern_chunk = 0;
    }
  }
  if (mode != kModeFolded) {
    tile_cols = strided_tile_cols;
    work = strided_work;
  }

  const uint32_t block_dim =
      static_cast<uint32_t>(std::min<int64_t>(work, cores));

  // Materialize strides. The kernel is handed data_ptr() and takes its extents
  // from the scalars above, so it never reads the shape -- and data_ptr()
  // already points past a view's storage offset.
  const at::Tensor input = x.contiguous();
  // P[j, i] is 1 when j % inner == i: identity matrices stacked vertically,
  // which is what folds each column of the input onto its output slot. It
  // depends only on `inner` and the chunk width, so it carries no input data
  // and is the same for every tile. Built only for the mode that reads it.
  // Padded out to the whole tile the kernel declares, so that its load is a
  // full-tile one: a partly valid load into a tile taller than one fractal
  // misplaces the rows past the first.
  const at::Tensor pattern =
      mode == kModeFolded
          ? at::constant_pad_nd(
                at::eye(inner, x.options()).repeat({pattern_chunk / inner, 1}),
                {0, kFractalCols - inner, 0, tile_cols - pattern_chunk})
          : input;
  const uint32_t outer_arg = static_cast<uint32_t>(outer);
  const uint32_t mid_arg = static_cast<uint32_t>(mid);
  const uint32_t inner_arg = static_cast<uint32_t>(inner);
  const uint32_t tile_cols_arg = static_cast<uint32_t>(tile_cols);
  if (dtype == at::kHalf) {
    EXEC_KERNEL_CMD(cube_reduce_dim_fp16, block_dim, input, pattern, output,
                    outer_arg, mid_arg, inner_arg, tile_cols_arg, mode);
  } else if (dtype == at::kFloat) {
    EXEC_KERNEL_CMD(cube_reduce_dim_fp32, block_dim, input, pattern, output,
                    outer_arg, mid_arg, inner_arg, tile_cols_arg, mode);
  } else {
    EXEC_KERNEL_CMD(cube_reduce_dim_int8, block_dim, input, pattern, output,
                    outer_arg, mid_arg, inner_arg, tile_cols_arg, mode);
  }
  return output;
}
}  // namespace pto_isa_ops
