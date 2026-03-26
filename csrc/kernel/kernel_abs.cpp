/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/

#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)

#define MEMORY_BASE

#include <pto/pto-inst.hpp>

#define GM_ADDR __gm__ uint8_t*  // To avoid #include "kernel_operator.h"

using namespace pto;

/**
 * @brief Elementwise absolute value: `torch.abs(x)`
 *
 * @tparam T Input data type. Supports `fp16` or `fp32`
 * @tparam TILE_LEN Tile length
 * @param x Input tensor
 * @param z Output tensor
 * @param total_length Number of elements
 */
template <typename T, unsigned TILE_LEN>
AICORE void runTAbs(__gm__ T* x, __gm__ T* z, uint32_t total_length) {
  // define GlobalData on global memory with shape and stride
  using ShapeDim5 = pto::Shape<1, 1, 1, TILE_LEN, TILE_LEN>;
  using StrideDim5 = pto::Stride<1, 1, 1, TILE_LEN, 1>;
  using GlobalData = pto::GlobalTensor<T, ShapeDim5, StrideDim5>;

  // define TileData on UB buffer with static shape and dynamic mask
  using TileData =
      Tile<TileType::Vec, T, TILE_LEN, TILE_LEN, BLayout::RowMajor, -1, -1>;

  set_mask_norm();
  set_vector_mask(-1, -1);

  constexpr unsigned UB_ZERO_ADDR = 0;
  constexpr unsigned TILE_SIZE_IN_BYTES = TILE_LEN * sizeof(T);
  const uint32_t num_tiles = total_length / TILE_LEN;
  const uint32_t max_num_tiles_per_block_ = num_tiles / get_block_num();
  const uint32_t global_offset =
      block_idx * TILE_LEN * max_num_tiles_per_block_;

  GlobalData xGlobal(x);
  GlobalData zGlobal(z);

  // define ping-pong buffer for related tiles
  TileData xTiles(TILE_LEN, TILE_LEN);
  TileData zTiles(TILE_LEN, TILE_LEN);

  // assign the UB address for each tile
  TASSIGN(xTiles, UB_ZERO_ADDR);
  TASSIGN(zTiles, UB_ZERO_ADDR + TILE_SIZE_IN_BYTES);

  for (uint32_t i = 0; i < max_num_tiles_per_block_; i++) {
    const unsigned inner_offset = global_offset + i * TILE_LEN;
    // Prepare read GM offset
    TASSIGN(xGlobal, x + inner_offset);
    TASSIGN(zGlobal, z + inner_offset);

    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

    // load data from global memory to UB buffer
    TLOAD(xTiles, xGlobal);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // perform elementwise absolute value
    TABS(zTiles, xTiles);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    // store data from UB buffer to global memory
    TSTORE(zGlobal, zTiles);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
}

extern "C" __global__ AICORE void vabs_fp16(GM_ADDR x, GM_ADDR z,
                                            uint32_t in_length) {
  constexpr unsigned TILE_LEN = 64;
  runTAbs<half, TILE_LEN>((__gm__ half*)x, (__gm__ half*)z, in_length);
}

extern "C" __global__ AICORE void vabs_fp32(GM_ADDR x, GM_ADDR z,
                                            uint32_t in_length) {
  constexpr unsigned TILE_LEN = 64;
  runTAbs<float, TILE_LEN>((__gm__ float*)x, (__gm__ float*)z, in_length);
}

#endif
