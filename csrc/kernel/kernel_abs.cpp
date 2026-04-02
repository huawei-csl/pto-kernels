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
template <typename T, uint32_t TILE_SIZE>
AICORE void runTAbs(__gm__ T* x, __gm__ T* z, uint32_t total_length) {
  // define GlobalData on global memory with shape and stride
  using ShapeDim5 = pto::Shape<1, 1, 1, 1, TILE_SIZE>;
  using StrideDim5 = pto::Stride<1, 1, 1, 1, 1>;
  using GlobalData = pto::GlobalTensor<T, ShapeDim5, StrideDim5>;

  // Define TileData on UB buffer with static shape and dynamic mask
  using TileData = Tile<TileType::Vec, T, 1, TILE_SIZE, BLayout::RowMajor>;

  set_mask_norm();
  set_vector_mask(-1, -1);

  constexpr uint32_t UB_ZERO_ADDR = 0;
  constexpr uint32_t TILE_SIZE_IN_BYTES = TILE_SIZE * sizeof(T);
  const uint32_t num_tiles = total_length / TILE_SIZE;
  const uint32_t max_num_tiles_per_block_ = num_tiles / get_block_num();
  const uint32_t global_offset =
      block_idx * TILE_SIZE * max_num_tiles_per_block_;

  GlobalData xGlobal(x);
  GlobalData zGlobal(z);

  // Define full tile UB buffers
  TileData xTiles;
  TileData zTiles;

  // Assign the UB address for each tile
  TASSIGN(xTiles, UB_ZERO_ADDR);
  TASSIGN(zTiles, UB_ZERO_ADDR + TILE_SIZE_IN_BYTES);

  // Unlock first iteration
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);

  // Loop for full size tiles
  for (uint32_t i = 0; i < max_num_tiles_per_block_; i++) {
    const uint32_t inner_offset = global_offset + i * TILE_SIZE;
    // Prepare read GM offset
    TASSIGN(xGlobal, x + inner_offset);
    TASSIGN(zGlobal, z + inner_offset);

    // MTE2 (load) wait for vector core to be done
    // (previous iteration's computation)
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

    // Load data from global memory to UB buffer
    TLOAD(xTiles, xGlobal);

    // Signal end of current load to vector core
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // Vector core wait for MTE2 (current load)
    // and MTE3 (previous store) to be done
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);

    // Perform elementwise absolute value
    TABS(zTiles, xTiles);

    // Signal both MTE2 and MTE3 that the computation is done
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    // MTE3 (store) wait for vector core to be done
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    // Store data from UB buffer to global memory
    TSTORE(zGlobal, zTiles);

    // Signal end of MTE3 (current store) to vector core
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }

  // Tail tile handling (if any)
  const int32_t remaining_elements = total_length % TILE_SIZE;
  if (remaining_elements && block_idx == get_block_num() - 1) {
    // Handle the remaining elements

    // Define global data for the tail tile
    using TailShapeDim5 = pto::Shape<1, 1, 1, 1, DYNAMIC>;
    using TailGlobalData = pto::GlobalTensor<T, TailShapeDim5, StrideDim5>;
    TailGlobalData xTailGlobal(x + num_tiles * TILE_SIZE, {remaining_elements});
    TailGlobalData zTailGlobal(z + num_tiles * TILE_SIZE, {remaining_elements});

    // Define tail tile UB buffers
    using TailTileData =
        Tile<TileType::Vec, T, 1, TILE_SIZE, BLayout::RowMajor, 1, DYNAMIC>;
    TailTileData xTailTile(remaining_elements);
    TailTileData zTailTile(remaining_elements);

    // Assign the UB address for tail tile
    TASSIGN(xTailTile, UB_ZERO_ADDR);
    TASSIGN(zTailTile, UB_ZERO_ADDR + 3 * TILE_SIZE_IN_BYTES);

    // MTE2 (load) wait for vector core to be done
    // (previous iteration's computation)
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

    // Load tail tile data from global memory to UB buffer
    TLOAD(xTailTile, xTailGlobal);

    // Signal end of current load to vector core
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // Vector core wait for MTE2 (current load)
    // and MTE3 (previous store) to be done
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);

    // Perform elementwise absolute value on the tail tile
    TABS(zTailTile, xTailTile);

    // Signal both MTE2 and MTE3 that the computation is done
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    // MTE3 (store) wait for vector core to be done
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    // Store tail tile data from UB buffer to global memory
    TSTORE(zTailGlobal, zTailTile);

    // Signal end of MTE3 (current store) to vector core
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }

  // Cleanup flags
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
}

extern "C" __global__ AICORE void vabs_fp16(GM_ADDR x, GM_ADDR z,
                                            uint32_t in_length) {
  constexpr uint32_t TILE_LEN = 64;
  runTAbs<half, TILE_LEN>((__gm__ half*)x, (__gm__ half*)z, in_length);
}

extern "C" __global__ AICORE void vabs_fp32(GM_ADDR x, GM_ADDR z,
                                            uint32_t in_length) {
  constexpr uint32_t TILE_LEN = 64;
  runTAbs<float, TILE_LEN>((__gm__ float*)x, (__gm__ float*)z, in_length);
}

#endif
