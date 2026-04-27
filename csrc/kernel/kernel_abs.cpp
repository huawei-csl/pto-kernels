/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/

#include "kernel_utils.h"

using namespace pto;

/**
 * @brief Elementwise absolute value: `torch.abs(x)`
 *
 * @tparam T Input data type. Supports `fp16` or `fp32`
 * @tparam TILE_LEN Tile length
 * @param x Input tensor
 * @param z Output tensor
 * @param total_size Number of elements
 */
template <typename T, uint32_t TILE_SIZE>
AICORE void runTAbs(__gm__ T* x, __gm__ T* z, uint32_t total_size) {
  // Define GM tile type
  using ShapeDim5 = pto::Shape<1, 1, 1, 1, DYNAMIC>;
  using StrideDim5 = pto::Stride<1, 1, 1, 1, 1>;
  using GlobalData = pto::GlobalTensor<T, ShapeDim5, StrideDim5>;

  // Define UB tile type
  using TileData =
      Tile<TileType::Vec, T, 1, TILE_SIZE, BLayout::RowMajor, 1, DYNAMIC>;

  set_mask_norm();
  set_vector_mask(-1, -1);

  const uint32_t num_aiv_cores = get_block_num();
  const uint32_t aiv_core_id = get_block_idx();

  constexpr uint32_t UB_ZERO_ADDR = 0;
  constexpr uint32_t TILE_SIZE_IN_BYTES = TILE_SIZE * sizeof(T);
  const uint32_t num_tiles = (total_size + TILE_SIZE - 1) / TILE_SIZE;
  const uint32_t num_tiles_per_block =
      (num_tiles + num_aiv_cores - 1) / num_aiv_cores;
  const uint32_t global_offset = aiv_core_id * TILE_SIZE * num_tiles_per_block;

  // Unlock first iteration
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);

  // Loop over tiles
  const uint32_t offset_end =
      min(global_offset + TILE_SIZE * num_tiles_per_block, total_size);
  for (uint32_t inner_offset = global_offset; inner_offset < offset_end;
       inner_offset += TILE_SIZE) {
    const uint32_t remainder_size = offset_end - inner_offset;
    const int32_t remaining_elements =
        remainder_size > TILE_SIZE ? TILE_SIZE : remainder_size;

    // Define tile on GM
    GlobalData xGlobal(x + inner_offset, {remaining_elements});
    GlobalData zGlobal(z + inner_offset, {remaining_elements});

    // Define tile UB buffer
    TileData xTiles(remaining_elements);
    TileData zTiles(remaining_elements);

    // Assign the UB address for each tile
    TASSIGN(xTiles, UB_ZERO_ADDR);
    TASSIGN(zTiles, UB_ZERO_ADDR + TILE_SIZE_IN_BYTES);

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

  // Cleanup flags
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
}

extern "C" __global__ AICORE void vabs_fp16(GM_ADDR x, GM_ADDR z,
                                            uint32_t in_length) {
#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)
  constexpr uint32_t TILE_LEN = 128;
  runTAbs<half, TILE_LEN>((__gm__ half*)x, (__gm__ half*)z, in_length);
#else
  (void)x;
  (void)z;
  (void)in_length;
#endif
}

extern "C" __global__ AICORE void vabs_fp32(GM_ADDR x, GM_ADDR z,
                                            uint32_t in_length) {
#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)

  constexpr uint32_t TILE_LEN = 128;
  runTAbs<float, TILE_LEN>((__gm__ float*)x, (__gm__ float*)z, in_length);
#else
  (void)x;
  (void)z;
  (void)in_length;
#endif
}
