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
 * @tparam X_TILE_SIZE Tile size for x
 * @param values Input tensor of values
 * @param indices Input tensor of indices
 * @param x Input tensor
 * @param z Output tensor
 * @param x_size Number of elements in x
 * @param idx_size Number of elements in indices
 */
template <typename T, uint32_t TILE_SIZE, uint32_t TILE_SIZE_X>
AICORE void runTCsrGather(__gm__ T* values, __gm__ int32_t* indices, __gm__ T* x, __gm__ T* z, uint32_t x_size, uint32_t indices_size) {
  set_mask_norm();
  set_vector_mask(-1, -1);

  const uint32_t num_aiv_cores = get_block_num();
  const uint32_t aiv_core_id = get_block_idx();

  constexpr uint32_t UB_ZERO_ADDR = 0;
  constexpr uint32_t TILE_SIZE_IN_BYTES = TILE_SIZE * sizeof(T);
  const uint32_t num_tiles = (indices_size + TILE_SIZE - 1) / TILE_SIZE;
  const uint32_t num_tiles_per_block =
      (num_tiles + num_aiv_cores - 1) / num_aiv_cores;
  const uint32_t global_offset = aiv_core_id * TILE_SIZE * num_tiles_per_block;

  // Define GM tile type
  using ShapeDim5 = pto::Shape<1, 1, 1, 1, DYNAMIC>;
  using StrideDim5 = pto::Stride<1, 1, 1, 1, 1>;
  using GlobalData = pto::GlobalTensor<T, ShapeDim5, StrideDim5>;

  // Define UB tile type for x
  using TileDataX =
      Tile<TileType::Vec, T, 1, TILE_SIZE_X, BLayout::RowMajor, 1, DYNAMIC>;

  // Copy full x to UB
  GlobalData xGlobal(x, {static_cast<int32_t>(x_size)});
  TileDataX xTiles(x_size);
  TASSIGN(xTiles, UB_ZERO_ADDR);
  TLOAD(xTiles, xGlobal);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);


  // Unlock first iteration
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);

  // Loop for full size tiles
  for (uint32_t inner_offset = global_offset; inner_offset < indices_size;
       inner_offset += TILE_SIZE) {
    const uint32_t remainder_size = indices_size - inner_offset;
    const int32_t remaining_elements =
        remainder_size > TILE_SIZE ? TILE_SIZE : remainder_size;

    // Define UB tile type
    using TileDataIdx =
        Tile<TileType::Vec, int32_t, 1, TILE_SIZE, BLayout::RowMajor, 1, DYNAMIC>;
    using TileDataVal =
        Tile<TileType::Vec, T, 1, TILE_SIZE, BLayout::RowMajor, 1, DYNAMIC>;

    // Define tile on GM
    GlobalData valGlobal(values + inner_offset, {remaining_elements});
    GlobalData zGlobal(z + inner_offset, {remaining_elements});
    using GlobalDataIdx = pto::GlobalTensor<int32_t, ShapeDim5, StrideDim5>;
    GlobalDataIdx idxGlobal(indices + inner_offset, {remaining_elements});
    
    // Define tile UB buffer
    TileDataVal valTiles(remaining_elements);
    TileDataVal wTiles(remaining_elements);
    TileDataVal zTiles(remaining_elements);
    TileDataIdx idxTiles(remaining_elements);

    // Assign the UB address for each tile
    constexpr uint32_t TILE_SIZE_X_IN_BYTES = TILE_SIZE_X * sizeof(uint32_t);
    TASSIGN(valTiles, UB_ZERO_ADDR + TILE_SIZE_X_IN_BYTES);
    TASSIGN(idxTiles, UB_ZERO_ADDR + TILE_SIZE_X_IN_BYTES + TILE_SIZE_IN_BYTES);
    TASSIGN(wTiles, UB_ZERO_ADDR + TILE_SIZE_X_IN_BYTES + 2 * TILE_SIZE_IN_BYTES);
    TASSIGN(zTiles, UB_ZERO_ADDR + TILE_SIZE_X_IN_BYTES + 3 * TILE_SIZE_IN_BYTES);


    // MTE2 (load) wait for vector core to be done
    // (previous iteration's computation)
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

    // Load data from global memory to UB buffer
    TLOAD(valTiles, valGlobal);
    TLOAD(idxTiles, idxGlobal);

    // Signal end of current load to vector core
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // Vector core wait for MTE2 (current load)
    // and MTE3 (previous store) to be done
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);

    // Gather
    TGATHER(wTiles, xTiles, idxTiles);

    // Mul
    TMUL(zTiles, valTiles, wTiles);


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

extern "C" __global__ AICORE void vcsr_gather_fp16(GM_ADDR values, GM_ADDR indices, GM_ADDR x, GM_ADDR z, uint32_t x_size, uint32_t indices_size) {
  constexpr uint32_t TILE_SIZE = 128;
  constexpr uint32_t TILE_SIZE_X = 512;
  runTCsrGather<half, TILE_SIZE, TILE_SIZE_X>((__gm__ half*)values, (__gm__ int32_t*)indices, (__gm__ half*)x, (__gm__ half*)z, x_size, indices_size);
}

extern "C" __global__ AICORE void vcsr_gather_fp32(GM_ADDR values, GM_ADDR indices, GM_ADDR x, GM_ADDR z, uint32_t x_size, uint32_t indices_size) {
  constexpr uint32_t TILE_SIZE = 128;
  constexpr uint32_t TILE_SIZE_X = 512;
  runTCsrGather<float, TILE_SIZE, TILE_SIZE_X>((__gm__ float*)values, (__gm__ int32_t*)indices, (__gm__ float*)x, (__gm__ float*)z, x_size, indices_size);
}

#endif
