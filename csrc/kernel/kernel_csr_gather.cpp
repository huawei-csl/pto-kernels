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

// clang-format off: so it does not get wrongfully flagged by linter
#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t*  // To avoid #include "kernel_operator.h"
#endif
// clang-format on

using namespace pto;

constexpr uint32_t UB_USABLE_BYTES = 192 * 1024;
constexpr uint32_t UB_ZERO_ADDR = 0;

/**
 * @brief Performs the CSR gather operation on the vector core as described
 * in SEGMV algorithm in [1]
 *
 * [1] Segmented operations for sparse matrix computation on vector
 * multiprocessors: https://dl.acm.org/doi/abs/10.5555/865221
 *
 * The operation is defined as:
 * z = values * x[indices]
 *
 * @tparam T Input data type. Supports `fp16` or `fp32`
 * @tparam TILE_LEN Tile length
 * @tparam X_TILE_SIZE Tile size for x
 * @param values Input tensor of values
 * @param indices Input tensor of indices
 * @param x Input tensor
 * @param z Output tensor
 * @param x_size Number of elements in x
 * @param indices_size Number of elements in indices
 */
template <typename T, uint32_t TILE_SIZE, uint32_t TILE_SIZE_X>
AICORE void runTCsrGather(__gm__ T* values, __gm__ int32_t* indices,
                          __gm__ T* x, __gm__ T* z, uint32_t x_size,
                          uint32_t indices_size) {
  set_mask_norm();
  set_vector_mask(-1, -1);

  constexpr uint32_t N_BUF = 3;
  // UB zero address and tile sizes
  constexpr uint32_t TILE_SIZE_IN_BYTES = TILE_SIZE * sizeof(T);
  constexpr uint32_t TILE_SIZE_X_IN_BYTES = TILE_SIZE_X * sizeof(T);
  constexpr uint32_t TILE_SIZE_IDX_IN_BYTES = TILE_SIZE * sizeof(int32_t);
  constexpr uint32_t V_T_ADDR = UB_ZERO_ADDR;
  constexpr uint32_t W_T_ADDR = V_T_ADDR + N_BUF * TILE_SIZE_IN_BYTES;
  // PTO-ISA 8.5.0 bug, gather uses dst address for a tmp of size index.
  constexpr uint32_t Z_T_ADDR = W_T_ADDR + 2 * N_BUF * TILE_SIZE_IDX_IN_BYTES;
  constexpr uint32_t IDX_T_ADDR = Z_T_ADDR + N_BUF * TILE_SIZE_IN_BYTES;
  constexpr uint32_t X_T_ADDR = IDX_T_ADDR + N_BUF * TILE_SIZE_IDX_IN_BYTES;
  static_assert(X_T_ADDR + TILE_SIZE_X_IN_BYTES <= UB_USABLE_BYTES,
                "CSR gather UB layout exceeds usable UB.");

  const uint32_t num_aiv_cores = get_block_num();
  const uint32_t aiv_core_id = get_block_idx();
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
  TASSIGN(xTiles, X_T_ADDR);
  TLOAD(xTiles, xGlobal);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);

  // Unlock first iteration
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);

  // Loop over tiles
  uint8_t stage = 0;
  const uint32_t offset_end =
      min(global_offset + TILE_SIZE * num_tiles_per_block, indices_size);
  for (uint32_t inner_offset = global_offset; inner_offset < offset_end;
       inner_offset += TILE_SIZE) {
    const uint32_t remainder_size = offset_end - inner_offset;
    const int32_t remaining_elements =
        remainder_size > TILE_SIZE ? TILE_SIZE : remainder_size;

    // Define UB tile type
    using TileDataIdx = Tile<TileType::Vec, int32_t, 1, TILE_SIZE,
                             BLayout::RowMajor, 1, DYNAMIC>;
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
    TASSIGN(valTiles, V_T_ADDR + stage * TILE_SIZE_IN_BYTES);
    TASSIGN(wTiles, W_T_ADDR + stage * 2 * TILE_SIZE_IDX_IN_BYTES);
    TASSIGN(zTiles, Z_T_ADDR + stage * TILE_SIZE_IN_BYTES);
    TASSIGN(idxTiles, IDX_T_ADDR + stage * TILE_SIZE_IDX_IN_BYTES);

    event_t ev0 = (event_t)((stage % N_BUF) * 2);
    event_t ev1 = (event_t)((stage % N_BUF) * 2 + 1);

    // MTE2 (load) wait for gather to be done
    // (previous iteration's computation)
    wait_flag(PIPE_V, PIPE_MTE2, ev1);

    // Load data from global memory to UB buffer
    TLOAD(idxTiles, idxGlobal);

    // Signal end of current load to vector core
    set_flag(PIPE_MTE2, PIPE_V, ev0);

    // Wait for MT2 (current load) to be done
    wait_flag(PIPE_MTE2, PIPE_V, ev0);
    // This barrier is needed only for the
    // first iteration when the load of x might not be done yet
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);

    // Wait for mul to be done (previous iteration's computation)
    pipe_barrier(PIPE_V);

    // Gather
    TGATHER(wTiles, xTiles, idxTiles);

    // Signal end of gather to MTE2 (next load)
    set_flag(PIPE_V, PIPE_MTE2, ev1);

    // Wait for mul to be done (previous iteration's computation)
    wait_flag(PIPE_V, PIPE_MTE2, ev0);

    // Load data from global memory to UB buffer
    TLOAD(valTiles, valGlobal);

    // Signal end of current load to vector core
    set_flag(PIPE_MTE2, PIPE_V, ev0);

    // Wait for MT2 (current load) and MT3
    // (previous iteration's store) to be done
    wait_flag(PIPE_MTE2, PIPE_V, ev0);
    wait_flag(PIPE_MTE3, PIPE_V, ev0);

    // Wait for gather to be done
    pipe_barrier(PIPE_V);

    // Mul
    TMUL(zTiles, valTiles, wTiles);

    // Signal end of computation to MT3 (current store)
    // and to MTE2 (next load)
    set_flag(PIPE_V, PIPE_MTE3, ev0);
    set_flag(PIPE_V, PIPE_MTE2, ev0);

    // Wait for mul to be done before store
    wait_flag(PIPE_V, PIPE_MTE3, ev0);

    // Store result from UB buffer to GM
    TSTORE(zGlobal, zTiles);

    // Signal end of MTE3 (current store) to vector core
    set_flag(PIPE_MTE3, PIPE_V, ev0);

    // Next stage
    stage = (stage + 1) % N_BUF;
  }
  // Cleanup flags
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
}

extern "C" __global__ AICORE void csr_gather_fp16(GM_ADDR values,
                                                  GM_ADDR indices, GM_ADDR x,
                                                  GM_ADDR z, uint32_t x_size,
                                                  uint32_t indices_size) {
  constexpr uint32_t TILE_SIZE = 512;
  constexpr uint32_t TILE_SIZE_X = 40960;
  runTCsrGather<half, TILE_SIZE, TILE_SIZE_X>(
      (__gm__ half*)values, (__gm__ int32_t*)indices, (__gm__ half*)x,
      (__gm__ half*)z, x_size, indices_size);
}

extern "C" __global__ AICORE void csr_gather_fp32(GM_ADDR values,
                                                  GM_ADDR indices, GM_ADDR x,
                                                  GM_ADDR z, uint32_t x_size,
                                                  uint32_t indices_size) {
  constexpr uint32_t TILE_SIZE = 512;
  constexpr uint32_t TILE_SIZE_X = 40960;
  runTCsrGather<float, TILE_SIZE, TILE_SIZE_X>(
      (__gm__ float*)values, (__gm__ int32_t*)indices, (__gm__ float*)x,
      (__gm__ float*)z, x_size, indices_size);
}

#endif
