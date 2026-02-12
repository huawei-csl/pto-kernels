/**
 Copyright (c) 2026 Huawei Technologies Co., Ltd.
 This program is free software, you can redistribute it and/or modify it
 under the terms and conditions of CANN Open Software License Agreement
 Version 2.0 (the "License"). Please refer to the License for details. You may
 not use this file except in compliance with the License. THIS SOFTWARE IS
 PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS
 OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY,
 OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software
 repository for the full text of the License.
*/

#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)

#define MEMORY_BASE

#include <pto/pto-inst.hpp>

#include "kernel_operator.h"

using namespace pto;

template <typename T, unsigned matrix_size>
AICORE void runTAbs(__gm__ T* x, __gm__ T* z, uint32_t total_length) {
  // define GlobalData on global memory with shape and stride
  using ShapeDim5 = pto::Shape<1, 1, 1, matrix_size, matrix_size>;
  using StrideDim5 = pto::Stride<1, 1, 1, matrix_size, 1>;
  using GlobalData = pto::GlobalTensor<T, ShapeDim5, StrideDim5>;

  // define TileData on UB buffer with static shape and dynamic mask
  using TileData = Tile<TileType::Vec, T, matrix_size, matrix_size,
                        BLayout::RowMajor, -1, -1>;

  set_mask_norm();
  set_vector_mask(-1, -1);

  constexpr uint32_t tile_len = matrix_size * matrix_size;

  constexpr unsigned UB_ZERO_ADDR = 0;
  constexpr unsigned TILE_SIZE_IN_BYTES = tile_len * sizeof(T);

  GlobalData xGlobal(x);
  GlobalData zGlobal(z);

  // define ping-pong buffer for related tiles
  TileData xTiles(matrix_size, matrix_size);
  TileData zTiles(matrix_size, matrix_size);

  // assign the UB address for each tile
  TASSIGN(xTiles, UB_ZERO_ADDR);
  TASSIGN(zTiles, UB_ZERO_ADDR + TILE_SIZE_IN_BYTES);

  // total number of loops of one vector core
  constexpr int32_t loopCount = matrix_size;
  // address offset between vector cores
  // 'block_idx' is a special variable
  const uint32_t offset = block_idx * tile_len;

  for (uint32_t i = 0; i < loopCount; i++) {
    const unsigned inner_offset = offset + i * matrix_size;
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
  constexpr unsigned martix_size = 64;
  // main kernel, in_length is dynamic input
  runTAbs<half, martix_size>((__gm__ half*)x, (__gm__ half*)z, in_length);
}

extern "C" __global__ AICORE void vabs_fp32(GM_ADDR x, GM_ADDR z,
                                            uint32_t in_length) {
  constexpr unsigned martix_size = 64;
  // main kernel, in_length is dynamic input
  runTAbs<float, martix_size>((__gm__ float*)x, (__gm__ float*)z, in_length);
}

#endif
