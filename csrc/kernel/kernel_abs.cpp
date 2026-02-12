/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted (subject to the limitations in the disclaimer
below) provided that the following conditions are met:

     * Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.

     * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

     * Neither the name of the copyright holder nor the names of its
     contributors may be used to endorse or promote products derived from this
     software without specific prior written permission.

NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
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
