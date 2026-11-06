/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/

#include "kernel_utils.h"

using namespace pto;

namespace {
constexpr uint32_t kColumns = 16;

// Each block accumulates A[S,S] @ ones[S,16] in L0C. Any one
// column of the resulting S x 16 matrix contains its row sums.
template <typename InputT, typename OutputT, uint32_t S>
AICORE void CubeReduce(__gm__ InputT* input, __gm__ InputT* ones,
                       __gm__ OutputT* partials, uint32_t num_tiles) {
  using AGlobal = GlobalTensor<InputT, TileShape2D<InputT, S, S, Layout::ND>,
                               BaseShape2D<InputT, S, S, Layout::ND>>;
  // All entries are one, so the same flat buffer also represents DN.
  // DN -> ZN supports the 32 x 16 int8 B fractal without padding N.
  using BGlobal =
      GlobalTensor<InputT, TileShape2D<InputT, S, kColumns, Layout::DN>,
                   BaseShape2D<InputT, S, kColumns, Layout::DN>, Layout::DN>;
  using CGlobal =
      GlobalTensor<OutputT, TileShape2D<OutputT, S, kColumns, Layout::ND>,
                   BaseShape2D<OutputT, S, kColumns, Layout::ND>>;
  using AMat = Tile<TileType::Mat, InputT, S, S, BLayout::ColMajor, S, S,
                    SLayout::RowMajor, 512>;
  using BMat = Tile<TileType::Mat, InputT, S, kColumns, BLayout::RowMajor, S,
                    kColumns, SLayout::ColMajor, 512>;
  AMat a_l1;
  BMat b_l1;
  TileLeft<InputT, S, S> a_l0;
  TileRight<InputT, S, kColumns> b_l0;
  TileAcc<OutputT, S, kColumns> c_l0;
  TASSIGN(a_l1, 0);
  TASSIGN(b_l1, S * S * sizeof(InputT));
  TASSIGN(a_l0, 0);
  TASSIGN(b_l0, 0);
  TASSIGN(c_l0, 0);

  // Balanced contiguous ranges ensure every launched block writes a result.
  const uint32_t id = get_block_idx();
  const uint32_t blocks = get_block_num();
  const uint32_t count = num_tiles / blocks + (id < num_tiles % blocks);
  const uint32_t begin =
      id * (num_tiles / blocks) + min(id, num_tiles % blocks);
  BGlobal b_global(ones);
  TLOAD(b_l1, b_global);
  kernel_utils::SetWaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
  TMOV(b_l0, b_l1);
  kernel_utils::SetWaitFlag<PIPE_MTE1, PIPE_M>(0);

  for (uint32_t i = 0; i < count; ++i) {
    AGlobal a_global(input + static_cast<uint64_t>(begin + i) * S * S);
    TLOAD(a_l1, a_global);
    kernel_utils::SetWaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
    TMOV(a_l0, a_l1);
    kernel_utils::SetWaitFlag<PIPE_MTE1, PIPE_M>(0);
    if (i == 0) {
      TMATMUL(c_l0, a_l0, b_l0);
    } else {
      TMATMUL_ACC(c_l0, c_l0, a_l0, b_l0);
    }
    // Complete the multiply before reusing L0A and L1A next iteration.
    pipe_barrier(PIPE_ALL);
  }
  kernel_utils::SetWaitFlag<PIPE_M, PIPE_FIX>(0);
  CGlobal c_global(partials + id * S * kColumns);
  TSTORE(c_global, c_l0);
  pipe_barrier(PIPE_ALL);
}

// A separate launch on the same stream orders all Fixpipe stores before
// these reads. One AIV sums the first column, with no atomics or device-wide
// barriers. Only S*16 elements of UB are needed regardless of block count.
template <typename T, uint32_t S>
AICORE void CompleteCubeReduce(__gm__ T* partials, __gm__ T* output,
                               uint32_t blocks) {
  using InputGlobal = GlobalTensor<T, TileShape2D<T, S, kColumns, Layout::ND>,
                                   BaseShape2D<T, S, kColumns, Layout::ND>>;
  using OutputGlobal =
      GlobalTensor<T, Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 1>>;
  Tile<TileType::Vec, T, S, kColumns> tile;
  Tile<TileType::Vec, T, 1, 8, BLayout::RowMajor, 1, 1> result;
  TASSIGN(tile, 0);
  TASSIGN(result, S * kColumns * sizeof(T));
  T total = 0;
  for (uint32_t block = 0; block < blocks; ++block) {
    InputGlobal source(partials + block * S * kColumns);
    TLOAD(tile, source);
    // GetValue uses the scalar pipe, so an MTE2 -> V event is insufficient.
    kernel_utils::SetWaitFlag<PIPE_MTE2, PIPE_S>(0);
    for (uint32_t row = 0; row < S; ++row) {
      total += tile.GetValue(row * kColumns);
    }
    kernel_utils::SetWaitFlag<PIPE_S, PIPE_MTE2>(0);
  }
  result.SetValue(0, total);
  kernel_utils::SetWaitFlag<PIPE_S, PIPE_MTE3>(0);
  OutputGlobal destination(output);
  TSTORE(destination, result);
  pipe_barrier(PIPE_ALL);
}
}  // namespace

extern "C" __global__ AICORE void cube_reduce_fp16(__gm__ half* input,
                                                   __gm__ half* ones,
                                                   __gm__ float* partials,
                                                   uint32_t num_tiles,
                                                   uint32_t size) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
#if defined(__DAV_CUBE__)
  switch (size) {
    case 32:
      CubeReduce<half, float, 32>(input, ones, partials, num_tiles);
      break;
    case 64:
      CubeReduce<half, float, 64>(input, ones, partials, num_tiles);
      break;
    case 128:
      CubeReduce<half, float, 128>(input, ones, partials, num_tiles);
      break;
  }
#endif
}

extern "C" __global__ AICORE void complete_cube_reduce_float(
    __gm__ float* partials, __gm__ float* output, uint32_t blocks,
    uint32_t size) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if defined(__DAV_VEC__)
  switch (size) {
    case 32:
      CompleteCubeReduce<float, 32>(partials, output, blocks);
      break;
    case 64:
      CompleteCubeReduce<float, 64>(partials, output, blocks);
      break;
    case 128:
      CompleteCubeReduce<float, 128>(partials, output, blocks);
      break;
  }
#endif
}

extern "C" void pto_launch_cube_reduce_fp16(uint32_t blockDim, void* stream,
                                            void* input, void* ones,
                                            void* partials, void* output,
                                            uint32_t num_tiles, uint32_t size) {
  cube_reduce_fp16<<<blockDim, nullptr, stream>>>(
      (__gm__ half*)input, (__gm__ half*)ones, (__gm__ float*)partials,
      num_tiles, size);
  complete_cube_reduce_float<<<1, nullptr, stream>>>(
      (__gm__ float*)partials, (__gm__ float*)output, blockDim, size);
}

extern "C" __global__ AICORE void cube_reduce_fp32(__gm__ float* input,
                                                   __gm__ float* ones,
                                                   __gm__ float* partials,
                                                   uint32_t num_tiles,
                                                   uint32_t size) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
#if defined(__DAV_CUBE__)
  switch (size) {
    case 32:
      CubeReduce<float, float, 32>(input, ones, partials, num_tiles);
      break;
    case 64:
      CubeReduce<float, float, 64>(input, ones, partials, num_tiles);
      break;
    case 128:
      CubeReduce<float, float, 128>(input, ones, partials, num_tiles);
      break;
  }
#endif
}

extern "C" void pto_launch_cube_reduce_fp32(uint32_t blockDim, void* stream,
                                            void* input, void* ones,
                                            void* partials, void* output,
                                            uint32_t num_tiles, uint32_t size) {
  cube_reduce_fp32<<<blockDim, nullptr, stream>>>(
      (__gm__ float*)input, (__gm__ float*)ones, (__gm__ float*)partials,
      num_tiles, size);
  complete_cube_reduce_float<<<1, nullptr, stream>>>(
      (__gm__ float*)partials, (__gm__ float*)output, blockDim, size);
}

extern "C" __global__ AICORE void cube_reduce_int8(__gm__ int8_t* input,
                                                   __gm__ int8_t* ones,
                                                   __gm__ int32_t* partials,
                                                   uint32_t num_tiles,
                                                   uint32_t size) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
#if defined(__DAV_CUBE__)
  switch (size) {
    case 32:
      CubeReduce<int8_t, int32_t, 32>(input, ones, partials, num_tiles);
      break;
    case 64:
      CubeReduce<int8_t, int32_t, 64>(input, ones, partials, num_tiles);
      break;
    case 128:
      CubeReduce<int8_t, int32_t, 128>(input, ones, partials, num_tiles);
      break;
  }
#endif
}

extern "C" __global__ AICORE void complete_cube_reduce_int8(
    __gm__ int32_t* partials, __gm__ int32_t* output, uint32_t blocks,
    uint32_t size) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if defined(__DAV_VEC__)
  switch (size) {
    case 32:
      CompleteCubeReduce<int32_t, 32>(partials, output, blocks);
      break;
    case 64:
      CompleteCubeReduce<int32_t, 64>(partials, output, blocks);
      break;
    case 128:
      CompleteCubeReduce<int32_t, 128>(partials, output, blocks);
      break;
  }
#endif
}

extern "C" void pto_launch_cube_reduce_int8(uint32_t blockDim, void* stream,
                                            void* input, void* ones,
                                            void* partials, void* output,
                                            uint32_t num_tiles, uint32_t size) {
  cube_reduce_int8<<<blockDim, nullptr, stream>>>(
      (__gm__ int8_t*)input, (__gm__ int8_t*)ones, (__gm__ int32_t*)partials,
      num_tiles, size);
  complete_cube_reduce_int8<<<1, nullptr, stream>>>(
      (__gm__ int32_t*)partials, (__gm__ int32_t*)output, blockDim, size);
}
