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
AICORE void CubeReduce(__gm__ InputT* input, __gm__ OutputT* partials,
                       uint32_t num_tiles) {
  using AGlobal = GlobalTensor<InputT, TileShape2D<InputT, S, S, Layout::ND>,
                               BaseShape2D<InputT, S, S, Layout::ND>>;
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
  // Build the all-ones operand on chip instead of copying it in from global
  // memory: on a Mat tile TEXPANDS lowers to create_cbuf_matrix, which fills
  // L1 directly. Every element is one, so the fractal layout of the tile is
  // irrelevant and the flat fill is already the matrix we want. Like TLOAD it
  // issues on MTE2, so the MTE2 -> MTE1 flag below still orders it before the
  // move into L0B.
  TEXPANDS(b_l1, static_cast<InputT>(1));
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
// these reads. One AIV folds every block into one running lane, with no
// atomics or device-wide barriers. UB use is independent of the block count.
template <typename T, uint32_t S>
AICORE void CompleteCubeReduce(__gm__ T* partials, __gm__ T* output,
                               uint32_t blocks) {
  using InputGlobal = GlobalTensor<T, TileShape2D<T, S, kColumns, Layout::ND>,
                                   BaseShape2D<T, S, kColumns, Layout::ND>>;
  using OutputGlobal =
      GlobalTensor<T, Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 1>>;
  using RowTile = Tile<TileType::Vec, T, 1, kColumns>;
  constexpr uint32_t kTileBytes = S * kColumns * sizeof(T);
  constexpr uint32_t kRowBytes = kColumns * sizeof(T);
  Tile<TileType::Vec, T, S, kColumns> tile;
  Tile<TileType::Vec, T, S, kColumns> reduce_tmp;
  RowTile block_sums;
  RowTile totals;
  Tile<TileType::Vec, T, 1, 8, BLayout::RowMajor, 1, 1> result;
  TASSIGN(tile, 0);
  TASSIGN(reduce_tmp, kTileBytes);
  TASSIGN(block_sums, 2 * kTileBytes);
  TASSIGN(totals, 2 * kTileBytes + kRowBytes);
  TASSIGN(result, 2 * kTileBytes + 2 * kRowBytes);
  TEXPANDS(totals, static_cast<T>(0));
  for (uint32_t block = 0; block < blocks; ++block) {
    InputGlobal source(partials + block * S * kColumns);
    TLOAD(tile, source);
    kernel_utils::SetWaitFlag<PIPE_MTE2, PIPE_V>(0);
    // Every one of the 16 columns holds the same row sums, so reducing the
    // block down its rows leaves this block's total in each lane. The pairwise
    // tree TCOLSUM runs over the rows is both faster and, in fp32, more
    // accurate than accumulating S values one at a time on the scalar pipe.
    TCOLSUM(block_sums, tile, reduce_tmp, /*isBinary=*/true);
    kernel_utils::PipeBarrierVec();
    TADD(totals, totals, block_sums);
    kernel_utils::PipeBarrierVec();
    kernel_utils::SetWaitFlag<PIPE_V, PIPE_MTE2>(0);
  }
  // GetValue uses the scalar pipe, so the vector accumulation must be visible
  // to it before the single remaining scalar read.
  kernel_utils::SetWaitFlag<PIPE_V, PIPE_S>(0);
  result.SetValue(0, totals.GetValue(0));
  kernel_utils::SetWaitFlag<PIPE_S, PIPE_MTE3>(0);
  OutputGlobal destination(output);
  TSTORE(destination, result);
  pipe_barrier(PIPE_ALL);
}
}  // namespace

extern "C" __global__ AICORE void cube_reduce_fp16(__gm__ half* input,
                                                   __gm__ float* partials,
                                                   uint32_t num_tiles,
                                                   uint32_t size) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
#if defined(__DAV_CUBE__)
  switch (size) {
    case 32:
      CubeReduce<half, float, 32>(input, partials, num_tiles);
      break;
    case 64:
      CubeReduce<half, float, 64>(input, partials, num_tiles);
      break;
    case 128:
      CubeReduce<half, float, 128>(input, partials, num_tiles);
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
                                            void* input, void* partials,
                                            void* output, uint32_t num_tiles,
                                            uint32_t size) {
  cube_reduce_fp16<<<blockDim, nullptr, stream>>>(
      (__gm__ half*)input, (__gm__ float*)partials, num_tiles, size);
  complete_cube_reduce_float<<<1, nullptr, stream>>>(
      (__gm__ float*)partials, (__gm__ float*)output, blockDim, size);
}

extern "C" __global__ AICORE void cube_reduce_fp32(__gm__ float* input,
                                                   __gm__ float* partials,
                                                   uint32_t num_tiles,
                                                   uint32_t size) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
#if defined(__DAV_CUBE__)
  switch (size) {
    case 32:
      CubeReduce<float, float, 32>(input, partials, num_tiles);
      break;
    case 64:
      CubeReduce<float, float, 64>(input, partials, num_tiles);
      break;
    case 128:
      CubeReduce<float, float, 128>(input, partials, num_tiles);
      break;
  }
#endif
}

extern "C" void pto_launch_cube_reduce_fp32(uint32_t blockDim, void* stream,
                                            void* input, void* partials,
                                            void* output, uint32_t num_tiles,
                                            uint32_t size) {
  cube_reduce_fp32<<<blockDim, nullptr, stream>>>(
      (__gm__ float*)input, (__gm__ float*)partials, num_tiles, size);
  complete_cube_reduce_float<<<1, nullptr, stream>>>(
      (__gm__ float*)partials, (__gm__ float*)output, blockDim, size);
}

extern "C" __global__ AICORE void cube_reduce_int8(__gm__ int8_t* input,
                                                   __gm__ int32_t* partials,
                                                   uint32_t num_tiles,
                                                   uint32_t size) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
#if defined(__DAV_CUBE__)
  switch (size) {
    case 32:
      CubeReduce<int8_t, int32_t, 32>(input, partials, num_tiles);
      break;
    case 64:
      CubeReduce<int8_t, int32_t, 64>(input, partials, num_tiles);
      break;
    case 128:
      CubeReduce<int8_t, int32_t, 128>(input, partials, num_tiles);
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
                                            void* input, void* partials,
                                            void* output, uint32_t num_tiles,
                                            uint32_t size) {
  cube_reduce_int8<<<blockDim, nullptr, stream>>>(
      (__gm__ int8_t*)input, (__gm__ int32_t*)partials, num_tiles, size);
  complete_cube_reduce_int8<<<1, nullptr, stream>>>(
      (__gm__ int32_t*)partials, (__gm__ int32_t*)output, blockDim, size);
}
