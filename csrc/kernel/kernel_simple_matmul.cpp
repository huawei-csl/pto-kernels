/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/
#if defined __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)

// Placeholder for VEC compilation (the real kernel is CUBE-only).
#define MEMORY_BASE
#include <pto/common/type.hpp>

extern "C" __global__ AICORE void simple_matmul_fp16(__gm__ void* a,
                                                     __gm__ void* b,
                                                     __gm__ void* c,
                                                     uint32_t matrix_size) {}

extern "C" __global__ AICORE void simple_matmul_fp32(__gm__ void* a,
                                                     __gm__ void* b,
                                                     __gm__ void* c,
                                                     uint32_t matrix_size) {}

#elif (__CHECK_FEATURE_AT_PRECOMPILE) || \
    (__CCE_AICORE__ == 220 && defined(__DAV_C220_CUBE__))

#define MEMORY_BASE

#include <pto/pto-inst.hpp>

#include "kernel_operator.h"

using namespace pto;

constexpr unsigned NUM_BLOCKS = 20;    // number of AICs
constexpr unsigned UB_SIZE = 0x30000;  // 192KB UB of A2A3

template <pipe_t SrcPipe, pipe_t DstPipe>
AICORE inline void SetFlag(uint32_t id) {
  set_flag(SrcPipe, DstPipe, static_cast<event_t>(id));
}
template <pipe_t SrcPipe, pipe_t DstPipe>
AICORE inline void WaitFlag(uint32_t id) {
  wait_flag(SrcPipe, DstPipe, static_cast<event_t>(id));
}

template <typename InputT, typename OutputT, uint32_t matrix_size>
AICORE void runKernelSimpleMatMul(__gm__ InputT* a, __gm__ InputT* b,
                                  __gm__ OutputT* c) {
  constexpr uint32_t tile_len = matrix_size * matrix_size;

  /* Global Memory / Tensors */
  using TensorShapeIn =
      TileShape2D<InputT, matrix_size, matrix_size, Layout::ND>;
  using TensorStridesIn =
      BaseShape2D<InputT, matrix_size, matrix_size, Layout::ND>;
  using GlobalTensorIn =
      GlobalTensor<InputT, TensorShapeIn, TensorStridesIn, Layout::ND>;

  using TensorShapeOut =
      TileShape2D<OutputT, matrix_size, matrix_size, Layout::ND>;
  using TensorStridesOut =
      BaseShape2D<OutputT, matrix_size, matrix_size, Layout::ND>;
  using GlobalTensorOut =
      GlobalTensor<OutputT, TensorShapeOut, TensorStridesOut, Layout::ND>;

  /* L1 Memory */
  using TileL1AB =
      Tile<TileType::Mat, InputT, matrix_size, matrix_size, BLayout::ColMajor,
           matrix_size, matrix_size, SLayout::RowMajor, 512>;

  /* L0 Memory */
  using TileL0A = TileLeft<InputT, matrix_size, matrix_size>;
  using TileL0B = TileRight<InputT, matrix_size, matrix_size>;
  using TileL0C = TileAcc<OutputT, matrix_size, matrix_size>;

  GlobalTensorIn a_global_in(a);
  GlobalTensorIn b_global_in(b);
  GlobalTensorOut c_global_out(c);
  TASSIGN(a_global_in, a);
  TASSIGN(b_global_in, b);
  TASSIGN(c_global_out, c);

  TileL1AB a_l1_tile;
  TileL1AB b_l1_tile;
  TASSIGN(a_l1_tile, 0x0);
  TASSIGN(b_l1_tile, 0x0 + tile_len * sizeof(InputT));

  TileL0A a_l0_tile;
  TileL0B b_l0_tile;
  TileL0C c_l0_tile;
  // L0A/L0B/L0C are distinct scratchpads
  TASSIGN(a_l0_tile, 0x0);
  TASSIGN(b_l0_tile, 0x0);
  TASSIGN(c_l0_tile, 0x0);

  // LOAD matrix A from GM -> L1 (MTE2)
  TLOAD(a_l1_tile, a_global_in);
  TLOAD(b_l1_tile, b_global_in);
  SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
  WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);

  // Copy A from L1 -> L0 (MTE1)
  // MatMul unit waits (using id:0) for MTE1 to load matrices into L0A/B
  TMOV(a_l0_tile, a_l1_tile);
  // Copy B from L1 -> L0B
  // MatMul unit waits (using id:1) for MTE1 to load matrices into L0A/B
  TMOV(b_l0_tile, b_l1_tile);
  SetFlag<PIPE_MTE1, PIPE_M>(0);   // MTE1 pipe sets flag for MM pipe
  WaitFlag<PIPE_MTE1, PIPE_M>(0);  // MM pipe waits for MTE1 pipe to set flag

  // MATMUL (M)
  TMATMUL(c_l0_tile, a_l0_tile, b_l0_tile);
  pipe_barrier(PIPE_ALL);
  SetFlag<PIPE_M, PIPE_FIX>(0);   // M pipe sets flag for FIX pipe
  WaitFlag<PIPE_M, PIPE_FIX>(0);  // FIX pipe waits for M pipe to set flag
  TSTORE(c_global_out, c_l0_tile);
}

template <typename T>
AICORE void run_simple_matmul(__gm__ T* a, __gm__ T* b, __gm__ float* c,
                              uint32_t matrix_size) {
  static_assert(std::is_same_v<T, half> or std::is_same_v<T, float>,
                "simple_matmul supports only fp16/fp32.");

  switch (matrix_size) {
    case 16:
      runKernelSimpleMatMul<T, float, 16>(a, b, c);
      break;
    case 32:
      runKernelSimpleMatMul<T, float, 32>(a, b, c);
      break;

    case 64:
      runKernelSimpleMatMul<T, float, 64>(a, b, c);
      break;

    case 96:
      runKernelSimpleMatMul<T, float, 96>(a, b, c);
      break;

    case 128:
      runKernelSimpleMatMul<T, float, 128>(a, b, c);
      break;
  }
}

extern "C" __global__ AICORE void simple_matmul_fp16(__gm__ void* a,
                                                     __gm__ void* b,
                                                     __gm__ void* c,
                                                     uint32_t matrix_size) {
  run_simple_matmul<half>((__gm__ half*)a, (__gm__ half*)b, (__gm__ float*)c,
                          matrix_size);
}

extern "C" __global__ AICORE void simple_matmul_fp32(__gm__ void* a,
                                                     __gm__ void* b,
                                                     __gm__ void* c,
                                                     uint32_t matrix_size) {
  run_simple_matmul<float>((__gm__ float*)a, (__gm__ float*)b, (__gm__ float*)c,
                           matrix_size);
}

#endif
