/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/

#define GM_ADDR __gm__ uint8_t*  // To avoid #include "kernel_operator.h"
#include <pto/pto-inst.hpp>

using namespace pto;

constexpr unsigned UB_SIZE = 0x30000;  // 192KB UB of A2A3

/**
 * @brief Kernel implementation for scan operation on a single cube.
 *
 * The implemetation follows the ScanUL1 algorithm described in [1]
 *
 * [1] Parallel Scan on Ascend AI Accelerators
 * (https://arxiv.org/abs/2505.15112v1).
 *
 * @tparam InputT Input data type. Supports `fp16` or `fp32`
 * @tparam OutputT Output data type 'fp32`
 * @tparam matrix_size Size of the square matrix
 *
 * @param x Input matrix in GM
 * @param o Ones matrix in GM
 * @param u Upper triangular matrix in GM
 * @param l Lower triangular matrix in GM
 * @param s Output matrix in GM, also used as intermediate buffer for C1
 */
template <typename InputT, typename OutputT, uint32_t matrix_size>
AICORE void runKernelScanUl1(__gm__ InputT* x, __gm__ InputT* o,
                             __gm__ InputT* u, __gm__ InputT* l,
                             __gm__ OutputT* s) {
#if (__CHECK_FEATURE_AT_PRECOMPILE) || \
    (__CCE_AICORE__ == 220 && defined(__DAV_C220_CUBE__))

  // Type definitions for different memory levels
  // GM
  using Shape = pto::Shape<1, 1, 1, matrix_size, matrix_size>;
  using Stride = pto::Stride<1, 1, 1, matrix_size, 1>;
  using GlobalDataIn = pto::GlobalTensor<InputT, Shape, Stride, Layout::ND>;
  using GlobalDataOut = pto::GlobalTensor<OutputT, Shape, Stride, Layout::ND>;

  // L1
  using TileL1In =
      Tile<TileType::Mat, InputT, matrix_size, matrix_size, BLayout::ColMajor,
           matrix_size, matrix_size, SLayout::RowMajor, 512>;
  using TileL1Out =
      Tile<TileType::Mat, OutputT, matrix_size, matrix_size, BLayout::ColMajor,
           matrix_size, matrix_size, SLayout::RowMajor, 512>;

  // L0
  using TileL0A = TileLeft<InputT, matrix_size, matrix_size>;
  using TileL0AOut = TileLeft<OutputT, matrix_size, matrix_size>;
  using TileL0B = TileRight<InputT, matrix_size, matrix_size>;
  using TileL0BOut = TileRight<OutputT, matrix_size, matrix_size>;
  using TileL0C = TileAcc<OutputT, matrix_size, matrix_size>;

  // GM Data
  GlobalDataIn xGlobal(x);
  GlobalDataIn oGlobal(o);
  GlobalDataIn uGlobal(u);
  GlobalDataIn lGlobal(l);
  GlobalDataOut sGlobal(s);
  // Reuse output buffer for intermediate result C1
  GlobalDataIn c1GM(reinterpret_cast<__gm__ InputT*>(s));

  // Load data from GM to L1
  TileL1In xL1;
  TileL1In oL1;
  TileL1In uL1;
  TileL1In c1L1;
  TileL1In lL1;
  TASSIGN(xL1, 0x0);
  const uint32_t tile_l1_in_byte_size =
      matrix_size * matrix_size * sizeof(InputT);
  const uint32_t tile_l1_out_byte_size =
      matrix_size * matrix_size * sizeof(OutputT);
  TASSIGN(oL1, 0x0 + tile_l1_in_byte_size);
  TASSIGN(uL1, 0x0 + 2 * tile_l1_in_byte_size);
  TASSIGN(c1L1, 0x0 + 3 * tile_l1_in_byte_size);
  TASSIGN(lL1, 0x0 + 4 * tile_l1_in_byte_size);

  TLOAD(xL1, xGlobal);
  TLOAD(oL1, oGlobal);
  TLOAD(uL1, uGlobal);
  TLOAD(lL1, lGlobal);

  // Wait for load to complete before moving data to L0
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

  // Load data from L1 to L0
  TileL0A xL0;
  TileL0B oL0;
  TileL0C sL0;

  // L0A/L0B/L0C are distinct scratchpads
  TASSIGN(xL0, 0x0);
  TASSIGN(oL0, 0x0);
  TASSIGN(sL0, 0x0);

  TMOV(xL0, xL1);
  TMOV(oL0, oL1);

  // Cube unit waits for MTE1
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

  // In the paper notation C1 = As @ 1s
  // row-wise reduction for the A tile
  TMATMUL(sL0, xL0, oL0);

  // Wait for matmul to complete before storing result back to GM
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

  // Move C1 from L0C to L1
  // TMOV_FLOAT(c1L1, sL0);

  // Move C1 from L0C to GM, in the float case,
  // we cannot move to L1 directly
  // because of downcasting
  TSTORE(c1GM, sL0);

  // Wait for FP
  set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);

  // Load Us from L1 to L0
  TileL0B uL0;
  TASSIGN(uL0, 0x0);
  TMOV(uL0, uL1);

  // Cube unit waits for MTE1
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

  // Int the paper notation C2 = A @ U
  // row-wise inclusive scan for the A tile
  pipe_barrier(PIPE_M);
  TMATMUL(sL0, xL0, uL0);

  // // >>>> DEBUG
  // //For debugging: store C2 to GM
  // set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  // wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  // TSTORE(c1GM, sL0);
  // // <<<< DEBUG

  // Wait for store to complete before loading C1
  set_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID0);

  // Load C1 from GM to L1
  TLOAD(c1L1, c1GM);

  // Wait for load to be complete before moving C1 to L0
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);

  // Wait for matmul to complet before loading Ls and C1
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);

  // Load Ls from L1 to L0
  TileL0A lL0;
  TASSIGN(lL0, 0x0);
  TMOV(lL0, lL1);

  // Load C1 from L1 to L0
  TileL0B c1L0;
  TASSIGN(c1L0, 0x0);
  TMOV(c1L0, c1L1);

  // Wait for load to be complete
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);

  // In the paper notation C2 += Ls @ C1
  pipe_barrier(PIPE_M);
  TMATMUL_ACC(sL0, sL0, lL0, c1L0);

  // // >>>> DEBUG
  // TMATMUL(sL0, lL0, c1L0);
  // // For debugging: store L @ C1 to GM
  // set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  // wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  // TSTORE(sGlobal, sL0);
  // // <<<< DEBUG

  // Wait for matmul to complete before storing result back to GM
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

  TSTORE(sGlobal, sL0);

#else
// Nothing to do on VEC
#endif
}

template <typename T>
AICORE void run_scan_ul1(__gm__ T* x, __gm__ T* o, __gm__ T* u, __gm__ T* l,
                         __gm__ float* s, uint32_t matrix_size) {
  static_assert(std::is_same_v<T, half> or std::is_same_v<T, float>,
                "scan_ul1 supports only fp16/fp32.");

  switch (matrix_size) {
    case 16:
      runKernelScanUl1<T, float, 16>(x, o, u, l, s);
      break;
    case 32:
      runKernelScanUl1<T, float, 32>(x, o, u, l, s);
      break;
    case 64:
      runKernelScanUl1<T, float, 64>(x, o, u, l, s);
      break;
    case 96:
      runKernelScanUl1<T, float, 96>(x, o, u, l, s);
      break;
    case 128:
      runKernelScanUl1<T, float, 128>(x, o, u, l, s);
      break;
  }
}

extern "C" __global__ AICORE void scan_ul1_fp16(__gm__ void* x, __gm__ void* o,
                                                __gm__ void* u, __gm__ void* l,
                                                __gm__ void* s,
                                                uint32_t matrix_size) {
  run_scan_ul1((__gm__ half*)x, (__gm__ half*)o, (__gm__ half*)u,
               (__gm__ half*)l, (__gm__ float*)s, matrix_size);
}

extern "C" __global__ AICORE void scan_ul1_fp32(__gm__ void* x, __gm__ void* o,
                                                __gm__ void* u,

                                                __gm__ void* l, __gm__ void* s,
                                                uint32_t matrix_size) {
  run_scan_ul1((__gm__ float*)x, (__gm__ float*)o, (__gm__ float*)u,
               (__gm__ float*)l, (__gm__ float*)s, matrix_size);
}
