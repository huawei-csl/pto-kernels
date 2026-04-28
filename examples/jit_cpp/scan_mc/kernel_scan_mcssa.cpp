/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/

#include <acl/acl.h>
#include <runtime/rt.h>

#include <pto/pto-inst.hpp>

#include "inter_core_flag.hpp"
#include "kernel_utils.h"

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
template <typename InputT, typename OutputT, uint32_t tile_size>
AICORE void scanULOne(__gm__ InputT* x, __gm__ InputT* o, __gm__ InputT* u,
                      __gm__ InputT* l, __gm__ OutputT* s, uint32_t scan_size) {
#if (__CHECK_FEATURE_AT_PRECOMPILE) || \
    (__CCE_AICORE__ == 220 && defined(__DAV_C220_CUBE__))

  // Type definitions for different memory levels
  // GM
  using Shape = pto::Shape<1, 1, 1, tile_size, tile_size>;
  using Stride = pto::Stride<1, 1, 1, tile_size, 1>;
  using GlobalDataIn = pto::GlobalTensor<InputT, Shape, Stride, Layout::ND>;
  using GlobalDataOut = pto::GlobalTensor<OutputT, Shape, Stride, Layout::ND>;

  const uint32_t elePerTile = tile_size * tile_size;

  // L1
  using TileL1In =
      Tile<TileType::Mat, InputT, tile_size, tile_size, BLayout::ColMajor,
           tile_size, tile_size, SLayout::RowMajor, 512>;
  using TileL1Out =
      Tile<TileType::Mat, OutputT, tile_size, tile_size, BLayout::ColMajor,
           tile_size, tile_size, SLayout::RowMajor, 512>;

  // L0
  using TileL0A = TileLeft<InputT, tile_size, tile_size>;
  using TileL0AOut = TileLeft<OutputT, tile_size, tile_size>;
  using TileL0B = TileRight<InputT, tile_size, tile_size>;
  using TileL0BOut = TileRight<OutputT, tile_size, tile_size>;
  using TileL0C = TileAcc<OutputT, tile_size, tile_size>;

  // GM Data
  uint32_t tile_shift = elePerTile * get_block_idx();
  GlobalDataIn xGlobal(x + tile_shift);
  GlobalDataIn oGlobal(o);
  GlobalDataIn uGlobal(u);
  GlobalDataIn lGlobal(l);
  GlobalDataOut sGlobal(s + tile_shift);
  // Reuse output buffer for intermediate result C1
  GlobalDataIn c1GM(reinterpret_cast<__gm__ InputT*>(s + tile_shift));

  // Load data from GM to L1
  TileL1In xL1;
  TileL1In oL1;
  TileL1In uL1;
  TileL1In c1L1;
  TileL1In lL1;
  TASSIGN(xL1, 0x0);
  const uint32_t tile_l1_in_byte_size = elePerTile * sizeof(InputT);
  const uint32_t tile_l1_out_byte_size = elePerTile * sizeof(OutputT);
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

  // Wait for matmul to complete before storing result back to GM
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

  TSTORE(sGlobal, sL0);
#endif
}

/**
 * @brief Naive block level scan implementation
 *
 * @tparam OutputT Output data type
 * @tparam tile_size Size of the square tile, should be the same as the one used
 *in scanULOne
 *
 * @param s Input and output matrix in GM. Should be the same buffer used for C1
 *in scanULOne
 * @param scan_size Total number of elements to scan, should be the same as the
 *one used in runKernelScanMCSSA
 * @param scan_core_buf Buffer in GM for storing intermediate scan results from
 *each tile, used for carry propagation between tiles
 *
 **/
template <typename OutputT, uint32_t tile_size>
AICORE void singleVecBlockScan(__gm__ OutputT* s, uint32_t scan_size,
                               __gm__ OutputT* scan_core_buf) {
#if (__CHECK_FEATURE_AT_PRECOMPILE) || \
    (__CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__))

  // Vec unit code path
  set_mask_norm();
  set_vector_mask(-1, -1);
  set_atomic_none();

  using Shape = pto::Shape<1, 1, 1, tile_size, tile_size>;
  using Stride = pto::Stride<1, 1, 1, tile_size, 1>;
  using GlobalDataOut = pto::GlobalTensor<OutputT, Shape, Stride, Layout::ND>;

  const uint32_t elePerTile = tile_size * tile_size;

  using TileDataOut = Tile<TileType::Vec, OutputT, 1, elePerTile,
                           BLayout::RowMajor, 1, elePerTile>;
  TileDataOut sVecTile;

  const uint32_t numberOfTiles = (scan_size + elePerTile - 1) / elePerTile;
  using TileScan = Tile<TileType::Vec, OutputT, 1, elePerTile,
                        BLayout::RowMajor, 1, elePerTile>;
  TileScan coreScanTile;

  GlobalDataOut coreScanGlobal(scan_core_buf);
  GlobalDataOut sGlobal(s);

  const uint32_t tile_ub_offset = 0x0;
  const uint32_t tile_byte_size = elePerTile * sizeof(OutputT);
  TASSIGN(sVecTile, tile_ub_offset);
  TASSIGN(coreScanTile, tile_ub_offset + tile_byte_size);

  if (get_block_idx() == 0 && get_subblockid() == 0) {
    // Only one vector core does the scan
    OutputT carry = 0;
    for (uint32_t it = 0; it < numberOfTiles; ++it) {
      uint32_t offset = it * elePerTile;
      TASSIGN(sGlobal, s + offset);

      // Load tile from GM to UB
      TLOAD(sVecTile, sGlobal);
      TLOAD(coreScanTile, coreScanGlobal);

      // Wait for load to complete
      set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

      // Store the carry-in to the first element of the scan tile
      coreScanTile.SetValue(it, carry);

      pipe_barrier(PIPE_ALL);
      // Extract the last element of the tile as the carry-out for the next tile
      carry += sVecTile.GetValue(elePerTile - 1);

      set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);

      set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);

      TSTORE(coreScanGlobal, coreScanTile);

      set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);

      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
  }

#endif
}

/**
 * @brief Adds the carry-in to the scan results from each tile
 *
 * @tparam OutputT Output data type
 * @tparam tile_size Size of the square tile
 *
 * @param s Input and output matrix in GM
 * @param scan_size Total number of elements to scan
 * @param scan_core_buf Buffer in GM for storing intermediate scan results from
 *each tile
 */
template <typename OutputT, uint32_t tile_size>
AICORE void addAllBlockScan(__gm__ OutputT* s, uint32_t scan_size,
                            __gm__ OutputT* scan_core_buf) {
#if (__CHECK_FEATURE_AT_PRECOMPILE) || \
    (__CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__))

  // Vec unit code path
  set_mask_norm();
  set_vector_mask(-1, -1);
  set_atomic_none();

  using Shape = pto::Shape<1, 1, 1, tile_size, tile_size>;
  using Stride = pto::Stride<1, 1, 1, tile_size, 1>;
  using GlobalDataOut = pto::GlobalTensor<OutputT, Shape, Stride, Layout::ND>;

  const uint32_t elePerTile = tile_size * tile_size;

  using TileDataOut = Tile<TileType::Vec, OutputT, 1, elePerTile,
                           BLayout::RowMajor, 1, elePerTile>;
  TileDataOut sVecTile;

  const uint32_t numberOfTiles = (scan_size + elePerTile - 1) / elePerTile;
  using TileScan = Tile<TileType::Vec, OutputT, 1, elePerTile,
                        BLayout::RowMajor, 1, elePerTile>;
  TileScan coreScanTile;

  GlobalDataOut coreScanGlobal(scan_core_buf);
  GlobalDataOut sGlobal(s);

  const uint32_t tile_ub_offset = 0x0;
  const uint32_t tile_byte_size = elePerTile * sizeof(OutputT);
  TASSIGN(sVecTile, tile_ub_offset);
  TASSIGN(coreScanTile, tile_ub_offset + tile_byte_size);

  // Cores, but only one vector each
  if (get_subblockid() == 0) {
    // Load the scan result from GM to UB
    TLOAD(coreScanTile, coreScanGlobal);

    // Wait for load to complete before doing addition
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

    // Extract the carry for this tile
    OutputT carry = coreScanTile.GetValue(get_block_idx());

    // Wait for the carry to be ready before doing addition
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);

    // LOAD the tile from GM to UB
    const uint32_t offset = get_block_idx() * elePerTile;
    TASSIGN(sGlobal, s + offset);
    TLOAD(sVecTile, sGlobal);

    // Wait for load to complete before doing addition
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TADDS(sVecTile, sVecTile, carry);

    // Wait for addition to complete before storing result back to GM
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(sGlobal, sVecTile);
  }

#endif
}

/**
 * @brief Kernel implementation for scan operation on a single cube.
 *
 * The implemetation follows the ScanMCSSA algorithm described in [1]
 *
 * [1] Parallel Scan on Ascend AI Accelerators
 * (https://arxiv.org/abs/2505.15112v1).
 *
 * @tparam InputT Input data type. Supports `fp16` or `fp32`
 * @tparam OutputT Output data type 'fp32`
 * @tparam tile_size Size of the square matrix
 *
 * @param x Input matrix in GM
 * @param o Ones matrix in GM
 * @param u Upper triangular matrix in GM
 * @param l Lower triangular matrix in GM
 * @param s Output matrix in GM, also used as intermediate buffer for C1
 */
template <typename InputT, typename OutputT, uint32_t tile_size>
AICORE void runKernelScanMCSSA(__gm__ InputT* x, __gm__ InputT* o,
                               __gm__ InputT* u, __gm__ InputT* l,
                               __gm__ OutputT* s, uint32_t scan_size,
                               __gm__ OutputT* scan_core_buf,
                               __gm__ uint8_t* ffts_addr) {
  set_ffts_base_addr((uint64_t)ffts_addr);

  scanULOne<InputT, OutputT, tile_size>(x, o, u, l, s, scan_size);

  SyncAllImpl<false>();

  singleVecBlockScan<OutputT, tile_size>(s, scan_size, scan_core_buf);

  SyncAllImpl<true>();

  addAllBlockScan<OutputT, tile_size>(s, scan_size, scan_core_buf);
}

template <typename T>
AICORE void run_scan_mcssa(__gm__ T* x, __gm__ T* o, __gm__ T* u, __gm__ T* l,
                           __gm__ float* s, uint32_t scan_size,
                           uint32_t tile_size, __gm__ float* scan_core_buf,
                           __gm__ uint8_t* ffts_addr) {
  static_assert(std::is_same_v<T, half> or std::is_same_v<T, float>,
                "scan_mcssa supports only fp16/fp32.");
  switch (tile_size) {
    case 16:
      runKernelScanMCSSA<T, float, 16>(x, o, u, l, s, scan_size, scan_core_buf,
                                       ffts_addr);
      break;
    case 32:
      runKernelScanMCSSA<T, float, 32>(x, o, u, l, s, scan_size, scan_core_buf,
                                       ffts_addr);
      break;
    case 64:
      runKernelScanMCSSA<T, float, 64>(x, o, u, l, s, scan_size, scan_core_buf,
                                       ffts_addr);
      break;
    case 96:
      runKernelScanMCSSA<T, float, 96>(x, o, u, l, s, scan_size, scan_core_buf,
                                       ffts_addr);
      break;
    case 128:
      runKernelScanMCSSA<T, float, 128>(x, o, u, l, s, scan_size, scan_core_buf,
                                        ffts_addr);
      break;
  }
}

// extern "C" __global__ AICORE void scan_mcssa_fp16(
//     __gm__ void* x, __gm__ void* o, __gm__ void* u, __gm__ void* l,
//     __gm__ void* s, uint32_t scan_size, uint32_t tile_size,
//     __gm__ float* scan_core_buf, __gm__ uint8_t* ffts_addr) {
//   run_scan_mcssa((__gm__ half*)x, (__gm__ half*)o, (__gm__ half*)u,
//                  (__gm__ half*)l, (__gm__ float*)s, scan_size, tile_size,
//                  scan_core_buf, ffts_addr);
// }

extern "C" __global__ AICORE void scan_mcssa_fp32(
    __gm__ float* x, __gm__ float* o, __gm__ float* u, __gm__ float* l,
    __gm__ float* s, uint32_t scan_size, uint32_t tile_size,
    __gm__ float* scan_core_buf, __gm__ uint8_t* ffts_addr) {
  run_scan_mcssa(x, o, u, l, s, scan_size, tile_size, scan_core_buf, ffts_addr);
}

extern "C" void scan_fp32(uint32_t blockDim, void* stream, void* x, void* o,
                          void* u, void* l, void* s, uint32_t scan_size,
                          uint32_t tile_size) {
  void* ffts_addr;
  uint32_t ffts_len;
  rtGetC2cCtrlAddr((uint64_t*)&ffts_addr, &ffts_len);

  // Allocate buffer for inter-core scan
  void* scan_core_buf;
  const uint32_t ele_per_tile = tile_size * tile_size;
  const uint32_t buf_size = ele_per_tile * sizeof(float);
  aclrtMalloc(&scan_core_buf, buf_size,
              aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_FIRST);

  scan_mcssa_fp32<<<blockDim, nullptr, stream>>>(
      (float*)x, (float*)o, (float*)u, (float*)l, (float*)s, scan_size,
      tile_size, (float*)scan_core_buf, (uint8_t*)ffts_addr);
}
