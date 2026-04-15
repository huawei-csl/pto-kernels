#include <pto/pto-inst.hpp>

#include "CustomTSync.hpp"
#include "runtime/rt.h"

using namespace pto;

#ifndef SCAN_TILE_SIZE
#define SCAN_TILE_SIZE 64
#endif

constexpr uint32_t TILE_SIZE = SCAN_TILE_SIZE;

template <typename InputT>
struct ScanAccType {};

template <>
struct ScanAccType<int8_t> {
  using type = int32_t;
};

template <>
struct ScanAccType<half> {
  using type = float;
};

template <>
struct ScanAccType<float> {
  using type = float;
};

template <>
struct ScanAccType<bfloat16_t> {
  using type = float;
};

template <typename InputT>
AICORE void run_scan_single_core_pto(
    __gm__ InputT *x, __gm__ typename ScanAccType<InputT>::type *y,
    __gm__ InputT *u, __gm__ uint8_t *ffts_addr, uint32_t total_len) {
  using AccT = typename ScanAccType<InputT>::type;

#if (__CHECK_FEATURE_AT_PRECOMPILE) || \
    (__CCE_AICORE__ == 220 && defined(__DAV_C220_CUBE__))

  // Cube code path

  set_ffts_base_addr((uint64_t)ffts_addr);
  set_padding(0);
  set_atomic_none();

  if (get_block_idx() != 0) return;  // Only process on a single core

  using TileL1 =
      Tile<TileType::Mat, InputT, TILE_SIZE, TILE_SIZE, BLayout::ColMajor,
           TILE_SIZE, TILE_SIZE, SLayout::RowMajor, 512>;
  TileL1 aTileL1;
  TileL1 uTileL1;
  TASSIGN(aTileL1, 0x0);
  TASSIGN(uTileL1, 0x0 + TILE_SIZE * TILE_SIZE * sizeof(InputT));

  using TensorShape = TileShape2D<InputT, TILE_SIZE, TILE_SIZE, Layout::ND>;
  using TensorStrides = BaseShape2D<InputT, TILE_SIZE, TILE_SIZE, Layout::ND>;
  using GlobalDataIn =
      GlobalTensor<InputT, TensorShape, TensorStrides, Layout::ND>;

  GlobalDataIn uGM(u);

  using NDValidShapeC = TileShape2D<AccT, TILE_SIZE, TILE_SIZE, Layout::ND>;
  using NDWholeShapeC = BaseShape2D<AccT, TILE_SIZE, TILE_SIZE, Layout::ND>;
  using GlobalDataOut =
      GlobalTensor<AccT, NDValidShapeC, NDWholeShapeC, Layout::ND>;

  using TileA = TileLeft<InputT, TILE_SIZE, TILE_SIZE>;
  using TileU = TileRight<InputT, TILE_SIZE, TILE_SIZE>;
  using TileC = TileAcc<AccT, TILE_SIZE, TILE_SIZE>;

  TileA aTile;
  TASSIGN(aTile, 0x0);
  TileU uTile;
  TASSIGN(uTile, 0x0);
  TileC cTile;
  TASSIGN(cTile, 0x0);  // L0C bound

  // Load U
  TLOAD(uTileL1, uGM);

  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

  TMOV(uTile, uTileL1);

  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

  for (uint32_t offset = 0; offset < total_len;
       offset += TILE_SIZE * TILE_SIZE) {
    GlobalDataIn gm_x_mat(x + offset);
    GlobalDataOut gm_out_mat(y + offset);

    // 1. Load A
    TLOAD(aTileL1, gm_x_mat);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);

    TMOV(aTile, aTileL1);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);

    // 2. Compute C = A @ U
    TMATMUL(cTile, aTile, uTile);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);

    // Store C to global memory
    TSTORE(gm_out_mat, cTile);

    set_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID2);

    // Sync with AIV: wait for GM to settle, then notify AIV.
    // pipe_barrier(PIPE_MTE3); // It was suggested that this is required, but
    // it doesn't seem to be the case.
    CustomTSync<0, CubeToVec>().record();
    // Wait for AIV to finish processing this chunk before advancing.
    CustomTSync<1, VecToCube>().wait();
  }

#elif (__CHECK_FEATURE_AT_PRECOMPILE) || \
    (__CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__))
  // AIV code path

  set_ffts_base_addr((uint64_t)ffts_addr);
  set_mask_norm();
  set_vector_mask(-1, -1);

  if (get_block_idx() != 0) return;

  uint32_t addr = 0;
  const uint32_t UB_R = addr;
  addr += TILE_SIZE * sizeof(AccT);

  using VectorRow = Tile<TileType::Vec, AccT, 1, TILE_SIZE>;
  VectorRow rowTile;
  TASSIGN(rowTile, UB_R);

  AccT running_sum = 0;

  using GlobalDataRow = GlobalTensor<AccT, Shape<1, 1, 1, TILE_SIZE, TILE_SIZE>,
                                     Stride<1, 1, 1, TILE_SIZE, 1>>;

  for (uint32_t offset = 0; offset < total_len;
       offset += TILE_SIZE * TILE_SIZE) {
    // Wait for AIC to finish generating this chunk.
    CustomTSync<0, CubeToVec>().wait();

    // 3. Vector phase: process row-by-row, only one vector core works, but both
    // sync
    if (get_subblockid() == 0) {
      for (uint32_t i = 0; i < TILE_SIZE; ++i) {
        GlobalDataRow gm_row(y + offset + i * TILE_SIZE);

        TLOAD(rowTile, gm_row);

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);

        TADDS(rowTile, rowTile, running_sum);

        // Sync before extracting value (simulating scalar fallback)
        set_flag(PIPE_V, PIPE_S, EVENT_ID2);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID2);
        running_sum = rowTile.GetValue(TILE_SIZE - 1);
        set_flag(PIPE_S, PIPE_V, EVENT_ID2);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID2);

        TSTORE(gm_row, rowTile);

        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      }

      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
    }
    // Notify AIC that chunk processing is fully completed.
    CustomTSync<1, VecToCube>().record();
  }
#endif
}

__global__ AICORE void kernel_scan_single_core(__gm__ void *x, __gm__ void *out,
                                               __gm__ void *u_s,
                                               __gm__ uint8_t *ffts_addr,
                                               uint32_t total_len) {
  run_scan_single_core_pto<float>((__gm__ float *)x, (__gm__ float *)out,
                                  (__gm__ float *)u_s, ffts_addr, total_len);
}

extern "C" void scan_fp32(uint32_t blockDim, void *stream, void *x, void *out,
                          void *u_s, const uint32_t total_len) {
  void *ffts_addr;
  uint32_t ffts_len;
  rtGetC2cCtrlAddr((uint64_t *)&ffts_addr, &ffts_len);

  kernel_scan_single_core<<<blockDim, nullptr, stream>>>(
      (float *)x, (float *)out, (float *)u_s, (__gm__ uint8_t *)ffts_addr,
      total_len);
}
