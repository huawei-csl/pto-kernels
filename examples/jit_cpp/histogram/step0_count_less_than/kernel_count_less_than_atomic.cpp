/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/

#include <pto/pto-inst.hpp>

#include "../kernel_utils.h"

using namespace pto;

/**
 * runTCountLessThan - Local count calculation with Atomic Addition to Global
 * Memory.
 */
template <typename T, unsigned TILE_SIZE>
AICORE void runTCountLessThan(__gm__ T *x, __gm__ int32_t *z,
                              const uint32_t total_length, const float pivot) {
#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  set_atomic_add();

  // --- Define Global Tensors ---
  using InputGlobalData = pto::GlobalTensor<T, pto::Shape<1, 1, 1, 1, DYNAMIC>,
                                            pto::Stride<1, 1, 1, 1, 1>>;
  using OutGlobalData = pto::GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1>,
                                          pto::Stride<1, 1, 1, 1, 1>>;

  // --- Work Distribution ---
  const uint32_t block_idx = get_block_idx();
  const uint32_t block_num = get_block_num();
  const uint32_t num_tiles_total =
      kernel_utils::CeilDiv(total_length, TILE_SIZE);
  const uint32_t num_tiles_per_core =
      kernel_utils::CeilDiv(num_tiles_total, block_num);
  const uint32_t start_idx = block_idx * num_tiles_per_core;
  const uint32_t end_idx = (start_idx + num_tiles_per_core > num_tiles_total)
                               ? num_tiles_total
                               : (start_idx + num_tiles_per_core);

  if (start_idx < end_idx) {
    // --- Define UB Tiles and Memory Layout ---
    uint32_t addr = 0;
    const uint32_t UB_X_ADDR = addr;
    addr += TILE_SIZE * sizeof(T);
    const uint32_t UB_CUR_MASK_ADDR = addr;
    addr += TILE_SIZE * sizeof(uint8_t);
    const uint32_t UB_ONES_ADDR = addr;
    addr += TILE_SIZE * sizeof(float);
    const uint32_t UB_ZEROS_ADDR = addr;
    addr += TILE_SIZE * sizeof(float);
    const uint32_t UB_TSEL_OUT_ADDR = addr;
    addr += TILE_SIZE * sizeof(float);
    const uint32_t UB_REDUCE_TMP_ADDR = addr;
    addr += TILE_SIZE * sizeof(float);
    const uint32_t UB_COUNT_ADDR = addr;
    addr += 8 * sizeof(float);
    const uint32_t UB_TOTAL_COUNT_ADDR = addr;
    addr += 8 * sizeof(float);
    const uint32_t UB_LOCAL_OUT_ADDR = addr;

    using InputTileData = Tile<TileType::Vec, T, 1, TILE_SIZE>;
    InputTileData x_tile;
    TASSIGN(x_tile, UB_X_ADDR);

    using MaskTileData = Tile<TileType::Vec, uint8_t, 1, TILE_SIZE>;
    MaskTileData current_mask;
    TASSIGN(current_mask, UB_CUR_MASK_ADDR);

    // Float conversion tiles
    using F32TileData = Tile<TileType::Vec, float, 1, TILE_SIZE>;
    F32TileData ones_tile;
    TASSIGN(ones_tile, UB_ONES_ADDR);
    TEXPANDS(ones_tile, 1.0f);

    F32TileData zeros_tile;
    TASSIGN(zeros_tile, UB_ZEROS_ADDR);
    TEXPANDS(zeros_tile, 0.0f);

    F32TileData tsel_out_tile;
    TASSIGN(tsel_out_tile, UB_TSEL_OUT_ADDR);

    F32TileData reduce_tmp;
    TASSIGN(reduce_tmp, UB_REDUCE_TMP_ADDR);

    using F32CountTile =
        Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 1>;
    F32CountTile count_f32_tile;
    TASSIGN(count_f32_tile, UB_COUNT_ADDR);

    F32CountTile total_count_f32_tile;
    TASSIGN(total_count_f32_tile, UB_TOTAL_COUNT_ADDR);
    TEXPANDS(total_count_f32_tile, 0.0f);

    using OutTile = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 1>;
    OutTile local_out;
    TASSIGN(local_out, UB_LOCAL_OUT_ADDR);
    TEXPANDS(local_out, (int32_t)0);

    // --- Main Calculation Loop ---
    for (uint32_t tile_idx = start_idx; tile_idx < end_idx; ++tile_idx) {
      const uint32_t offset = tile_idx * TILE_SIZE;
      InputGlobalData x_gm(x + offset, {static_cast<int>(total_length)});

      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      TLOAD(x_tile, x_gm);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

      // Generate packed bit-mask
      TCMPS(current_mask, x_tile, static_cast<T>(pivot), CmpMode::LT);
      // Select 1.0f or 0.0f based on the packed bit-mask
      TSEL(tsel_out_tile, current_mask, ones_tile, zeros_tile);

      TEXPANDS(count_f32_tile, 0.0f);
      TEXPANDS(reduce_tmp, 0.0f);
      TROWSUM(count_f32_tile, tsel_out_tile, reduce_tmp);

      TADD(total_count_f32_tile, total_count_f32_tile, count_f32_tile);
    }

    // Convert accumulated total into our UB local count
    TCVT(local_out, total_count_f32_tile, RoundMode::CAST_RINT);

    // --- Final Atomic Store to Global Memory ---
    OutGlobalData z_gm(z);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    // Doesn't do atomic adds
    TSTORE<OutTile, OutGlobalData, AtomicType::AtomicAdd>(z_gm, local_out);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }

#endif
}

__global__ AICORE void count_less_than(__gm__ void *x, __gm__ void *z,
                                       const uint32_t in_length,
                                       const float pivot) {
  constexpr unsigned TILE_SIZE = 512;
  runTCountLessThan<float, TILE_SIZE>((__gm__ float *)x, (__gm__ int32_t *)z,
                                      in_length, pivot);
}

extern "C" void count_less_than_fp32(uint32_t num_blocks, void *stream,
                                     uint8_t *x, uint8_t *z, uint32_t in_length,
                                     float pivot) {
  count_less_than<<<num_blocks, nullptr, stream>>>(x, z, in_length, pivot);
}
