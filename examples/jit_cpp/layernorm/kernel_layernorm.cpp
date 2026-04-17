/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/

#include <pto/pto-inst.hpp>

// clang-format off: so it does not get wrongfully flagged by linter
#define GM_ADDR __gm__ uint8_t*  // To avoid #include "kernel_operator.h"
// clang-format on

using namespace pto;

constexpr uint32_t MAX_HIDDEN = 262144;
constexpr uint32_t CHUNK_HIDDEN = 8192;
constexpr uint32_t OUTPUT_DB_CHUNK_HIDDEN = 4096;
constexpr uint32_t STATS_DB_CHUNK_HIDDEN = 12288;
constexpr uint32_t ANCHORED_STATS_MIN_HIDDEN = STATS_DB_CHUNK_HIDDEN;
constexpr uint32_t ROW_TILE = 24;
constexpr uint32_t UB_USABLE_BYTES = 192 * 1024;
constexpr uint32_t STAT_TILE_BYTES = 256;
constexpr unsigned UB_BASE = 0x00000;

namespace UbLayout {
// Stats, chunk, and output layouts intentionally overlay the same UB region.
// Only one phase is live at a time, so each phase starts from UB_BASE.
namespace Stats {
constexpr unsigned X_HALF_BASE = UB_BASE;
constexpr unsigned X_HALF_STRIDE = STATS_DB_CHUNK_HIDDEN * sizeof(half);
constexpr unsigned X_FLOAT_BASE = X_HALF_BASE + 2 * X_HALF_STRIDE;
constexpr unsigned REDUCE_TMP_BASE = X_FLOAT_BASE + STATS_DB_CHUNK_HIDDEN * sizeof(float);
constexpr unsigned PHASE_END = REDUCE_TMP_BASE + STATS_DB_CHUNK_HIDDEN * sizeof(float);
}  // namespace Stats

namespace Chunk {
constexpr unsigned X_HALF_BASE = UB_BASE;
constexpr unsigned X_FLOAT_BASE = X_HALF_BASE + CHUNK_HIDDEN * sizeof(half);
}  // namespace Chunk

namespace OutputDb {
constexpr unsigned X_HALF_BASE = UB_BASE;
constexpr unsigned X_HALF_STRIDE = OUTPUT_DB_CHUNK_HIDDEN * sizeof(half);
constexpr unsigned X_FLOAT_STRIDE = OUTPUT_DB_CHUNK_HIDDEN * sizeof(float);
constexpr unsigned Y_HALF_BASE = X_HALF_BASE + 2 * X_HALF_STRIDE;
constexpr unsigned Y_HALF_STRIDE = OUTPUT_DB_CHUNK_HIDDEN * sizeof(half);
constexpr unsigned GAMMA_HALF_BASE = Y_HALF_BASE + 2 * Y_HALF_STRIDE;
constexpr unsigned BETA_HALF_BASE = GAMMA_HALF_BASE + OUTPUT_DB_CHUNK_HIDDEN * sizeof(half);
constexpr unsigned X_FLOAT_BASE = BETA_HALF_BASE + OUTPUT_DB_CHUNK_HIDDEN * sizeof(half);
constexpr unsigned GAMMA_FLOAT_BASE = X_FLOAT_BASE + 2 * X_FLOAT_STRIDE;
constexpr unsigned BETA_FLOAT_BASE = GAMMA_FLOAT_BASE + OUTPUT_DB_CHUNK_HIDDEN * sizeof(float);
constexpr unsigned PHASE_END = BETA_FLOAT_BASE + OUTPUT_DB_CHUNK_HIDDEN * sizeof(float);
}  // namespace OutputDb

namespace MediumHidden {
constexpr unsigned X_HALF_BASE = UB_BASE;
constexpr unsigned X_HALF_STRIDE = CHUNK_HIDDEN * sizeof(half);
constexpr unsigned X_FLOAT_BASE = X_HALF_BASE + 2 * X_HALF_STRIDE;
constexpr unsigned GAMMA_FLOAT_BASE = X_FLOAT_BASE + CHUNK_HIDDEN * sizeof(float);
constexpr unsigned BETA_FLOAT_BASE = GAMMA_FLOAT_BASE + CHUNK_HIDDEN * sizeof(float);
constexpr unsigned REDUCE_TMP_BASE = BETA_FLOAT_BASE + CHUNK_HIDDEN * sizeof(float);
constexpr unsigned Y_HALF_BASE = REDUCE_TMP_BASE + CHUNK_HIDDEN * sizeof(float);
constexpr unsigned SUM_BASE = Y_HALF_BASE + CHUNK_HIDDEN * sizeof(half);
constexpr unsigned MEAN_BASE = SUM_BASE + STAT_TILE_BYTES;
constexpr unsigned VAR_BASE = MEAN_BASE + STAT_TILE_BYTES;
constexpr unsigned INV_STD_BASE = VAR_BASE + STAT_TILE_BYTES;
constexpr unsigned CHUNK_STAT_BASE = INV_STD_BASE + STAT_TILE_BYTES;
constexpr unsigned PHASE_END = CHUNK_STAT_BASE + STAT_TILE_BYTES;
}  // namespace MediumHidden

namespace RowStats {
constexpr unsigned SUM_BASE = Stats::PHASE_END > OutputDb::PHASE_END
          ? Stats::PHASE_END
          : OutputDb::PHASE_END;
constexpr unsigned STRIDE = ROW_TILE * STAT_TILE_BYTES;
constexpr unsigned MEAN_BASE = SUM_BASE + STRIDE;
constexpr unsigned VAR_BASE = MEAN_BASE + STRIDE;
constexpr unsigned INV_STD_BASE = VAR_BASE + STRIDE;
constexpr unsigned CHUNK_STAT_BASE = INV_STD_BASE + STRIDE;
constexpr unsigned LAYOUT_END = CHUNK_STAT_BASE + STAT_TILE_BYTES;
}  // namespace RowStats
}  // namespace UbLayout

static_assert(UbLayout::RowStats::LAYOUT_END <= UB_USABLE_BYTES,
              "LayerNorm chunk UB layout exceeds usable UB.");
static_assert(UbLayout::MediumHidden::PHASE_END <= UB_USABLE_BYTES,
              "LayerNorm medium-hidden UB layout exceeds usable UB.");

#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)

using StrideDim5 = pto::Stride<1, 1, 1, 1, 1>;
template <uint32_t Columns>
using Shape1D = pto::Shape<1, 1, 1, 1, Columns>;
template <typename T, uint32_t Columns>
using Global1D = pto::GlobalTensor<T, Shape1D<Columns>, StrideDim5>;
template <typename T, uint32_t Columns>
using VecRowTile =
    Tile<TileType::Vec, T, 1, Columns, BLayout::RowMajor, -1, -1>;
template <typename T>
using ScalarTile = VecRowTile<T, 16>;
using StatTileColMajor =
    Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1>;
using StatTileRowMajor = VecRowTile<float, 8>;

template <typename T, uint32_t ChunkHidden>
AICORE void issueStatsXLoad(__gm__ T *x, uint32_t gm_offset,
                            uint32_t cur_hidden, unsigned x_half_base,
                            event_t ev) {
#if defined(__DAV_VEC__)
  VecRowTile<T, ChunkHidden> xChunkHalf(1, cur_hidden);
  Global1D<T, ChunkHidden> xChunkGlobal(x + gm_offset);
  TASSIGN(xChunkHalf, x_half_base);
  TASSIGN(xChunkGlobal, (x + gm_offset));

  wait_flag(PIPE_V, PIPE_MTE2, ev);
  TLOAD(xChunkHalf, xChunkGlobal);
  set_flag(PIPE_MTE2, PIPE_V, ev);
#endif
}

template <typename T>
AICORE void issueLayerNormOutputXLoad(__gm__ T *x, uint32_t gm_offset,
                                      uint32_t cur_hidden, unsigned x_half_base,
                                      event_t ev) {
#if defined(__DAV_VEC__)
  VecRowTile<T, OUTPUT_DB_CHUNK_HIDDEN> xChunkHalf(1, cur_hidden);
  Global1D<T, OUTPUT_DB_CHUNK_HIDDEN> xChunkGlobal(x + gm_offset);
  TASSIGN(xChunkHalf, x_half_base);
  TASSIGN(xChunkGlobal, (x + gm_offset));

  wait_flag(PIPE_V, PIPE_MTE2, ev);
  TLOAD(xChunkHalf, xChunkGlobal);
  set_flag(PIPE_MTE2, PIPE_V, ev);
#endif
}

template <typename Tile0, typename Global0>
AICORE void loadTilePairs(Tile0 &tile0, Global0 &global0) {
#if defined(__DAV_VEC__)
  TLOAD(tile0, global0);
#endif
}

template <typename Tile0, typename Global0, typename... Rest>
AICORE void loadTilePairs(Tile0 &tile0, Global0 &global0, Rest &...rest) {
#if defined(__DAV_VEC__)
  TLOAD(tile0, global0);
  if constexpr (sizeof...(Rest) > 0) {
    loadTilePairs(rest...);
  }
#endif
}

template <typename Tile0, typename Global0, typename... Rest>
AICORE void loadTilesSync(Tile0 &tile0, Global0 &global0, Rest &...rest) {
#if defined(__DAV_VEC__)
  static_assert(sizeof...(Rest) % 2 == 0,
                "loadTilesSync expects tile/global pairs.");
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  loadTilePairs(tile0, global0, rest...);
  pipe_barrier(PIPE_ALL);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
}

template <typename Global, typename Tile>
AICORE void storeTileAfterWait(Global &global, Tile &tile,
                               event_t ev = EVENT_ID0) {
#if defined(__DAV_VEC__)
  set_flag(PIPE_V, PIPE_MTE3, ev);
  wait_flag(PIPE_V, PIPE_MTE3, ev);
  TSTORE(global, tile);
  set_flag(PIPE_MTE3, PIPE_V, ev);
#endif
}

AICORE void initDbPipeFlags() {
#if defined(__DAV_VEC__)
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
#endif
}

AICORE void drainDbPipeFlags() {
#if defined(__DAV_VEC__)
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
#endif
}

template <typename AccTile, typename ValueTile>
AICORE void addOrCopyFirst(AccTile &acc, ValueTile &value, bool first) {
#if defined(__DAV_VEC__)
  if (first) {
    TMULS(acc, value, 1.0f);
  } else {
    TADD(acc, acc, value);
  }
  pipe_barrier(PIPE_V);
#endif
}

template <typename StatTile>
AICORE void computeInvStd(StatTile &invStdRow, StatTile &varRow,
                          StatTile &tmpRow, float eps) {
#if defined(__DAV_VEC__)
  TADDS(varRow, varRow, eps);
  pipe_barrier(PIPE_V);
  TRSQRT(invStdRow, varRow);
  pipe_barrier(PIPE_V);

  TMUL(tmpRow, invStdRow, invStdRow);
  pipe_barrier(PIPE_V);
  TMUL(tmpRow, tmpRow, varRow);
  pipe_barrier(PIPE_V);
  TMULS(tmpRow, tmpRow, -0.5f);
  pipe_barrier(PIPE_V);
  TADDS(tmpRow, tmpRow, 1.5f);
  pipe_barrier(PIPE_V);
  TMUL(invStdRow, invStdRow, tmpRow);
  pipe_barrier(PIPE_V);

  TMUL(tmpRow, invStdRow, invStdRow);
  pipe_barrier(PIPE_V);
  TMUL(tmpRow, tmpRow, varRow);
  pipe_barrier(PIPE_V);
  TMULS(tmpRow, tmpRow, -0.5f);
  pipe_barrier(PIPE_V);
  TADDS(tmpRow, tmpRow, 1.5f);
  pipe_barrier(PIPE_V);
  TMUL(invStdRow, invStdRow, tmpRow);
  pipe_barrier(PIPE_V);
#endif
}

template <bool AddToMean, typename StatTile>
AICORE void finalizeLayerNormStats(StatTile &meanRow, StatTile &sumRow,
                                   StatTile &varRow, StatTile &invStdRow,
                                   float inv_hidden, float eps) {
#if defined(__DAV_VEC__)
  TMULS(sumRow, sumRow, inv_hidden);
  pipe_barrier(PIPE_V);
  TMULS(varRow, varRow, inv_hidden);
  pipe_barrier(PIPE_V);
  TMUL(invStdRow, sumRow, sumRow);
  pipe_barrier(PIPE_V);
  TSUB(varRow, varRow, invStdRow);
  pipe_barrier(PIPE_V);
  if constexpr (AddToMean) {
    TADD(meanRow, meanRow, sumRow);
  } else {
    TMULS(meanRow, sumRow, 1.0f);
  }
  pipe_barrier(PIPE_V);

  computeInvStd(invStdRow, varRow, sumRow, eps);
#endif
}

template <typename T, uint32_t ChunkHidden>
AICORE void accumulateCenteredChunkMoments(__gm__ T *x, uint32_t row_offset,
                                           uint32_t hidden,
                                           StatTileColMajor &meanCol,
                                           StatTileColMajor &chunkStatCol,
                                           StatTileRowMajor &sumRow,
                                           StatTileRowMajor &varRow,
                                           StatTileRowMajor &chunkStatRow) {
#if defined(__DAV_VEC__)
  using ChunkHalfTile = VecRowTile<T, ChunkHidden>;
  using ChunkFloatTile = VecRowTile<float, ChunkHidden>;
  constexpr bool kUseStatsDbLayout = ChunkHidden == STATS_DB_CHUNK_HIDDEN;
  constexpr unsigned kXHalfBase =
      kUseStatsDbLayout ? UbLayout::Stats::X_HALF_BASE : UbLayout::Chunk::X_HALF_BASE;
  constexpr unsigned kXHalfStride =
      kUseStatsDbLayout ? UbLayout::Stats::X_HALF_STRIDE : 0;
  constexpr unsigned kXFloatBase =
      kUseStatsDbLayout ? UbLayout::Stats::X_FLOAT_BASE : UbLayout::Chunk::X_FLOAT_BASE;

  if constexpr (kUseStatsDbLayout) {
    const uint32_t first_hidden = hidden < ChunkHidden ? hidden : ChunkHidden;
    issueStatsXLoad<T, ChunkHidden>(x, row_offset, first_hidden, kXHalfBase,
                                    EVENT_ID0);

    bool first_chunk = true;
    bool ping = true;
    for (uint32_t col = 0; col < hidden; col += ChunkHidden) {
      const uint32_t remain = hidden - col;
      const uint32_t cur_hidden = remain < ChunkHidden ? remain : ChunkHidden;

      const int8_t buf = ping ? 0 : 1;
      const event_t current_ev = ping ? (event_t)EVENT_ID0 : (event_t)EVENT_ID1;
      const event_t next_ev = ping ? (event_t)EVENT_ID1 : (event_t)EVENT_ID0;
      const unsigned x_half_base = kXHalfBase + buf * kXHalfStride;

      ChunkHalfTile xChunkHalf(1, cur_hidden);
      ChunkFloatTile xChunkFloat(1, cur_hidden);
      ChunkFloatTile reduceTmp(1, cur_hidden);
      TASSIGN(xChunkHalf, x_half_base);
      TASSIGN(xChunkFloat, kXFloatBase);
      TASSIGN(reduceTmp, UbLayout::Stats::REDUCE_TMP_BASE);

      wait_flag(PIPE_MTE2, PIPE_V, current_ev);

      const uint32_t next_col = col + ChunkHidden;
      if (next_col < hidden) {
        const uint32_t next_remain = hidden - next_col;
        const uint32_t next_hidden =
            next_remain < ChunkHidden ? next_remain : ChunkHidden;
        const unsigned next_x_half_base =
            ping ? (kXHalfBase + kXHalfStride) : kXHalfBase;
        issueStatsXLoad<T, ChunkHidden>(x, row_offset + next_col, next_hidden,
                                        next_x_half_base, next_ev);
      }

      TCVT(xChunkFloat, xChunkHalf, RoundMode::CAST_NONE);
      pipe_barrier(PIPE_V);
      TROWEXPANDSUB(xChunkFloat, xChunkFloat, meanCol);
      pipe_barrier(PIPE_V);

      TROWSUM(chunkStatCol, xChunkFloat, reduceTmp);
      pipe_barrier(PIPE_V);
      addOrCopyFirst(sumRow, chunkStatRow, first_chunk);

      TMUL(xChunkFloat, xChunkFloat, xChunkFloat);
      pipe_barrier(PIPE_V);
      TROWSUM(chunkStatCol, xChunkFloat, reduceTmp);
      pipe_barrier(PIPE_V);
      addOrCopyFirst(varRow, chunkStatRow, first_chunk);

      first_chunk = false;
      set_flag(PIPE_V, PIPE_MTE2, current_ev);
      ping = !ping;
    }
  } else {
    using ChunkGlobal = Global1D<T, ChunkHidden>;

    bool first_chunk = true;
    for (uint32_t col = 0; col < hidden; col += ChunkHidden) {
      const uint32_t remain = hidden - col;
      const uint32_t cur_hidden = remain < ChunkHidden ? remain : ChunkHidden;

      ChunkHalfTile xChunkHalf(1, cur_hidden);
      ChunkFloatTile xChunkFloat(1, cur_hidden);
      ChunkFloatTile reduceTmp(1, cur_hidden);
      TASSIGN(xChunkHalf, kXHalfBase);
      TASSIGN(xChunkFloat, kXFloatBase);
      TASSIGN(reduceTmp, UbLayout::Stats::REDUCE_TMP_BASE);

      ChunkGlobal xChunkGlobal(x + row_offset + col);
      TASSIGN(xChunkGlobal, (x + row_offset + col));

      loadTilesSync(xChunkHalf, xChunkGlobal);

      TCVT(xChunkFloat, xChunkHalf, RoundMode::CAST_NONE);
      pipe_barrier(PIPE_V);
      TROWEXPANDSUB(xChunkFloat, xChunkFloat, meanCol);
      pipe_barrier(PIPE_V);

      TROWSUM(chunkStatCol, xChunkFloat, reduceTmp);
      pipe_barrier(PIPE_V);
      addOrCopyFirst(sumRow, chunkStatRow, first_chunk);

      TMUL(xChunkFloat, xChunkFloat, xChunkFloat);
      pipe_barrier(PIPE_V);
      TROWSUM(chunkStatCol, xChunkFloat, reduceTmp);
      pipe_barrier(PIPE_V);
      addOrCopyFirst(varRow, chunkStatRow, first_chunk);

      first_chunk = false;
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    }
  }
#endif
}

template <typename T>
AICORE void computeLayerNormRowStats(__gm__ T *x, uint32_t row_offset,
                                     uint32_t hidden, uint32_t stat_offset,
                                     float eps, float inv_hidden) {
#if defined(__DAV_VEC__)
  using ScalarGlobal = Global1D<T, 1>;
  using ScalarHalfTile = ScalarTile<T>;

  StatTileColMajor meanCol(1, 1);
  StatTileColMajor chunkStatCol(1, 1);
  StatTileRowMajor sumRow(1, 1);
  StatTileRowMajor meanRow(1, 1);
  StatTileRowMajor varRow(1, 1);
  StatTileRowMajor invStdRow(1, 1);
  StatTileRowMajor chunkStatRow(1, 1);

  TASSIGN(meanCol, UbLayout::RowStats::MEAN_BASE + stat_offset);
  TASSIGN(chunkStatCol, UbLayout::RowStats::CHUNK_STAT_BASE);
  TASSIGN(sumRow, UbLayout::RowStats::SUM_BASE + stat_offset);
  TASSIGN(meanRow, UbLayout::RowStats::MEAN_BASE + stat_offset);
  TASSIGN(varRow, UbLayout::RowStats::VAR_BASE + stat_offset);
  TASSIGN(invStdRow, UbLayout::RowStats::INV_STD_BASE + stat_offset);
  TASSIGN(chunkStatRow, UbLayout::RowStats::CHUNK_STAT_BASE);

  ScalarHalfTile anchorHalf(1, 1);
  ScalarGlobal anchorGlobal(x + row_offset);
  TASSIGN(anchorHalf, UbLayout::Chunk::X_HALF_BASE);
  TASSIGN(anchorGlobal, (x + row_offset));

  loadTilesSync(anchorHalf, anchorGlobal);

  TCVT(meanRow, anchorHalf, RoundMode::CAST_NONE);
  pipe_barrier(PIPE_V);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

  if (hidden >= ANCHORED_STATS_MIN_HIDDEN) {
    accumulateCenteredChunkMoments<T, STATS_DB_CHUNK_HIDDEN>(
        x, row_offset, hidden, meanCol, chunkStatCol, sumRow, varRow,
        chunkStatRow);
  } else {
    accumulateCenteredChunkMoments<T, CHUNK_HIDDEN>(
        x, row_offset, hidden, meanCol, chunkStatCol, sumRow, varRow,
        chunkStatRow);
  }

  finalizeLayerNormStats<true>(meanRow, sumRow, varRow, invStdRow,
                               inv_hidden, eps);
#endif
}

AICORE inline bool useMediumHiddenFastPath(uint32_t rows, uint32_t hidden) {
  return rows >= 2048 && hidden <= CHUNK_HIDDEN;
}

template <typename T>
AICORE void runLayerNormMediumHidden(__gm__ T *x, __gm__ T *gamma,
                                     __gm__ T *beta, __gm__ T *y,
                                     uint32_t row_begin, uint32_t row_end,
                                     uint32_t hidden, float eps,
                                     float inv_hidden) {
#if defined(__DAV_VEC__)
  using FullGlobal = Global1D<T, CHUNK_HIDDEN>;
  using FullHalfTile = VecRowTile<T, CHUNK_HIDDEN>;
  using FullFloatTile = VecRowTile<float, CHUNK_HIDDEN>;

  FullHalfTile xHalf(1, hidden);
  FullHalfTile ioHalf(1, hidden);
  FullFloatTile xFloat(1, hidden);
  FullFloatTile gammaFloat(1, hidden);
  FullFloatTile betaFloat(1, hidden);
  FullFloatTile reduceTmp(1, hidden);
  StatTileColMajor meanCol(1, 1);
  StatTileColMajor invStdCol(1, 1);
  StatTileColMajor chunkStatCol(1, 1);
  StatTileRowMajor chunkStatRow(1, 1);
  StatTileRowMajor sumRow(1, 1);
  StatTileRowMajor meanRow(1, 1);
  StatTileRowMajor varRow(1, 1);
  StatTileRowMajor invStdRow(1, 1);

  TASSIGN(ioHalf, UbLayout::MediumHidden::Y_HALF_BASE);
  TASSIGN(xFloat, UbLayout::MediumHidden::X_FLOAT_BASE);
  TASSIGN(gammaFloat, UbLayout::MediumHidden::GAMMA_FLOAT_BASE);
  TASSIGN(betaFloat, UbLayout::MediumHidden::BETA_FLOAT_BASE);
  TASSIGN(reduceTmp, UbLayout::MediumHidden::REDUCE_TMP_BASE);
  TASSIGN(meanCol, UbLayout::MediumHidden::MEAN_BASE);
  TASSIGN(invStdCol, UbLayout::MediumHidden::INV_STD_BASE);
  TASSIGN(chunkStatCol, UbLayout::MediumHidden::CHUNK_STAT_BASE);
  TASSIGN(chunkStatRow, UbLayout::MediumHidden::CHUNK_STAT_BASE);
  TASSIGN(sumRow, UbLayout::MediumHidden::SUM_BASE);
  TASSIGN(meanRow, UbLayout::MediumHidden::MEAN_BASE);
  TASSIGN(varRow, UbLayout::MediumHidden::VAR_BASE);
  TASSIGN(invStdRow, UbLayout::MediumHidden::INV_STD_BASE);

  FullGlobal gammaGlobal(gamma);
  FullGlobal betaGlobal(beta);
  TASSIGN(gammaGlobal, (gamma));
  TASSIGN(betaGlobal, (beta));

  loadTilesSync(ioHalf, gammaGlobal);
  TCVT(gammaFloat, ioHalf, RoundMode::CAST_NONE);
  pipe_barrier(PIPE_V);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

  loadTilesSync(ioHalf, betaGlobal);
  TCVT(betaFloat, ioHalf, RoundMode::CAST_NONE);
  pipe_barrier(PIPE_V);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

  issueStatsXLoad<T, CHUNK_HIDDEN>(x, row_begin * hidden, hidden,
                                   UbLayout::MediumHidden::X_HALF_BASE,
                                   EVENT_ID0);

  bool ping = true;

  for (uint32_t row = row_begin; row < row_end; ++row) {
    const uint32_t row_offset = row * hidden;
    const int8_t buf = ping ? 0 : 1;
    const event_t current_ev = ping ? (event_t)EVENT_ID0 : (event_t)EVENT_ID1;
    const event_t next_ev = ping ? (event_t)EVENT_ID1 : (event_t)EVENT_ID0;
    const unsigned x_half_base =
        UbLayout::MediumHidden::X_HALF_BASE + buf * UbLayout::MediumHidden::X_HALF_STRIDE;

    FullGlobal yGlobal(y + row_offset);
    TASSIGN(yGlobal, (y + row_offset));

    TASSIGN(xHalf, x_half_base);

    wait_flag(PIPE_MTE2, PIPE_V, current_ev);

    const uint32_t next_row = row + 1;
    if (next_row < row_end) {
      const unsigned next_x_half_base = ping
          ? (UbLayout::MediumHidden::X_HALF_BASE + UbLayout::MediumHidden::X_HALF_STRIDE)
          : UbLayout::MediumHidden::X_HALF_BASE;
      issueStatsXLoad<T, CHUNK_HIDDEN>(x, next_row * hidden, hidden,
                                       next_x_half_base, next_ev);
    }

    TCVT(xFloat, xHalf, RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
    TROWSUM(chunkStatCol, xFloat, reduceTmp);
    pipe_barrier(PIPE_V);
    TMULS(sumRow, chunkStatRow, 1.0f);
    pipe_barrier(PIPE_V);

    TMUL(xFloat, xFloat, xFloat);
    pipe_barrier(PIPE_V);
    TROWSUM(chunkStatCol, xFloat, reduceTmp);
    pipe_barrier(PIPE_V);
    TMULS(varRow, chunkStatRow, 1.0f);
    pipe_barrier(PIPE_V);

    finalizeLayerNormStats<false>(meanRow, sumRow, varRow, invStdRow,
                    inv_hidden, eps);

    TCVT(xFloat, xHalf, RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
    TROWEXPANDSUB(xFloat, xFloat, meanCol);
    pipe_barrier(PIPE_V);
    TROWEXPANDMUL(xFloat, xFloat, invStdCol);
    pipe_barrier(PIPE_V);
    TMUL(xFloat, xFloat, gammaFloat);
    pipe_barrier(PIPE_V);
    TADD(xFloat, xFloat, betaFloat);
    pipe_barrier(PIPE_V);

    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TCVT(ioHalf, xFloat, RoundMode::CAST_RINT);
    pipe_barrier(PIPE_V);

    set_flag(PIPE_V, PIPE_MTE2, current_ev);
    storeTileAfterWait(yGlobal, ioHalf, EVENT_ID0);

    ping = !ping;
  }
#endif
}

template <typename T>
AICORE void runLayerNormOutputRowsDb(__gm__ T *x, __gm__ T *gamma,
                                     __gm__ T *beta, __gm__ T *y,
                                     uint32_t row_tile_begin,
                                     uint32_t row_tile_rows, uint32_t hidden) {
#if defined(__DAV_VEC__)
  using DbChunkGlobal = Global1D<T, OUTPUT_DB_CHUNK_HIDDEN>;
  using DbHalfTile = VecRowTile<T, OUTPUT_DB_CHUNK_HIDDEN>;
  using DbFloatTile = VecRowTile<float, OUTPUT_DB_CHUNK_HIDDEN>;

  for (uint32_t col = 0; col < hidden; col += OUTPUT_DB_CHUNK_HIDDEN) {
    const uint32_t remain = hidden - col;
    const uint32_t cur_hidden =
        remain < OUTPUT_DB_CHUNK_HIDDEN ? remain : OUTPUT_DB_CHUNK_HIDDEN;

    DbHalfTile gammaChunkHalf(1, cur_hidden);
    DbHalfTile betaChunkHalf(1, cur_hidden);
    DbFloatTile gammaChunkFloat(1, cur_hidden);
    DbFloatTile betaChunkFloat(1, cur_hidden);

    TASSIGN(gammaChunkHalf, UbLayout::OutputDb::GAMMA_HALF_BASE);
    TASSIGN(betaChunkHalf, UbLayout::OutputDb::BETA_HALF_BASE);
    TASSIGN(gammaChunkFloat, UbLayout::OutputDb::GAMMA_FLOAT_BASE);
    TASSIGN(betaChunkFloat, UbLayout::OutputDb::BETA_FLOAT_BASE);

    DbChunkGlobal gammaChunkGlobal(gamma + col);
    DbChunkGlobal betaChunkGlobal(beta + col);
    TASSIGN(gammaChunkGlobal, (gamma + col));
    TASSIGN(betaChunkGlobal, (beta + col));

    loadTilesSync(gammaChunkHalf, gammaChunkGlobal, betaChunkHalf,
                  betaChunkGlobal);

    TCVT(gammaChunkFloat, gammaChunkHalf, RoundMode::CAST_NONE);
    TCVT(betaChunkFloat, betaChunkHalf, RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

    const uint32_t first_row_offset = row_tile_begin * hidden;
    issueLayerNormOutputXLoad<T>(x, first_row_offset + col, cur_hidden,
                                 UbLayout::OutputDb::X_HALF_BASE, EVENT_ID0);

    bool ping = true;
    for (uint32_t row_slot = 0; row_slot < row_tile_rows; ++row_slot) {
      const uint32_t row = row_tile_begin + row_slot;
      const uint32_t row_offset = row * hidden;
      const uint32_t stat_offset = row_slot * STAT_TILE_BYTES;
      const int8_t buf = ping ? 0 : 1;
      const event_t current_ev = ping ? (event_t)EVENT_ID0 : (event_t)EVENT_ID1;
      const event_t next_ev = ping ? (event_t)EVENT_ID1 : (event_t)EVENT_ID0;
      const unsigned x_half_base =
          UbLayout::OutputDb::X_HALF_BASE + buf * UbLayout::OutputDb::X_HALF_STRIDE;
      const unsigned y_half_base =
          UbLayout::OutputDb::Y_HALF_BASE + buf * UbLayout::OutputDb::Y_HALF_STRIDE;
      const unsigned x_float_base =
          UbLayout::OutputDb::X_FLOAT_BASE + buf * UbLayout::OutputDb::X_FLOAT_STRIDE;

      StatTileColMajor meanCol(1, 1);
      StatTileColMajor invStdCol(1, 1);
      DbHalfTile xChunkHalf(1, cur_hidden);
      DbHalfTile yChunkHalf(1, cur_hidden);
      DbFloatTile xChunkFloat(1, cur_hidden);

      TASSIGN(meanCol, UbLayout::RowStats::MEAN_BASE + stat_offset);
      TASSIGN(invStdCol, UbLayout::RowStats::INV_STD_BASE + stat_offset);
      TASSIGN(xChunkHalf, x_half_base);
      TASSIGN(yChunkHalf, y_half_base);
      TASSIGN(xChunkFloat, x_float_base);

      DbChunkGlobal yChunkGlobal(y + row_offset + col);
      TASSIGN(yChunkGlobal, (y + row_offset + col));

      wait_flag(PIPE_MTE2, PIPE_V, current_ev);

      const uint32_t next_row_slot = row_slot + 1;
      if (next_row_slot < row_tile_rows) {
        const uint32_t next_row = row_tile_begin + next_row_slot;
        const uint32_t next_row_offset = next_row * hidden;
        const unsigned next_x_half_base =
            ping ? (UbLayout::OutputDb::X_HALF_BASE + UbLayout::OutputDb::X_HALF_STRIDE)
              : UbLayout::OutputDb::X_HALF_BASE;
        issueLayerNormOutputXLoad<T>(x, next_row_offset + col, cur_hidden,
                                     next_x_half_base, next_ev);
      }

      TCVT(xChunkFloat, xChunkHalf, RoundMode::CAST_NONE);
      pipe_barrier(PIPE_V);
      TROWEXPANDSUB(xChunkFloat, xChunkFloat, meanCol);
      pipe_barrier(PIPE_V);
      TROWEXPANDMUL(xChunkFloat, xChunkFloat, invStdCol);
      pipe_barrier(PIPE_V);
      TMUL(xChunkFloat, xChunkFloat, gammaChunkFloat);
      pipe_barrier(PIPE_V);
      TADD(xChunkFloat, xChunkFloat, betaChunkFloat);
      pipe_barrier(PIPE_V);

      wait_flag(PIPE_MTE3, PIPE_V, current_ev);
      TCVT(yChunkHalf, xChunkFloat, RoundMode::CAST_RINT);
      pipe_barrier(PIPE_V);

      storeTileAfterWait(yChunkGlobal, yChunkHalf, current_ev);
      set_flag(PIPE_V, PIPE_MTE2, current_ev);

      ping = !ping;
    }
  }
#endif
}


template <typename T>
AICORE void runTLayerNorm(__gm__ T *x, __gm__ T *gamma, __gm__ T *beta,
                          __gm__ T *y, uint32_t rows, uint32_t hidden,
                          float eps, float inv_hidden) {
#if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);

  if (rows == 0 || hidden == 0 || hidden > MAX_HIDDEN) {
    return;
  }

  const uint32_t num_workers = get_block_num() * get_subblockdim();
  const uint32_t worker_id =
      get_block_idx() * get_subblockdim() + get_subblockid();
  const uint32_t base_rows_per_worker = rows / num_workers;
  const uint32_t extra_rows = rows - base_rows_per_worker * num_workers;
  const uint32_t worker_rows =
      base_rows_per_worker + (worker_id < extra_rows ? 1 : 0);
  if (worker_rows == 0) {
    return;
  }
  const uint32_t row_begin = worker_id * base_rows_per_worker +
                             (worker_id < extra_rows ? worker_id : extra_rows);
  const uint32_t row_end = row_begin + worker_rows;
  const bool use_medium_hidden_fast_path = useMediumHiddenFastPath(rows, hidden);

  initDbPipeFlags();

  if (use_medium_hidden_fast_path) {
    runLayerNormMediumHidden<T>(x, gamma, beta, y, row_begin, row_end, hidden,
                                eps, inv_hidden);
    drainDbPipeFlags();
    return;
  }

  for (uint32_t row_tile_begin = row_begin; row_tile_begin < row_end;
       row_tile_begin += ROW_TILE) {
    uint32_t row_tile_end = row_tile_begin + ROW_TILE;
    if (row_tile_end > row_end) row_tile_end = row_end;
    const uint32_t row_tile_rows = row_tile_end - row_tile_begin;

    for (uint32_t row_slot = 0; row_slot < row_tile_rows; ++row_slot) {
      computeLayerNormRowStats<T>(x, (row_tile_begin + row_slot) * hidden,
                                  hidden, row_slot * STAT_TILE_BYTES, eps,
                                  inv_hidden);
    }

    runLayerNormOutputRowsDb<T>(x, gamma, beta, y, row_tile_begin,
                                row_tile_rows, hidden);
  }

  drainDbPipeFlags();
#endif
}

#endif

extern "C" __global__ AICORE void layernorm_fp16(GM_ADDR x, GM_ADDR gamma,
                                                 GM_ADDR beta, GM_ADDR y,
                                                 uint32_t rows, uint32_t hidden,
                                                 float eps, float inv_hidden) {
#if defined(__DAV_VEC__)
  runTLayerNorm<half>((__gm__ half *)x, (__gm__ half *)gamma,
                      (__gm__ half *)beta, (__gm__ half *)y, rows, hidden, eps,
                      inv_hidden);
#else
  (void)x;
  (void)gamma;
  (void)beta;
  (void)y;
  (void)rows;
  (void)hidden;
  (void)eps;
  (void)inv_hidden;
#endif
}

extern "C" void call_layernorm_kernel(uint32_t blockDim, void *stream,
                                      uint8_t *x, uint8_t *gamma, uint8_t *beta,
                                      uint8_t *y, uint32_t rows,
                                      uint32_t hidden, float eps,
                                      float inv_hidden) {
  layernorm_fp16<<<blockDim * 2, nullptr, stream>>>(x, gamma, beta, y, rows,
                                                    hidden, eps, inv_hidden);
}
