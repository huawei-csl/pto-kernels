#include <pto/pto-inst.hpp>

using namespace pto;

#define DIV_ROUNDUP(x, y) (((x) + (y) - 1) / (y))
#define ALIGN_UP(x, y) (DIV_ROUNDUP((x), (y)) * (y))

constexpr uint32_t UB_SLOT_BYTES = (192 * 1024) / 6;
constexpr uint32_t X0_BUFFER_BYTES = UB_SLOT_BYTES;
constexpr uint32_t X1_BUFFER_BYTES = UB_SLOT_BYTES;
constexpr uint32_t Y_BUFFER_BYTES = UB_SLOT_BYTES;
constexpr uint32_t INPUT_ELEMENTS_PER_TILE = (64 * 1024) / sizeof(half);
constexpr uint32_t ELEMENTS_PER_TILE = Y_BUFFER_BYTES / sizeof(half);
constexpr uint32_t UB_USABLE_BYTES = 192 * 1024;
constexpr uint32_t TILE_ALIGNMENT = 16;
constexpr uint32_t TARGET_2D_ACTIVE_TILES_DIVISOR = 2;
constexpr uint32_t MIN_LONG_ROW_TILE_ELEMENTS = 1024;
constexpr uint32_t MAX_LONG_ROW_TILES_PER_ROW = 16;

constexpr unsigned X0_PING = 0x00000;
constexpr unsigned X1_PING = X0_PING + X0_BUFFER_BYTES;
constexpr unsigned Y_PING = X1_PING + X1_BUFFER_BYTES;
constexpr unsigned X0_PONG = Y_PING + Y_BUFFER_BYTES;
constexpr unsigned X1_PONG = X0_PONG + X0_BUFFER_BYTES;
constexpr unsigned Y_PONG = X1_PONG + X1_BUFFER_BYTES;

static_assert(UB_SLOT_BYTES * 6 == UB_USABLE_BYTES,
              "SwiGLU UB slots must fully pack the usable UB budget.");
static_assert(ELEMENTS_PER_TILE <= INPUT_ELEMENTS_PER_TILE / 2,
              "SwiGLU tile size exceeds kernel max output tile.");
static_assert(Y_PONG + Y_BUFFER_BYTES <= UB_USABLE_BYTES,
              "SwiGLU UB layout exceeds usable UB.");

namespace {

struct Tile1DWork {
  uint32_t sample_index;
  uint32_t gm_offset;
  uint32_t elements;
};

struct Tile2DConfig {
  uint32_t row_tile_len;
  uint32_t col_tile_len;
  uint32_t total_tiles;
  uint32_t area;
  bool meets_target;
};

struct Tile2DWork {
  uint32_t row_offset;
  uint32_t col_offset;
  uint32_t row_count;
  uint32_t col_count;
};

AICORE inline uint32_t max2DRowsForCols(uint32_t col_tile_len) {
  return ELEMENTS_PER_TILE / col_tile_len;
}

AICORE inline Tile2DConfig makeTile2DConfig(uint32_t batch, uint32_t output_n,
                                            uint32_t num_cores,
                                            uint32_t col_tile_len) {
  const uint32_t max_rows = max2DRowsForCols(col_tile_len);
  const uint32_t col_tiles = DIV_ROUNDUP(output_n, col_tile_len);
  const uint32_t target_tiles =
      max(1U, num_cores / TARGET_2D_ACTIVE_TILES_DIVISOR);
  const uint32_t min_row_tiles = max(1U, DIV_ROUNDUP(target_tiles, col_tiles));

  uint32_t row_tile_len = min(batch, max_rows);
  if (min_row_tiles > 1) {
    const uint32_t capped = batch / min_row_tiles;
    row_tile_len = min(row_tile_len, max(1U, capped));
  }
  if (row_tile_len == 0) {
    row_tile_len = 1;
  }

  const uint32_t row_tiles = DIV_ROUNDUP(batch, row_tile_len);
  const uint32_t total_tiles = row_tiles * col_tiles;
  return Tile2DConfig{
      row_tile_len,
      col_tile_len,
      total_tiles,
      row_tile_len * col_tile_len,
      total_tiles >= target_tiles,
  };
}

AICORE inline bool preferTile2DConfig(const Tile2DConfig &cand,
                                      const Tile2DConfig &best) {
  if (cand.meets_target != best.meets_target) {
    return cand.meets_target;
  }
  if (cand.area != best.area) {
    return cand.area > best.area;
  }
  if (cand.meets_target) {
    if (cand.total_tiles != best.total_tiles) {
      return cand.total_tiles < best.total_tiles;
    }
  } else if (cand.total_tiles != best.total_tiles) {
    return cand.total_tiles > best.total_tiles;
  }
  if (cand.col_tile_len != best.col_tile_len) {
    return cand.col_tile_len > best.col_tile_len;
  }
  return cand.row_tile_len > best.row_tile_len;
}

AICORE inline Tile2DConfig chooseTile2DConfig(uint32_t batch, uint32_t output_n,
                                              uint32_t num_cores) {
  Tile2DConfig best = makeTile2DConfig(batch, output_n, num_cores, 128);

  if (output_n >= 256) {
    const Tile2DConfig cand = makeTile2DConfig(batch, output_n, num_cores, 256);
    if (preferTile2DConfig(cand, best)) {
      best = cand;
    }
  }
  if (output_n >= 512) {
    const Tile2DConfig cand = makeTile2DConfig(batch, output_n, num_cores, 512);
    if (preferTile2DConfig(cand, best)) {
      best = cand;
    }
  }
  if (output_n >= 1024) {
    const Tile2DConfig cand =
        makeTile2DConfig(batch, output_n, num_cores, 1024);
    if (preferTile2DConfig(cand, best)) {
      best = cand;
    }
  }
  if (output_n >= 2048) {
    const Tile2DConfig cand =
        makeTile2DConfig(batch, output_n, num_cores, 2048);
    if (preferTile2DConfig(cand, best)) {
      best = cand;
    }
  }
  if (output_n >= 4096) {
    const Tile2DConfig cand =
        makeTile2DConfig(batch, output_n, num_cores, 4096);
    if (preferTile2DConfig(cand, best)) {
      best = cand;
    }
  }
  if (output_n >= 8192) {
    const Tile2DConfig cand =
        makeTile2DConfig(batch, output_n, num_cores, 8192);
    if (preferTile2DConfig(cand, best)) {
      best = cand;
    }
  }
  if (output_n >= 16384) {
    const Tile2DConfig cand =
        makeTile2DConfig(batch, output_n, num_cores, 16384);
    if (preferTile2DConfig(cand, best)) {
      best = cand;
    }
  }
  return best;
}

AICORE inline Tile2DWork makeTile2DWork(uint32_t global_tile_idx,
                                        uint32_t row_tile_len,
                                        uint32_t col_tile_len, uint32_t batch,
                                        uint32_t output_n) {
  const uint32_t col_tiles = DIV_ROUNDUP(output_n, col_tile_len);
  const uint32_t row_tile_idx = global_tile_idx / col_tiles;
  const uint32_t col_tile_idx = global_tile_idx % col_tiles;
  const uint32_t row_offset = row_tile_idx * row_tile_len;
  const uint32_t col_offset = col_tile_idx * col_tile_len;
  return Tile2DWork{
      row_offset,
      col_offset,
      min(row_tile_len, batch - row_offset),
      min(col_tile_len, output_n - col_offset),
  };
}

AICORE inline uint32_t pick1DFallbackTileElements(uint32_t batch,
                                                  uint32_t output_n,
                                                  uint32_t num_cores) {
  if (output_n <= ELEMENTS_PER_TILE && batch >= num_cores) {
    return output_n;
  }

  uint32_t desired_tiles_per_row =
      (batch >= num_cores) ? 1U : DIV_ROUNDUP(num_cores, batch);
  if (desired_tiles_per_row > MAX_LONG_ROW_TILES_PER_ROW) {
    desired_tiles_per_row = MAX_LONG_ROW_TILES_PER_ROW;
  }

  uint32_t tile_elements = DIV_ROUNDUP(output_n, desired_tiles_per_row);
  if (tile_elements < MIN_LONG_ROW_TILE_ELEMENTS &&
      output_n > MIN_LONG_ROW_TILE_ELEMENTS) {
    tile_elements = MIN_LONG_ROW_TILE_ELEMENTS;
  }
  tile_elements = ALIGN_UP(tile_elements, TILE_ALIGNMENT);
  if (tile_elements > ELEMENTS_PER_TILE) {
    tile_elements = ELEMENTS_PER_TILE;
  }
  if (tile_elements > output_n) {
    tile_elements = output_n;
  }
  return tile_elements;
}

AICORE inline Tile1DWork makeTile1DWork(uint32_t global_tile_idx,
                                        uint32_t tiles_per_row,
                                        uint32_t tile_elements,
                                        uint32_t output_n) {
  const uint32_t sample_index = global_tile_idx / tiles_per_row;
  const uint32_t gm_offset = (global_tile_idx % tiles_per_row) * tile_elements;
  const uint32_t remaining = output_n - gm_offset;
  return Tile1DWork{
      sample_index,
      gm_offset,
      (remaining >= tile_elements) ? tile_elements : remaining,
  };
}

template <typename TileData, typename T>
AICORE inline void computeSwiGLUTile(TileData &x0Tile, TileData &x1Tile,
                                     TileData &yTile) {
  TMULS(yTile, x0Tile, (T)-1);
  pipe_barrier(PIPE_V);
  TEXP(yTile, yTile);
  pipe_barrier(PIPE_V);
  TADDS(yTile, yTile, (T)1);
  pipe_barrier(PIPE_V);
  TDIV(yTile, x0Tile, yTile);
  pipe_barrier(PIPE_V);
  TMUL(yTile, yTile, x1Tile);
  pipe_barrier(PIPE_V);
}

template <typename T>
AICORE void issueTile1DLoad(__gm__ T *x, uint32_t n, const Tile1DWork &tile,
                            unsigned x0_base, unsigned x1_base, event_t ev) {
  using ShapeDim5 = pto::Shape<1, 1, 1, 1, ELEMENTS_PER_TILE>;
  using StridDim5 = pto::Stride<1, 1, 1, 1, 1>;
  using GlobalData = pto::GlobalTensor<T, ShapeDim5, StridDim5>;
  using FullTile =
      Tile<TileType::Vec, T, 1, ELEMENTS_PER_TILE, BLayout::RowMajor, -1, -1>;

  FullTile x0Tile(1, tile.elements);
  FullTile x1Tile(1, tile.elements);
  TASSIGN(x0Tile, x0_base);
  TASSIGN(x1Tile, x1_base);

  GlobalData x0Global(x + tile.gm_offset);
  GlobalData x1Global(x + n + tile.gm_offset);
  TASSIGN(x0Global, (x + tile.gm_offset));
  TASSIGN(x1Global, (x + n + tile.gm_offset));

  wait_flag(PIPE_V, PIPE_MTE2, ev);
  wait_flag(PIPE_MTE3, PIPE_V, ev);
  TLOAD(x0Tile, x0Global);
  TLOAD(x1Tile, x1Global);
  set_flag(PIPE_MTE2, PIPE_V, ev);
}

template <typename T, uint32_t kTileRows, uint32_t kTileCols>
AICORE void issueTile2DLoad(__gm__ T *x, uint32_t input_n, uint32_t output_n,
                            const Tile2DWork &tile, unsigned x0_base,
                            unsigned x1_base, event_t ev) {
  using TileShapeND = TileShape2D<T, DYNAMIC, DYNAMIC, Layout::ND>;
  using DynStrideND = Stride<1, 1, 1, DYNAMIC, 1>;
  using GlobalData = GlobalTensor<T, TileShapeND, DynStrideND, Layout::ND>;
  using TileData = Tile<TileType::Vec, T, kTileRows, kTileCols, BLayout::RowMajor,
                        DYNAMIC, DYNAMIC>;

  TileData x0Tile(tile.row_count, tile.col_count);
  TileData x1Tile(tile.row_count, tile.col_count);
  TASSIGN(x0Tile, x0_base);
  TASSIGN(x1Tile, x1_base);

  const uint32_t input_offset = tile.row_offset * input_n + tile.col_offset;
  const TileShapeND shape(tile.row_count, tile.col_count);
  const DynStrideND stride(input_n);

  GlobalData x0Global(x + input_offset, shape, stride);
  GlobalData x1Global(x + input_offset + output_n, shape, stride);

  wait_flag(PIPE_V, PIPE_MTE2, ev);
  wait_flag(PIPE_MTE3, PIPE_V, ev);
  TLOAD(x0Tile, x0Global);
  TLOAD(x1Tile, x1Global);
  set_flag(PIPE_MTE2, PIPE_V, ev);
}

template <typename T, uint32_t kTileRows, uint32_t kTileCols>
AICORE void storeTile2D(__gm__ T *y, uint32_t output_n, const Tile2DWork &tile,
                        unsigned y_base, event_t ev) {
  using TileShapeND = TileShape2D<T, DYNAMIC, DYNAMIC, Layout::ND>;
  using DynStrideND = Stride<1, 1, 1, DYNAMIC, 1>;
  using GlobalData = GlobalTensor<T, TileShapeND, DynStrideND, Layout::ND>;
  using TileData = Tile<TileType::Vec, T, kTileRows, kTileCols, BLayout::RowMajor,
                        DYNAMIC, DYNAMIC>;

  TileData yTile(tile.row_count, tile.col_count);
  TASSIGN(yTile, y_base);

  const uint32_t output_offset = tile.row_offset * output_n + tile.col_offset;
  const TileShapeND shape(tile.row_count, tile.col_count);
  const DynStrideND stride(output_n);
  GlobalData yGlobal(y + output_offset, shape, stride);

  set_flag(PIPE_V, PIPE_MTE3, ev);
  wait_flag(PIPE_V, PIPE_MTE3, ev);
  TSTORE(yGlobal, yTile);
  set_flag(PIPE_MTE3, PIPE_V, ev);
  set_flag(PIPE_V, PIPE_MTE2, ev);
}

template <uint32_t kTileCols, typename T>
AICORE void runTSwiGLU2DTiled(__gm__ T *x, __gm__ T *y, uint32_t batch,
                              uint32_t input_n, uint32_t num_cores,
                              uint32_t vid, uint32_t row_tile_len) {
#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)
  constexpr uint32_t kTileRows = ELEMENTS_PER_TILE / kTileCols;
  static_assert(kTileRows * kTileCols == ELEMENTS_PER_TILE,
                "2D tile shape must match the UB vector tile capacity.");

  const uint32_t output_n = input_n >> 1;
  const uint32_t col_tiles = DIV_ROUNDUP(output_n, kTileCols);
  const uint32_t row_tiles = DIV_ROUNDUP(batch, row_tile_len);
  const uint32_t total_tiles = row_tiles * col_tiles;
  if (vid >= total_tiles) {
    return;
  }

  using TileData = Tile<TileType::Vec, T, kTileRows, kTileCols, BLayout::RowMajor,
                        DYNAMIC, DYNAMIC>;

  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

  uint32_t current_tile_idx = vid;
  Tile2DWork current_tile =
      makeTile2DWork(current_tile_idx, row_tile_len, kTileCols, batch, output_n);
  bool ping = true;
  issueTile2DLoad<T, kTileRows, kTileCols>(x, input_n, output_n, current_tile,
                                           X0_PING, X1_PING, (event_t)EVENT_ID0);

  while (true) {
    const event_t current_ev = ping ? (event_t)EVENT_ID0 : (event_t)EVENT_ID1;
    const unsigned current_x0_base = ping ? X0_PING : X0_PONG;
    const unsigned current_x1_base = ping ? X1_PING : X1_PONG;
    const unsigned current_y_base = ping ? Y_PING : Y_PONG;

    wait_flag(PIPE_MTE2, PIPE_V, current_ev);

    Tile2DWork next_tile{0, 0, 0, 0};
    const uint32_t next_tile_idx = current_tile_idx + num_cores;
    const bool has_next = next_tile_idx < total_tiles;
    if (has_next) {
      next_tile =
          makeTile2DWork(next_tile_idx, row_tile_len, kTileCols, batch, output_n);
      const event_t next_ev = ping ? (event_t)EVENT_ID1 : (event_t)EVENT_ID0;
      const unsigned next_x0_base = ping ? X0_PONG : X0_PING;
      const unsigned next_x1_base = ping ? X1_PONG : X1_PING;
      issueTile2DLoad<T, kTileRows, kTileCols>(x, input_n, output_n, next_tile,
                                               next_x0_base, next_x1_base,
                                               next_ev);
    }

    TileData x0Tile(current_tile.row_count, current_tile.col_count);
    TileData x1Tile(current_tile.row_count, current_tile.col_count);
    TileData yTile(current_tile.row_count, current_tile.col_count);
    TASSIGN(x0Tile, current_x0_base);
    TASSIGN(x1Tile, current_x1_base);
    TASSIGN(yTile, current_y_base);

    computeSwiGLUTile<TileData, T>(x0Tile, x1Tile, yTile);
    storeTile2D<T, kTileRows, kTileCols>(y, output_n, current_tile,
                                         current_y_base, current_ev);

    if (!has_next) {
      break;
    }

    current_tile = next_tile;
    current_tile_idx = next_tile_idx;
    ping = !ping;
  }

  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
#else
  (void)x;
  (void)y;
  (void)batch;
  (void)input_n;
  (void)num_cores;
  (void)vid;
  (void)row_tile_len;
#endif
}

template <typename T>
AICORE void runTSwiGLU2DMainTiled(__gm__ T *x, __gm__ T *y, uint32_t batch,
                                  uint32_t input_n, uint32_t num_cores,
                                  uint32_t vid) {
#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)
  const uint32_t output_n = input_n >> 1;
  const Tile2DConfig cfg = chooseTile2DConfig(batch, output_n, num_cores);

  switch (cfg.col_tile_len) {
    case 4096:
      runTSwiGLU2DTiled<4096, T>(x, y, batch, input_n, num_cores, vid,
                                 cfg.row_tile_len);
      break;
    case 8192:
      runTSwiGLU2DTiled<8192, T>(x, y, batch, input_n, num_cores, vid,
                                 cfg.row_tile_len);
      break;
    case 16384:
      runTSwiGLU2DTiled<16384, T>(x, y, batch, input_n, num_cores, vid,
                                  cfg.row_tile_len);
      break;
    case 2048:
      runTSwiGLU2DTiled<2048, T>(x, y, batch, input_n, num_cores, vid,
                                 cfg.row_tile_len);
      break;
    case 1024:
      runTSwiGLU2DTiled<1024, T>(x, y, batch, input_n, num_cores, vid,
                                 cfg.row_tile_len);
      break;
    case 512:
      runTSwiGLU2DTiled<512, T>(x, y, batch, input_n, num_cores, vid,
                                cfg.row_tile_len);
      break;
    case 256:
      runTSwiGLU2DTiled<256, T>(x, y, batch, input_n, num_cores, vid,
                                cfg.row_tile_len);
      break;
    default:
      runTSwiGLU2DTiled<128, T>(x, y, batch, input_n, num_cores, vid,
                                cfg.row_tile_len);
      break;
  }
#else
  (void)x;
  (void)y;
  (void)batch;
  (void)input_n;
  (void)num_cores;
  (void)vid;
#endif
}

template <typename T>
AICORE void runTSwiGLU1DFallbackTiled(__gm__ T *x, __gm__ T *y, uint32_t batch,
                                      uint32_t input_n, uint32_t num_cores,
                                      uint32_t vid) {
#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)
  const uint32_t output_n = input_n >> 1;
  const uint32_t tile_elements =
      pick1DFallbackTileElements(batch, output_n, num_cores);
  const uint32_t tiles_per_row = DIV_ROUNDUP(output_n, tile_elements);
  const uint32_t total_tiles = batch * tiles_per_row;
  if (vid >= total_tiles) {
    return;
  }

  using OutShapeDim5 = pto::Shape<1, 1, 1, 1, ELEMENTS_PER_TILE>;
  using StridDim5 = pto::Stride<1, 1, 1, 1, 1>;
  using GlobalData = pto::GlobalTensor<T, OutShapeDim5, StridDim5>;
  using FullTile =
      Tile<TileType::Vec, T, 1, ELEMENTS_PER_TILE, BLayout::RowMajor, -1, -1>;

  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

  uint32_t current_tile_idx = vid;
  Tile1DWork current_tile = makeTile1DWork(current_tile_idx, tiles_per_row,
                                           tile_elements, output_n);
  bool ping = true;

  __gm__ T *current_sample_x = x + current_tile.sample_index * input_n;
  issueTile1DLoad(current_sample_x, output_n, current_tile, X0_PING, X1_PING,
                  (event_t)EVENT_ID0);

  while (true) {
    const event_t current_ev = ping ? (event_t)EVENT_ID0 : (event_t)EVENT_ID1;
    const unsigned current_x0_base = ping ? X0_PING : X0_PONG;
    const unsigned current_x1_base = ping ? X1_PING : X1_PONG;
    const unsigned current_y_base = ping ? Y_PING : Y_PONG;

    wait_flag(PIPE_MTE2, PIPE_V, current_ev);

    Tile1DWork next_tile{0, 0, 0};
    const uint32_t next_tile_idx = current_tile_idx + num_cores;
    const bool has_next = next_tile_idx < total_tiles;
    if (has_next) {
      next_tile = makeTile1DWork(next_tile_idx, tiles_per_row, tile_elements,
                                 output_n);

      const event_t next_ev = ping ? (event_t)EVENT_ID1 : (event_t)EVENT_ID0;
      const unsigned next_x0_base = ping ? X0_PONG : X0_PING;
      const unsigned next_x1_base = ping ? X1_PONG : X1_PING;
      __gm__ T *next_sample_x = x + next_tile.sample_index * input_n;
      issueTile1DLoad(next_sample_x, output_n, next_tile, next_x0_base,
                      next_x1_base, next_ev);
    }

    FullTile x0Tile(1, current_tile.elements);
    FullTile x1Tile(1, current_tile.elements);
    FullTile yTile(1, current_tile.elements);
    TASSIGN(x0Tile, current_x0_base);
    TASSIGN(x1Tile, current_x1_base);
    TASSIGN(yTile, current_y_base);

    __gm__ T *current_sample_y = y + current_tile.sample_index * output_n;
    GlobalData yGlobal(current_sample_y + current_tile.gm_offset);
    TASSIGN(yGlobal, (current_sample_y + current_tile.gm_offset));

    computeSwiGLUTile<FullTile, T>(x0Tile, x1Tile, yTile);

    set_flag(PIPE_V, PIPE_MTE3, current_ev);
    wait_flag(PIPE_V, PIPE_MTE3, current_ev);
    TSTORE(yGlobal, yTile);
    set_flag(PIPE_MTE3, PIPE_V, current_ev);
    set_flag(PIPE_V, PIPE_MTE2, current_ev);

    if (!has_next) {
      break;
    }

    current_tile = next_tile;
    current_tile_idx = next_tile_idx;
    ping = !ping;
  }

  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
#else
  (void)x;
  (void)y;
  (void)batch;
  (void)input_n;
  (void)num_cores;
  (void)vid;
#endif
}

template <typename T>
AICORE void runTSwiGLU(__gm__ T *x, __gm__ T *y, uint32_t batch,
                       uint32_t input_n, uint32_t num_cores, uint32_t vid) {
#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);

  if (input_n == 0 || (input_n & 1U) != 0 || input_n > INPUT_ELEMENTS_PER_TILE) {
    return;
  }

  const uint32_t output_n = input_n >> 1;

  // Main path: PTO-native equivalent of the AscendC queue/DataCopy tiler.
  // Fallback: one generic 1D tiled path for widths that do not fit the 2D path.
  if ((output_n & (TILE_ALIGNMENT - 1U)) == 0) {
    runTSwiGLU2DMainTiled(x, y, batch, input_n, num_cores, vid);
  } else {
    runTSwiGLU1DFallbackTiled(x, y, batch, input_n, num_cores, vid);
  }
#else
  (void)x;
  (void)y;
  (void)batch;
  (void)input_n;
  (void)num_cores;
  (void)vid;
#endif
}

}  // namespace

__global__ AICORE void swiglu_dynamic_fp16(__gm__ void *x, __gm__ void *y,
                                           uint32_t batch, uint32_t input_n) {
#if defined(__DAV_VEC__)
  const uint32_t num_cores = get_block_num() * get_subblockdim();
  const uint32_t vid = get_block_idx() * get_subblockdim() + get_subblockid();
  runTSwiGLU<half>((__gm__ half *)x, (__gm__ half *)y, batch, input_n, num_cores,
                   vid);
#else
  (void)x;
  (void)y;
  (void)batch;
  (void)input_n;
#endif
}

extern "C" void call_swiglu_kernel(uint32_t blockDim, void *stream, uint8_t *x,
                                   uint8_t *y, uint32_t batch,
                                   uint32_t input_n) {
  swiglu_dynamic_fp16<<<blockDim * 2, nullptr, stream>>>(x, y, batch, input_n);
}
