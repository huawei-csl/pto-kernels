#include <pto/pto-inst.hpp>

using namespace pto;

#define DIV_ROUNDUP(x, y) (((x) + (y) - 1) / (y))
#define ALIGN_UP(x, y) (DIV_ROUNDUP((x), (y)) * (y))

constexpr uint32_t UB_SLOT_BYTES = (192 * 1024) / 6;
constexpr uint32_t X0_BUFFER_BYTES = UB_SLOT_BYTES;
constexpr uint32_t X1_BUFFER_BYTES = UB_SLOT_BYTES;
constexpr uint32_t Y_BUFFER_BYTES = UB_SLOT_BYTES;
constexpr uint32_t ELEMENTS_PER_TILE = Y_BUFFER_BYTES / sizeof(half);
constexpr uint32_t MAX_INPUT_N = 2 * ELEMENTS_PER_TILE;
constexpr uint32_t UB_USABLE_BYTES = 192 * 1024;
constexpr uint32_t TILE_ALIGNMENT = 16;

#define SWIGLU_FOR_EACH_COL_TILE(X) \
  X(16)                             \
  X(32)                             \
  X(64)                             \
  X(128)                            \
  X(256)                            \
  X(512)                            \
  X(1024)                           \
  X(2048)                           \
  X(4096)                           \
  X(8192)                           \
  X(16384)

// Col tile widths tried by chooseTileConfig: powers of 2 from TILE_ALIGNMENT
// up to ELEMENTS_PER_TILE. The 16/32/64 entries handle narrow hidden dims.
#define SWIGLU_COL_VALUE(width) width,
constexpr uint32_t COL_TILE_CANDIDATES[] = {
    SWIGLU_FOR_EACH_COL_TILE(SWIGLU_COL_VALUE)};
#undef SWIGLU_COL_VALUE
constexpr uint32_t NUM_COL_TILE_CANDIDATES =
    sizeof(COL_TILE_CANDIDATES) / sizeof(COL_TILE_CANDIDATES[0]);

// Keep the active-tile target fixed at half the available cores.
constexpr uint32_t TARGET_ACTIVE_TILES_DIVISOR = 2;

constexpr unsigned X0_PING = 0x00000;
constexpr unsigned X1_PING = X0_PING + X0_BUFFER_BYTES;
constexpr unsigned Y_PING = X1_PING + X1_BUFFER_BYTES;
constexpr unsigned X0_PONG = Y_PING + Y_BUFFER_BYTES;
constexpr unsigned X1_PONG = X0_PONG + X0_BUFFER_BYTES;
constexpr unsigned Y_PONG = X1_PONG + X1_BUFFER_BYTES;

static_assert(UB_SLOT_BYTES * 6 == UB_USABLE_BYTES,
              "SwiGLU UB slots must fully pack the usable UB budget.");
static_assert(ELEMENTS_PER_TILE <= MAX_INPUT_N / 2,
              "SwiGLU tile size exceeds kernel max output tile.");
static_assert(Y_PONG + Y_BUFFER_BYTES <= UB_USABLE_BYTES,
              "SwiGLU UB layout exceeds usable UB.");

namespace {

struct TileConfig {
  uint32_t row_tile_len;
  uint32_t col_tile_len;
  uint32_t total_tiles;
  uint32_t area;
  bool meets_target;
};

struct TileWork {
  uint32_t row_offset;
  uint32_t col_offset;
  uint32_t row_count;
  uint32_t col_count;        // padded to TILE_ALIGNMENT; for UB tile sizing
  uint32_t col_count_store;  // actual output elements; for HBM stores/loads
};

AICORE inline uint32_t maxRowsForColTile(uint32_t col_tile_len) {
  return ELEMENTS_PER_TILE / col_tile_len;
}

AICORE inline void initTilePipeFlags() {
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
}

AICORE inline void drainTilePipeFlags() {
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
}

AICORE inline TileConfig makeTileConfig(uint32_t batch, uint32_t output_n,
                                        uint32_t num_cores,
                                        uint32_t col_tile_len) {
  const uint32_t max_rows = maxRowsForColTile(col_tile_len);
  const uint32_t col_tiles = DIV_ROUNDUP(output_n, col_tile_len);
  const uint32_t target_tiles =
      max(1U, num_cores / TARGET_ACTIVE_TILES_DIVISOR);
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
  return TileConfig{
      row_tile_len,
      col_tile_len,
      total_tiles,
      row_tile_len * col_tile_len,
      total_tiles >= target_tiles,
  };
}

AICORE inline bool preferTileConfig(const TileConfig &cand,
                                    const TileConfig &best) {
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

AICORE inline TileConfig chooseTileConfig(uint32_t batch, uint32_t output_n,
                                          uint32_t num_cores) {
  TileConfig best =
      makeTileConfig(batch, output_n, num_cores, COL_TILE_CANDIDATES[0]);
  for (uint32_t c = 1; c < NUM_COL_TILE_CANDIDATES; ++c) {
    const TileConfig cand =
        makeTileConfig(batch, output_n, num_cores, COL_TILE_CANDIDATES[c]);
    if (preferTileConfig(cand, best)) {
      best = cand;
    }
  }
  return best;
}

AICORE inline TileWork makeTileWork(uint32_t global_tile_idx,
                                    uint32_t row_tile_len,
                                    uint32_t col_tile_len, uint32_t batch,
                                    uint32_t output_n) {
  const uint32_t col_tiles = DIV_ROUNDUP(output_n, col_tile_len);
  const uint32_t row_tile_idx = global_tile_idx / col_tiles;
  const uint32_t col_tile_idx = global_tile_idx % col_tiles;
  const uint32_t row_offset = row_tile_idx * row_tile_len;
  const uint32_t col_offset = col_tile_idx * col_tile_len;
  const uint32_t actual_col = min(col_tile_len, output_n - col_offset);
  return TileWork{
      row_offset,
      col_offset,
      min(row_tile_len, batch - row_offset),
      ALIGN_UP(actual_col, TILE_ALIGNMENT),
      actual_col,
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

// col_count is padded for UB tile sizing; col_count_store is the actual GM
// element count so tail columns never read past the tensor end.
template <typename T, uint32_t kTileRows, uint32_t kTileCols>
AICORE void issueTLoad(__gm__ T *x, uint32_t input_n, uint32_t output_n,
                       const TileWork &tile, unsigned x0_base, unsigned x1_base,
                       event_t ev) {
  using TileShapeND = TileShape2D<T, DYNAMIC, DYNAMIC, Layout::ND>;
  using DynStrideND = Stride<1, 1, 1, DYNAMIC, 1>;
  using GlobalData = GlobalTensor<T, TileShapeND, DynStrideND, Layout::ND>;
  using TileData = Tile<TileType::Vec, T, kTileRows, kTileCols,
                        BLayout::RowMajor, DYNAMIC, DYNAMIC>;

  TileData x0Tile(tile.row_count, tile.col_count);
  TileData x1Tile(tile.row_count, tile.col_count);
  TASSIGN(x0Tile, x0_base);
  TASSIGN(x1Tile, x1_base);

  const uint32_t input_offset = tile.row_offset * input_n + tile.col_offset;
  const TileShapeND shape(tile.row_count, tile.col_count_store);
  const DynStrideND stride(input_n);

  GlobalData x0Global(x + input_offset, shape, stride);
  GlobalData x1Global(x + input_offset + output_n, shape, stride);

  wait_flag(PIPE_V, PIPE_MTE2, ev);
  wait_flag(PIPE_MTE3, PIPE_V, ev);
  TLOAD(x0Tile, x0Global);
  TLOAD(x1Tile, x1Global);
  set_flag(PIPE_MTE2, PIPE_V, ev);
}

// col_count is padded for UB tile sizing; col_count_store is the actual GM
// element count so tail columns never write past the tensor end.
template <typename T, uint32_t kTileRows, uint32_t kTileCols>
AICORE void issueTStore(__gm__ T *y, uint32_t output_n, const TileWork &tile,
                        unsigned y_base, event_t ev) {
  using TileShapeND = TileShape2D<T, DYNAMIC, DYNAMIC, Layout::ND>;
  using DynStrideND = Stride<1, 1, 1, DYNAMIC, 1>;
  using GlobalData = GlobalTensor<T, TileShapeND, DynStrideND, Layout::ND>;
  using TileData = Tile<TileType::Vec, T, kTileRows, kTileCols,
                        BLayout::RowMajor, DYNAMIC, DYNAMIC>;

  TileData yTile(tile.row_count, tile.col_count);
  TASSIGN(yTile, y_base);

  const uint32_t output_offset = tile.row_offset * output_n + tile.col_offset;
  const TileShapeND shape(tile.row_count, tile.col_count_store);
  const DynStrideND stride(output_n);
  GlobalData yGlobal(y + output_offset, shape, stride);

  set_flag(PIPE_V, PIPE_MTE3, ev);
  wait_flag(PIPE_V, PIPE_MTE3, ev);
  TSTORE(yGlobal, yTile);
  set_flag(PIPE_MTE3, PIPE_V, ev);
  set_flag(PIPE_V, PIPE_MTE2, ev);
}

template <uint32_t kTileCols, typename T>
AICORE void runTSwiGLUTiled(__gm__ T *x, __gm__ T *y, uint32_t batch,
                            uint32_t input_n, uint32_t num_cores, uint32_t vid,
                            uint32_t row_tile_len) {
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

  using TileData = Tile<TileType::Vec, T, kTileRows, kTileCols,
                        BLayout::RowMajor, DYNAMIC, DYNAMIC>;

  initTilePipeFlags();

  uint32_t current_tile_idx = vid;
  TileWork current_tile =
      makeTileWork(current_tile_idx, row_tile_len, kTileCols, batch, output_n);
  bool ping = true;
  issueTLoad<T, kTileRows, kTileCols>(x, input_n, output_n, current_tile,
                                      X0_PING, X1_PING, (event_t)EVENT_ID0);

  while (true) {
    const event_t current_ev = ping ? (event_t)EVENT_ID0 : (event_t)EVENT_ID1;
    const unsigned current_x0_base = ping ? X0_PING : X0_PONG;
    const unsigned current_x1_base = ping ? X1_PING : X1_PONG;
    const unsigned current_y_base = ping ? Y_PING : Y_PONG;

    wait_flag(PIPE_MTE2, PIPE_V, current_ev);

    TileWork next_tile{0, 0, 0, 0, 0};
    const uint32_t next_tile_idx = current_tile_idx + num_cores;
    const bool has_next = next_tile_idx < total_tiles;
    if (has_next) {
      next_tile =
          makeTileWork(next_tile_idx, row_tile_len, kTileCols, batch, output_n);
      const event_t next_ev = ping ? (event_t)EVENT_ID1 : (event_t)EVENT_ID0;
      const unsigned next_x0_base = ping ? X0_PONG : X0_PING;
      const unsigned next_x1_base = ping ? X1_PONG : X1_PING;
      issueTLoad<T, kTileRows, kTileCols>(x, input_n, output_n, next_tile,
                                          next_x0_base, next_x1_base, next_ev);
    }

    TileData x0Tile(current_tile.row_count, current_tile.col_count);
    TileData x1Tile(current_tile.row_count, current_tile.col_count);
    TileData yTile(current_tile.row_count, current_tile.col_count);
    TASSIGN(x0Tile, current_x0_base);
    TASSIGN(x1Tile, current_x1_base);
    TASSIGN(yTile, current_y_base);

    computeSwiGLUTile<TileData, T>(x0Tile, x1Tile, yTile);
    issueTStore<T, kTileRows, kTileCols>(y, output_n, current_tile,
                                         current_y_base, current_ev);

    if (!has_next) {
      break;
    }

    current_tile = next_tile;
    current_tile_idx = next_tile_idx;
    ping = !ping;
  }

  drainTilePipeFlags();
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
AICORE void runTSwiGLUMainTiled(__gm__ T *x, __gm__ T *y, uint32_t batch,
                                uint32_t input_n, uint32_t num_cores,
                                uint32_t vid) {
#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)
  const uint32_t output_n = input_n >> 1;
  const TileConfig cfg = chooseTileConfig(batch, output_n, num_cores);

  switch (cfg.col_tile_len) {
#define SWIGLU_TILE_CASE(width)                                     \
  case width:                                                       \
    runTSwiGLUTiled<width, T>(x, y, batch, input_n, num_cores, vid, \
                              cfg.row_tile_len);                    \
    break;
    SWIGLU_FOR_EACH_COL_TILE(SWIGLU_TILE_CASE)
#undef SWIGLU_TILE_CASE
    default:
      runTSwiGLUTiled<128, T>(x, y, batch, input_n, num_cores, vid,
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
AICORE void runTSwiGLU(__gm__ T *x, __gm__ T *y, uint32_t batch,
                       uint32_t input_n, uint32_t num_cores, uint32_t vid) {
#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);

  if (input_n == 0 || (input_n & 1U) != 0 || input_n > MAX_INPUT_N) {
    return;
  }

  // The 2D path handles all shapes including non-16-aligned output_n via
  // col_count_store: UB tiles are padded to TILE_ALIGNMENT, HBM
  // loads/stores use the actual element count.
  runTSwiGLUMainTiled(x, y, batch, input_n, num_cores, vid);
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
  runTSwiGLU<half>((__gm__ half *)x, (__gm__ half *)y, batch, input_n,
                   num_cores, vid);
#else
  (void)x;
  (void)y;
  (void)batch;
  (void)input_n;
#endif
}

/**
 * Launch the fp16 SwiGLU kernel.
 *
 * The input matrix is interpreted as two contiguous halves along the last
 * dimension:
 *   `x = [A | B]`, where `A = x[:, :input_n / 2]` and
 *   `B = x[:, input_n / 2:]`.
 *
 * The kernel computes SwiGLU row-wise:
 *   `y = silu(A) * B = (A * sigmoid(A)) * B`.
 *
 * Implementation notes:
 * - The output tensor is processed with a 2D tiler over rows and output cols.
 * - For each tile, the kernel loads matching slices from `A` and `B`,
 *   computes `silu(A) * B` in UB, then stores the result tile to `y`.
 * - Tail cols that are not 16-aligned are padded in UB, while GM loads/stores
 *   use the true element count to avoid reading or writing past the tensor end.
 *
 * @param blockDim Number of physical blocks to launch. The kernel expands this
 *     to `blockDim * 2` logical vector blocks.
 * @param stream Ascend runtime stream used for the kernel launch.
 * @param x Input buffer in global memory with shape `[batch, input_n]` and
 *     dtype `fp16`. `input_n` must be even.
 * @param y Output buffer in global memory with shape `[batch, input_n / 2]`
 *     and dtype `fp16`.
 * @param batch Number of rows in `x` and `y`.
 * @param input_n Input hidden dimension. The output hidden dimension is
 *     `input_n / 2`.
 */
extern "C" void call_swiglu_kernel(uint32_t blockDim, void *stream, uint8_t *x,
                                   uint8_t *y, uint32_t batch,
                                   uint32_t input_n) {
  swiglu_dynamic_fp16<<<blockDim * 2, nullptr, stream>>>(x, y, batch, input_n);
}
