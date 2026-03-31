#include <pto/pto-inst.hpp>

using namespace pto;

#define DIV_ROUNDUP(x, y) (((x) + (y) - 1) / (y))

constexpr uint32_t X_BUFFER_BYTES = 64 * 1024;
constexpr uint32_t ELEMENTS_PER_TILE = X_BUFFER_BYTES / sizeof(half);

constexpr unsigned X_PING = 0x00000;
constexpr unsigned X_PONG = X_PING + X_BUFFER_BYTES;
static_assert(X_PONG + X_BUFFER_BYTES <= 192 * 1024,
              "PTO copy UB layout exceeds 192 KB.");

namespace {

struct TileWork {
  uint32_t gm_offset, sample_count, elements;
};

template <typename InputT>
AICORE void issueTLoad(__gm__ InputT *x, const TileWork &tile, unsigned x_base,
                       event_t ev) {
  using ShapeDim5 = pto::Shape<1, 1, 1, 1, ELEMENTS_PER_TILE>;
  using StridDim5 = pto::Stride<1, 1, 1, 1, 1>;
  using InGlobal = pto::GlobalTensor<InputT, ShapeDim5, StridDim5>;
  using FullTile = Tile<TileType::Vec, InputT, 1, ELEMENTS_PER_TILE,
                        BLayout::RowMajor, -1, -1>;

  FullTile xBulkTile(1, tile.elements);
  TASSIGN(xBulkTile, x_base);

  InGlobal xGlobal(x + tile.gm_offset);
  TASSIGN(xGlobal, (x + tile.gm_offset));

  wait_flag(PIPE_V, PIPE_MTE2, ev);
  TLOAD(xBulkTile, xGlobal);
  set_flag(PIPE_MTE2, PIPE_V, ev);
}

AICORE bool nextTile(uint32_t &sample_done, uint32_t gm_offset_base,
                     uint32_t samples_to_process, uint32_t samples_per_load,
                     uint32_t n, TileWork &tile) {
  if (sample_done >= samples_to_process) {
    return false;
  }

  tile.sample_count = min(samples_per_load, samples_to_process - sample_done);
  tile.elements = tile.sample_count * n;
  tile.gm_offset = gm_offset_base + sample_done * n;
  sample_done += tile.sample_count;
  return true;
}

template <typename T>
AICORE void runTPtoCopy(__gm__ T *x, __gm__ T *y, uint32_t batch, uint32_t n) {
#if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);

  if (n == 0 || n > ELEMENTS_PER_TILE) {
    return;
  }

  const uint32_t num_cores = get_block_num() * get_subblockdim();
  const uint32_t vid = get_block_idx() * get_subblockdim() + get_subblockid();

  const uint32_t samples_per_core = DIV_ROUNDUP(batch, num_cores);
  const uint32_t sample_offset = samples_per_core * vid;
  if (sample_offset >= batch) {
    return;
  }

  uint32_t samples_to_process = samples_per_core;
  if (sample_offset + samples_to_process > batch) {
    samples_to_process = batch - sample_offset;
  }
  if (samples_to_process == 0) {
    return;
  }

  using ShapeDim5 = pto::Shape<1, 1, 1, 1, ELEMENTS_PER_TILE>;
  using StridDim5 = pto::Stride<1, 1, 1, 1, 1>;
  using GlobalData = pto::GlobalTensor<T, ShapeDim5, StridDim5>;
  using FullTile =
      Tile<TileType::Vec, T, 1, ELEMENTS_PER_TILE, BLayout::RowMajor, -1, -1>;

  const uint32_t samples_per_load =
      (n < ELEMENTS_PER_TILE) ? ELEMENTS_PER_TILE / n : 1;

  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

  uint32_t sample_done = 0;
  const uint32_t gm_offset_base = sample_offset * n;
  TileWork current_tile;
  if (!nextTile(sample_done, gm_offset_base, samples_to_process,
                samples_per_load, n, current_tile)) {
    return;
  }

  bool ping = true;
  issueTLoad(x, current_tile, X_PING, EVENT_ID0);

  while (true) {
    const event_t current_ev = ping ? (event_t)EVENT_ID0 : (event_t)EVENT_ID1;
    const unsigned x_base = ping ? X_PING : X_PONG;

    wait_flag(PIPE_MTE2, PIPE_V, current_ev);

    TileWork next_tile;
    const bool has_next =
        nextTile(sample_done, gm_offset_base, samples_to_process,
                 samples_per_load, n, next_tile);
    if (has_next) {
      const event_t next_ev = ping ? (event_t)EVENT_ID1 : (event_t)EVENT_ID0;
      const unsigned next_x_base = ping ? X_PONG : X_PING;
      issueTLoad(x, next_tile, next_x_base, next_ev);
    }

    FullTile xBulkTile(1, current_tile.elements);
    TASSIGN(xBulkTile, x_base);

    GlobalData yGlobal(y + current_tile.gm_offset);
    TASSIGN(yGlobal, (y + current_tile.gm_offset));

    wait_flag(PIPE_MTE3, PIPE_V, current_ev);
    set_flag(PIPE_V, PIPE_MTE3, current_ev);
    wait_flag(PIPE_V, PIPE_MTE3, current_ev);
    TSTORE(yGlobal, xBulkTile);

    set_flag(PIPE_MTE3, PIPE_V, current_ev);
    set_flag(PIPE_V, PIPE_MTE2, current_ev);

    if (!has_next) {
      break;
    }

    current_tile = next_tile;
    ping = !ping;
  }

  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
#endif
}

}  // namespace

extern "C" __global__ AICORE void pto_copy_fp16(__gm__ void *x, __gm__ void *y,
                                                uint32_t batch, uint32_t n) {
  runTPtoCopy<half>((__gm__ half *)x, (__gm__ half *)y, batch, n);
}

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *x,
                            uint8_t *y, uint32_t batch, uint32_t n) {
  pto_copy_fp16<<<blockDim * 2, nullptr, stream>>>(x, y, batch, n);
}
