#include <pto/pto-inst.hpp>

using namespace pto;

#define DIV_ROUNDUP(x, y) (((x) + (y) - 1) / (y))

constexpr uint32_t X_BUFFER_BYTES = 32 * 1024;
constexpr uint32_t ELEMENTS_PER_TILE = X_BUFFER_BYTES / sizeof(half);
constexpr uint32_t Y_BUFFER_BYTES = ELEMENTS_PER_TILE * sizeof(int8_t);
constexpr uint32_t UB_USABLE_BYTES = 256 * 1024;

constexpr unsigned X_PING = 0x00000;
constexpr unsigned Y_PING = (X_PING + X_BUFFER_BYTES + 0x100);
constexpr unsigned X_PONG = (Y_PING + Y_BUFFER_BYTES + 0x100);
constexpr unsigned Y_PONG = (X_PONG + X_BUFFER_BYTES + 0x100);
static_assert(Y_PONG + Y_BUFFER_BYTES <= UB_USABLE_BYTES,
              "Quantize UB layout exceeds usable UB.");

namespace {

template <typename InputT, typename OutputT>
AICORE void runTQuantize(__gm__ OutputT *y, __gm__ InputT *x, uint32_t batch,
                         uint32_t n, uint32_t num_cores, uint32_t vid,
                         float scale) {
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
  using InGlobal = pto::GlobalTensor<InputT, ShapeDim5, StridDim5>;
  using OutGlobal = pto::GlobalTensor<OutputT, ShapeDim5, StridDim5>;
  using InTile = Tile<TileType::Vec, InputT, 1, ELEMENTS_PER_TILE,
                      BLayout::RowMajor, -1, -1>;
  using OutTile = Tile<TileType::Vec, OutputT, 1, ELEMENTS_PER_TILE,
                       BLayout::RowMajor, -1, -1>;

  const uint32_t samples_per_load =
      (n < ELEMENTS_PER_TILE) ? (ELEMENTS_PER_TILE / n) : 1;

  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

  uint32_t gm_offset = sample_offset * n;

  for (uint32_t sample_done = 0, ping = 1; sample_done < samples_to_process;
       sample_done += samples_per_load) {
    uint32_t cur_samples = samples_per_load;
    if (sample_done + cur_samples > samples_to_process) {
      cur_samples = samples_to_process - sample_done;
    }

    const uint32_t group_elements = cur_samples * n;

    for (uint32_t processed = 0; processed < group_elements;
         processed += ELEMENTS_PER_TILE) {
      const uint32_t remaining = group_elements - processed;
      const uint32_t cur_cols =
          (remaining >= ELEMENTS_PER_TILE) ? ELEMENTS_PER_TILE : remaining;

      const event_t ev = ping ? (event_t)EVENT_ID0 : (event_t)EVENT_ID1;
      const unsigned x_base = ping ? X_PING : X_PONG;
      const unsigned y_base = ping ? Y_PING : Y_PONG;

      InTile xTile(1, cur_cols);
      OutTile yTile(1, cur_cols);
      TASSIGN(xTile, x_base);
      TASSIGN(yTile, y_base);

      InGlobal xGlobal(x + gm_offset + processed);
      OutGlobal yGlobal(y + gm_offset + processed);
      TASSIGN(xGlobal, (x + gm_offset + processed));
      TASSIGN(yGlobal, (y + gm_offset + processed));

      wait_flag(PIPE_V, PIPE_MTE2, ev);
      TLOAD(xTile, xGlobal);

      set_flag(PIPE_MTE2, PIPE_V, ev);
      wait_flag(PIPE_MTE2, PIPE_V, ev);
      wait_flag(PIPE_MTE3, PIPE_V, ev);

      TMULS(xTile, xTile, (InputT)scale);
      pipe_barrier(PIPE_V);

      TCVT(yTile, xTile, RoundMode::CAST_NONE);
      pipe_barrier(PIPE_V);

      set_flag(PIPE_V, PIPE_MTE3, ev);
      wait_flag(PIPE_V, PIPE_MTE3, ev);
      TSTORE(yGlobal, yTile);

      set_flag(PIPE_MTE3, PIPE_V, ev);
      set_flag(PIPE_V, PIPE_MTE2, ev);

      ping = 1 - ping;
    }

    gm_offset += group_elements;
  }

  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
}

}  // namespace

__global__ AICORE void quantize_fp16_to_int8(__gm__ void *x, __gm__ void *y,
                                             uint32_t batch, uint32_t n,
                                             float scale) {
#if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);

  if (n == 0) {
    return;
  }

  const uint32_t num_cores = get_block_num() * get_subblockdim();
  const uint32_t vid = get_block_idx() * get_subblockdim() + get_subblockid();

  runTQuantize<half, int8_t>((__gm__ int8_t *)y, (__gm__ half *)x, batch, n,
                             num_cores, vid, scale);
#else
  (void)x;
  (void)y;
  (void)batch;
  (void)n;
  (void)scale;
#endif
}

extern "C" void call_quantize_kernel(uint32_t blockDim, void *stream,
                                     uint8_t *x, uint8_t *y, uint32_t batch,
                                     uint32_t n, float scale) {
  blockDim = blockDim * 2;
  quantize_fp16_to_int8<<<blockDim, nullptr, stream>>>(x, y, batch, n, scale);
}
