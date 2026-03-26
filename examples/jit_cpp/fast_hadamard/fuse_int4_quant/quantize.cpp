#include <pto/pto-inst.hpp>

#include "int4_cvt.hpp"

using namespace pto;

#define DIV_ROUNDUP(x, y) (((x) + (y) - 1) / (y))

// INT4 path: balance tile size between small-batch (large tiles for
// amortization) and large-batch (smaller tiles for cache efficiency). Using
// 48KB buffers gives good balance: large enough for amortization, small enough
// to avoid TLB penalties on batch >= 128.
constexpr uint32_t X_BUFFER_BYTES = 32 * 1024;
constexpr uint32_t ELEMENTS_PER_TILE = X_BUFFER_BYTES / sizeof(half);
constexpr uint32_t Y_BUFFER_BYTES = ELEMENTS_PER_TILE / 2;
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
  // Partition by pairs for uniform load distribution across cores
  const uint32_t total_pairs = (batch * n) >> 1;
  const uint32_t pairs_per_core = DIV_ROUNDUP(total_pairs, num_cores);
  const uint32_t pair_offset = pairs_per_core * vid;
  if (pair_offset >= total_pairs) {
    return;
  }

  uint32_t pairs_to_process = pairs_per_core;
  if (pair_offset + pairs_to_process > total_pairs) {
    pairs_to_process = total_pairs - pair_offset;
  }
  if (pairs_to_process == 0) {
    return;
  }

  using InShapeDim5 = pto::Shape<1, 1, 1, 1, ELEMENTS_PER_TILE>;
  using OutShapeDim5 = pto::Shape<1, 1, 1, 1, ELEMENTS_PER_TILE / 2>;
  using StridDim5 = pto::Stride<1, 1, 1, 1, 1>;
  using InGlobal = pto::GlobalTensor<InputT, InShapeDim5, StridDim5>;
  using OutGlobal = pto::GlobalTensor<OutputT, OutShapeDim5, StridDim5>;
  using InTile = Tile<TileType::Vec, InputT, 1, ELEMENTS_PER_TILE,
                      BLayout::RowMajor, -1, -1>;
  using OutTile = Tile<TileType::Vec, OutputT, 1, ELEMENTS_PER_TILE / 2,
                       BLayout::RowMajor, -1, -1>;

  constexpr uint32_t PAIRS_PER_TILE = ELEMENTS_PER_TILE >> 1;

  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

  const uint32_t gm_x_offset = pair_offset << 1;
  const uint32_t gm_y_offset = pair_offset;

  for (uint32_t processed_pairs = 0, ping = 1;
       processed_pairs < pairs_to_process; processed_pairs += PAIRS_PER_TILE) {
    const uint32_t remaining_pairs = pairs_to_process - processed_pairs;
    const uint32_t cur_pairs =
        (remaining_pairs >= PAIRS_PER_TILE) ? PAIRS_PER_TILE : remaining_pairs;
    const uint32_t cur_cols = cur_pairs << 1;

    const event_t ev = ping ? (event_t)EVENT_ID0 : (event_t)EVENT_ID1;
    const unsigned x_base = ping ? X_PING : X_PONG;
    const unsigned y_base = ping ? Y_PING : Y_PONG;

    InTile xTile(1, cur_cols);
    OutTile yTile(1, cur_pairs);
    TASSIGN(xTile, x_base);
    TASSIGN(yTile, y_base);

    InGlobal xGlobal(x + gm_x_offset + (processed_pairs << 1));
    OutGlobal yGlobal(y + gm_y_offset + processed_pairs);
    TASSIGN(xGlobal, (x + gm_x_offset + (processed_pairs << 1)));
    TASSIGN(yGlobal, (y + gm_y_offset + processed_pairs));

    wait_flag(PIPE_V, PIPE_MTE2, ev);
    TLOAD(xTile, xGlobal);

    set_flag(PIPE_MTE2, PIPE_V, ev);
    wait_flag(PIPE_MTE2, PIPE_V, ev);
    wait_flag(PIPE_MTE3, PIPE_V, ev);

    TMULS(xTile, xTile, (InputT)scale);
    pipe_barrier(PIPE_V);

    fast_hadamard_int4::TCVT_FP16_TO_INT4_PACKED(yTile, xTile,
                                                 RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);

    set_flag(PIPE_V, PIPE_MTE3, ev);
    wait_flag(PIPE_V, PIPE_MTE3, ev);
    TSTORE(yGlobal, yTile);

    set_flag(PIPE_MTE3, PIPE_V, ev);
    set_flag(PIPE_V, PIPE_MTE2, ev);

    ping = 1 - ping;
  }

  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
}

}  // namespace

__global__ AICORE void quantize_fp16_to_int4(__gm__ void *x, __gm__ void *y,
                                             uint32_t batch, uint32_t n,
                                             float scale) {
#if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);

  if (n == 0 || (n & 1U) != 0) {
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
  quantize_fp16_to_int4<<<blockDim, nullptr, stream>>>(x, y, batch, n, scale);
}
