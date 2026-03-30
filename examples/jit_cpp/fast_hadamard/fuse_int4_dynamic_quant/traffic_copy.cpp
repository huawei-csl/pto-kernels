#include <pto/pto-inst.hpp>

using namespace pto;

#define DIV_ROUNDUP(x, y) (((x) + (y) - 1) / (y))

constexpr uint32_t COPY_TILE_BYTES = 32 * 1024;
constexpr uint32_t UB_USABLE_BYTES = 72 * 1024;

constexpr unsigned COPY_PING = 0x00000;
constexpr unsigned COPY_PONG = COPY_PING + COPY_TILE_BYTES + 0x100;
static_assert(COPY_PONG + COPY_TILE_BYTES <= UB_USABLE_BYTES,
              "Traffic copy UB layout exceeds usable UB.");

namespace {

template <typename T, uint32_t kTileBytes>
AICORE void runTFlatCopy(__gm__ T *dst, __gm__ T *src, uint32_t count,
                         uint32_t num_cores, uint32_t vid) {
  constexpr uint32_t kElementsPerTile = kTileBytes / sizeof(T);

  const uint32_t elems_per_core = DIV_ROUNDUP(count, num_cores);
  const uint32_t elem_offset = elems_per_core * vid;
  if (elem_offset >= count) {
    return;
  }

  uint32_t elems_to_process = elems_per_core;
  if (elem_offset + elems_to_process > count) {
    elems_to_process = count - elem_offset;
  }
  if (elems_to_process == 0) {
    return;
  }

  using ShapeDim5 = pto::Shape<1, 1, 1, 1, kElementsPerTile>;
  using StridDim5 = pto::Stride<1, 1, 1, 1, 1>;
  using GlobalData = pto::GlobalTensor<T, ShapeDim5, StridDim5>;
  using CopyTile = Tile<TileType::Vec, T, 1, kElementsPerTile, BLayout::RowMajor,
                        -1, -1>;

  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

  bool ping = true;
  for (uint32_t processed = 0; processed < elems_to_process;
       processed += kElementsPerTile) {
    const uint32_t remaining = elems_to_process - processed;
    const uint32_t cur_count =
        (remaining >= kElementsPerTile) ? kElementsPerTile : remaining;

    const event_t ev = ping ? (event_t)EVENT_ID0 : (event_t)EVENT_ID1;
    const unsigned copy_base = ping ? COPY_PING : COPY_PONG;

    CopyTile copy_tile(1, cur_count);
    TASSIGN(copy_tile, copy_base);

    GlobalData src_global(src + elem_offset + processed);
    GlobalData dst_global(dst + elem_offset + processed);
    TASSIGN(src_global, (src + elem_offset + processed));
    TASSIGN(dst_global, (dst + elem_offset + processed));

    wait_flag(PIPE_V, PIPE_MTE2, ev);
    TLOAD(copy_tile, src_global);

    set_flag(PIPE_MTE2, PIPE_V, ev);
    wait_flag(PIPE_MTE2, PIPE_V, ev);
    wait_flag(PIPE_MTE3, PIPE_V, ev);

    set_flag(PIPE_V, PIPE_MTE3, ev);
    wait_flag(PIPE_V, PIPE_MTE3, ev);
    TSTORE(dst_global, copy_tile);

    set_flag(PIPE_MTE3, PIPE_V, ev);
    set_flag(PIPE_V, PIPE_MTE2, ev);
    ping = !ping;
  }

  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
}

}  // namespace

__global__ AICORE void traffic_copy_bytes(__gm__ void *src, __gm__ void *dst,
                                          uint32_t byte_count) {
#if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);

  const uint32_t num_cores = get_block_num() * get_subblockdim();
  const uint32_t vid = get_block_idx() * get_subblockdim() + get_subblockid();
  runTFlatCopy<int8_t, COPY_TILE_BYTES>((__gm__ int8_t *)dst,
                                        (__gm__ int8_t *)src, byte_count,
                                        num_cores, vid);
#else
  (void)src;
  (void)dst;
  (void)byte_count;
#endif
}

extern "C" void call_traffic_copy_kernel(uint32_t blockDim, void *stream,
                                         uint8_t *src, uint8_t *dst,
                                         uint32_t byte_count) {
  blockDim = blockDim * 2;
  traffic_copy_bytes<<<blockDim, nullptr, stream>>>(src, dst, byte_count);
}
