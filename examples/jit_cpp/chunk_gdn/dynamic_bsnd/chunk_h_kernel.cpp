#include <pto/pto-inst.hpp>
#include <runtime/rt_ffts.h>

#include "gdn_pto_shared.h"

using namespace pto;

#ifndef GDN_H
#define GDN_H 2
#endif

#ifndef GDN_D
#define GDN_D 128
#endif

#ifndef GDN_C
#define GDN_C 128
#endif

template <int32_t NumHeads, int32_t HiddenSize, int32_t ChunkSize>
AICORE void ws_kernel(__gm__ half *w_packed, __gm__ half *state_packed,
                      __gm__ float *ws_out, int64_t total_chunks,
                      uint64_t ffts_addr) {
  constexpr int32_t ChunkHiddenElems = ChunkSize * HiddenSize;
  constexpr int32_t HiddenSquareElems = HiddenSize * HiddenSize;
  constexpr int32_t WL1Addr = 0;
  constexpr int32_t SL1Addr = 32768;

  using PackedChunk = GlobalTensor<half, TileShape2D<half, ChunkSize, HiddenSize, Layout::ND>,
                                   BaseShape2D<half, ChunkSize, HiddenSize, Layout::ND>, Layout::ND>;
  using PackedState = GlobalTensor<half, TileShape2D<half, HiddenSize, HiddenSize, Layout::ND>,
                                   BaseShape2D<half, HiddenSize, HiddenSize, Layout::ND>, Layout::ND>;
  using PackedOut = GlobalTensor<float, TileShape2D<float, ChunkSize, HiddenSize, Layout::ND>,
                                 BaseShape2D<float, ChunkSize, HiddenSize, Layout::ND>, Layout::ND>;

  set_ffts_base_addr(ffts_addr);
  const int64_t cid = get_block_idx();
  const int64_t total_work = total_chunks * NumHeads;

  GdnL1Mat<half, ChunkSize, HiddenSize> w_l1;
  GdnL1Mat<half, HiddenSize, HiddenSize> s_l1;
  TASSIGN(w_l1, WL1Addr);
  TASSIGN(s_l1, SL1Addr);
  TileAcc<float, ChunkSize, HiddenSize, ChunkSize, HiddenSize> out_l0;
  TASSIGN(out_l0, 0);

#if defined(__DAV_C220_CUBE__)
  for (int64_t work_idx = 0; work_idx < (total_work + block_num - 1) / block_num;
       ++work_idx) {
    const int64_t pid = work_idx * block_num + cid;
    if (pid >= total_work) {
      continue;
    }
    const int64_t packed_base = pid;
    PackedChunk w_global(w_packed + packed_base * ChunkHiddenElems);
    PackedState s_global(state_packed + packed_base * HiddenSquareElems);
    PackedOut out_global(ws_out + packed_base * ChunkHiddenElems);
    TLOAD(w_l1, w_global);
    TLOAD(s_l1, s_global);
    pipe_barrier(PIPE_ALL);
    GdnMatmulL1<ChunkSize, HiddenSize, HiddenSize, false, false>(out_l0, w_l1,
                                                                 s_l1, true);
    TSTORE(out_global, out_l0);
    pipe_barrier(PIPE_ALL);
  }
#endif
}

template <int32_t NumHeads, int32_t HiddenSize, int32_t ChunkSize>
AICORE void kv_kernel(__gm__ half *k_scaled, __gm__ half *new_v,
                      __gm__ float *kv_out, int64_t total_chunks,
                      uint64_t ffts_addr) {
  constexpr int32_t ChunkHiddenElems = ChunkSize * HiddenSize;
  constexpr int32_t HiddenSquareElems = HiddenSize * HiddenSize;
  constexpr int32_t KL1Addr = 0;
  constexpr int32_t VL1Addr = 32768;

  using PackedChunk = GlobalTensor<half, TileShape2D<half, ChunkSize, HiddenSize, Layout::ND>,
                                   BaseShape2D<half, ChunkSize, HiddenSize, Layout::ND>, Layout::ND>;
  using PackedOut = GlobalTensor<float, TileShape2D<float, HiddenSize, HiddenSize, Layout::ND>,
                                 BaseShape2D<float, HiddenSize, HiddenSize, Layout::ND>, Layout::ND>;

  set_ffts_base_addr(ffts_addr);
  const int64_t cid = get_block_idx();
  const int64_t total_work = total_chunks * NumHeads;

  GdnL1Mat<half, ChunkSize, HiddenSize> k_l1;
  GdnL1Mat<half, ChunkSize, HiddenSize> v_l1;
  TASSIGN(k_l1, KL1Addr);
  TASSIGN(v_l1, VL1Addr);
  TileAcc<float, HiddenSize, HiddenSize, HiddenSize, HiddenSize> out_l0;
  TASSIGN(out_l0, 0);

#if defined(__DAV_C220_CUBE__)
  for (int64_t work_idx = 0; work_idx < (total_work + block_num - 1) / block_num;
       ++work_idx) {
    const int64_t pid = work_idx * block_num + cid;
    if (pid >= total_work) {
      continue;
    }
    const int64_t packed_base = pid;
    PackedChunk k_global(k_scaled + packed_base * ChunkHiddenElems);
    PackedChunk v_global(new_v + packed_base * ChunkHiddenElems);
    PackedOut out_global(kv_out + packed_base * HiddenSquareElems);
    TLOAD(k_l1, k_global);
    TLOAD(v_l1, v_global);
    pipe_barrier(PIPE_ALL);
    GdnMatmulL1<HiddenSize, HiddenSize, ChunkSize, true, false>(out_l0, k_l1,
                                                                v_l1, true);
    TSTORE(out_global, out_l0);
    pipe_barrier(PIPE_ALL);
  }
#endif
}

extern "C" __global__ AICORE void launch_chunk_h_ws(
    __gm__ uint8_t *w_packed, __gm__ uint8_t *state_packed, __gm__ uint8_t *ws_out,
    int64_t total_chunks, uint64_t ffts_addr) {
  ws_kernel<GDN_H, GDN_D, GDN_C>(
      reinterpret_cast<__gm__ half *>(w_packed),
      reinterpret_cast<__gm__ half *>(state_packed),
      reinterpret_cast<__gm__ float *>(ws_out), total_chunks, ffts_addr);
}

extern "C" __global__ AICORE void launch_chunk_h_kv(
    __gm__ uint8_t *k_scaled, __gm__ uint8_t *new_v, __gm__ uint8_t *kv_out,
    int64_t total_chunks, uint64_t ffts_addr) {
  kv_kernel<GDN_H, GDN_D, GDN_C>(
      reinterpret_cast<__gm__ half *>(k_scaled),
      reinterpret_cast<__gm__ half *>(new_v),
      reinterpret_cast<__gm__ float *>(kv_out), total_chunks, ffts_addr);
}

extern "C" void call_ws_kernel(uint32_t blockDim, void *stream, uint8_t *w_packed,
                               uint8_t *state_packed, uint8_t *ws_out,
                               int64_t total_chunks) {
  uint32_t ffts_len = 0;
  uint64_t ffts_addr = 0;
  rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
  launch_chunk_h_ws<<<blockDim, nullptr, stream>>>(w_packed, state_packed, ws_out,
                                                   total_chunks, ffts_addr);
}

extern "C" void call_kv_kernel(uint32_t blockDim, void *stream, uint8_t *k_scaled,
                               uint8_t *new_v, uint8_t *kv_out,
                               int64_t total_chunks) {
  uint32_t ffts_len = 0;
  uint64_t ffts_addr = 0;
  rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
  launch_chunk_h_kv<<<blockDim, nullptr, stream>>>(k_scaled, new_v, kv_out,
                                                   total_chunks, ffts_addr);
}
