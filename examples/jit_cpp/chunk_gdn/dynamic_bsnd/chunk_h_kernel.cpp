#include <pto/pto-inst.hpp>
#include <runtime/rt_ffts.h>

#include "gdn_pto_shared.h"
#include "gdn_seq_info.h"

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

AICORE inline uint32_t GdnMinU32(uint32_t a, uint32_t b) { return a < b ? a : b; }

template <int32_t NumHeads, int32_t HiddenSize, int32_t ChunkSize>
AICORE void chunk_h_main_kernel(
    __gm__ half *k_bsnd, __gm__ half *w_packed, __gm__ half *u_packed,
    __gm__ float *g_packed, __gm__ half *s_out, __gm__ half *nv_out,
    __gm__ half *fs_out, __gm__ half *workspace,
    __gm__ int32_t *cu_seqlens, int64_t batch_size,
    int64_t fixed_seq_len, uint64_t ffts_addr) {
  constexpr int32_t HalfChunk = ChunkSize / 2;
  constexpr int32_t ChunkHiddenElems = ChunkSize * HiddenSize;
  constexpr int32_t HiddenSquareElems = HiddenSize * HiddenSize;

  constexpr int32_t WorkspaceBlockStride = 3 * ChunkHiddenElems;

  constexpr int32_t AL1Addr = 0;
  constexpr int32_t BL1Addr = 32768;

  constexpr int32_t SUbAddr = 0;
  constexpr int32_t KHalfUbAddr = SUbAddr + HalfChunk * HiddenSize * static_cast<int32_t>(sizeof(float));
  constexpr int32_t GUbAddr = KHalfUbAddr + HalfChunk * HiddenSize * static_cast<int32_t>(sizeof(half));
  constexpr int32_t UHalfUbAddr = GUbAddr + ChunkSize * static_cast<int32_t>(sizeof(float));
  constexpr int32_t KUbAddr = UHalfUbAddr + HalfChunk * HiddenSize * static_cast<int32_t>(sizeof(half));
  constexpr int32_t GvUbAddr = KUbAddr + HalfChunk * HiddenSize * static_cast<int32_t>(sizeof(float));
  constexpr int32_t CoeffUbAddr = GvUbAddr + HalfChunk * static_cast<int32_t>(sizeof(float));
  constexpr int32_t UUbAddr = CoeffUbAddr + HalfChunk * static_cast<int32_t>(sizeof(float));
  constexpr int32_t WsUbAddr = UUbAddr + HalfChunk * HiddenSize * static_cast<int32_t>(sizeof(float));
  constexpr int32_t SHalfUbAddr = WsUbAddr + HalfChunk * HiddenSize * static_cast<int32_t>(sizeof(float));
  constexpr int32_t KvUbAddr = UHalfUbAddr;

  using PackedHidden =
      GlobalTensor<half, TileShape2D<half, ChunkSize, HiddenSize, Layout::ND>,
                   BaseShape2D<half, ChunkSize, HiddenSize, Layout::ND>, Layout::ND>;
  using PackedHiddenHalf =
      GlobalTensor<half, TileShape2D<half, HalfChunk, HiddenSize, Layout::ND>,
                   BaseShape2D<half, HalfChunk, HiddenSize, Layout::ND>, Layout::ND>;
  using PackedGGlobal =
      GlobalTensor<float, TileShape2D<float, 1, ChunkSize, Layout::ND>,
                   BaseShape2D<float, 1, ChunkSize, Layout::ND>, Layout::ND>;
  using DynGlobalShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using DynGlobalStride = Stride<1, 1, 1, DYNAMIC, 1>;
  using DynGlobalHalf = GlobalTensor<half, DynGlobalShape, DynGlobalStride, Layout::ND>;
  using DynL1 = Tile<TileType::Mat, half, ChunkSize, HiddenSize, BLayout::ColMajor,
                     DYNAMIC, DYNAMIC, SLayout::RowMajor, 512, PadValue::Zero>;

  using SUb = GdnUbND<float, HalfChunk, HiddenSize>;
  using KHalfUb = GdnUbND<half, HalfChunk, HiddenSize>;
  using GUb = GdnUbND<float, 1, ChunkSize>;
  using UHalfUb = GdnUbND<half, HalfChunk, HiddenSize>;
  using KUb = GdnUbND<float, HalfChunk, HiddenSize>;
  using GvUb = GdnUbND<float, 1, HalfChunk>;
  using CoeffUb = GdnUbND<float, 1, HalfChunk>;
  using UUb = GdnUbND<float, HalfChunk, HiddenSize>;
  using WsUb = GdnUbND<float, HalfChunk, HiddenSize>;
  using SHalfUb = GdnUbND<half, HalfChunk, HiddenSize>;
  using CoeffColUb = GdnUbDN<float, HalfChunk, 1>;
  using KHalfUbDyn = Tile<TileType::Vec, half, HalfChunk, HiddenSize,
                          BLayout::RowMajor, DYNAMIC, DYNAMIC,
                          SLayout::NoneBox, 512, PadValue::Zero>;
  using UHalfUbDyn = Tile<TileType::Vec, half, HalfChunk, HiddenSize,
                          BLayout::RowMajor, DYNAMIC, DYNAMIC,
                          SLayout::NoneBox, 512, PadValue::Zero>;

  set_ffts_base_addr(ffts_addr);
  const int64_t cid = get_block_idx();
  const int64_t vid = get_subblockid();
  const int64_t total_work = batch_size * NumHeads;

  const int32_t ws_kv_base =
      static_cast<int32_t>(cid) * WorkspaceBlockStride;
  const int32_t kscaled_base = ws_kv_base + ChunkHiddenElems;
  const int32_t state_base = ws_kv_base + 2 * ChunkHiddenElems;

  GdnL1Mat<half, ChunkSize, HiddenSize> a_l1;
  GdnL1Mat<half, ChunkSize, HiddenSize> b_l1;
  TASSIGN(a_l1, AL1Addr);
  TASSIGN(b_l1, BL1Addr);
  TileAcc<float, ChunkSize, HiddenSize, ChunkSize, HiddenSize> out_l0;
  TASSIGN(out_l0, 0);

  SUb s_ub;
  KHalfUb k_ub_half;
  GUb g_ub;
  UHalfUb u_ub_half;
  KUb k_ub;
  GvUb g_v_ub;
  CoeffUb coeff_ub;
  UUb u_ub;
  WsUb ws_ub;
  SHalfUb s_ub_half;
  CoeffColUb coeff_col_ub;
  SUb kv_ub;
  TASSIGN(s_ub, SUbAddr);
  TASSIGN(k_ub_half, KHalfUbAddr);
  TASSIGN(g_ub, GUbAddr);
  TASSIGN(u_ub_half, UHalfUbAddr);
  TASSIGN(k_ub, KUbAddr);
  TASSIGN(g_v_ub, GvUbAddr);
  TASSIGN(coeff_ub, CoeffUbAddr);
  TASSIGN(u_ub, UUbAddr);
  TASSIGN(ws_ub, WsUbAddr);
  TASSIGN(s_ub_half, SHalfUbAddr);
  TASSIGN(coeff_col_ub, CoeffUbAddr);
  TASSIGN(kv_ub, KvUbAddr);

#if defined(__DAV_C220_CUBE__)
  for (int64_t work_idx = 0;
       work_idx < (total_work + block_num - 1) / block_num; ++work_idx) {
    const int64_t pid = work_idx * block_num + cid;
    if (pid >= total_work) continue;
    const uint32_t head_idx = static_cast<uint32_t>(pid % NumHeads);
    const uint32_t seq_idx = static_cast<uint32_t>(pid / NumHeads);
    const GdnBsndSeqInfo seq = GetGdnBsndSeqInfo(
        seq_idx, head_idx, NumHeads, HiddenSize, ChunkSize,
        static_cast<uint32_t>(fixed_seq_len), cu_seqlens);
    const uint32_t chunk_num = GdnDivCeilU32(seq.seq_len, ChunkSize);

    for (uint32_t chunk_idx = 0; chunk_idx < chunk_num; ++chunk_idx) {
      const uint32_t row_start = chunk_idx * ChunkSize;
      const uint32_t valid_rows = GdnMinU32(
          static_cast<uint32_t>(seq.seq_len - row_start),
          static_cast<uint32_t>(ChunkSize));
      const int32_t chunk_base = static_cast<int32_t>(
          (seq.chunk_offset + chunk_idx) * NumHeads + head_idx);

      GdnWaitCrossFlag(3);
      pipe_barrier(PIPE_ALL);
      {
        PackedHidden w_global(w_packed + chunk_base * ChunkHiddenElems);
        PackedHidden state_global(workspace + state_base);
        TLOAD(a_l1, w_global);
        TLOAD(b_l1, state_global);
        pipe_barrier(PIPE_ALL);
        GdnMatmulL1<ChunkSize, HiddenSize, HiddenSize, false, false>(
            out_l0, a_l1, b_l1, true);
        PackedHidden ws_global(workspace + ws_kv_base);
        TSTORE(ws_global, out_l0);
        pipe_barrier(PIPE_ALL);
      }
      GdnSetCrossFlag<PIPE_FIX>(0, 2);

      GdnWaitCrossFlag(1);
      pipe_barrier(PIPE_ALL);
      {
        DynL1 k_dyn(valid_rows, HiddenSize);
        DynL1 v_dyn(valid_rows, HiddenSize);
        TASSIGN(k_dyn, AL1Addr);
        TASSIGN(v_dyn, BL1Addr);
        PackedHidden kscaled_global(workspace + kscaled_base);
        PackedHidden nv_global(nv_out + chunk_base * ChunkHiddenElems);
        TLOAD(k_dyn, kscaled_global);
        TLOAD(v_dyn, nv_global);
        pipe_barrier(PIPE_ALL);
        GdnMatmulL1<HiddenSize, HiddenSize, ChunkSize, true, false>(
            out_l0, a_l1, b_l1, true);
        PackedHidden kv_global(workspace + ws_kv_base);
        TSTORE(kv_global, out_l0);
        pipe_barrier(PIPE_ALL);
      }
      GdnSetCrossFlag<PIPE_FIX>(2, 2);
    }
  }
#endif

#if defined(__DAV_C220_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);

  for (int64_t work_idx = 0;
       work_idx < (total_work + block_num - 1) / block_num; ++work_idx) {
    const int64_t pid = work_idx * block_num + cid;
    if (pid >= total_work) continue;
    const uint32_t head_idx = static_cast<uint32_t>(pid % NumHeads);
    const uint32_t seq_idx = static_cast<uint32_t>(pid / NumHeads);
    const GdnBsndSeqInfo seq = GetGdnBsndSeqInfo(
        seq_idx, head_idx, NumHeads, HiddenSize, ChunkSize,
        static_cast<uint32_t>(fixed_seq_len), cu_seqlens);
    const uint32_t chunk_num = GdnDivCeilU32(seq.seq_len, ChunkSize);

    TEXPANDS(s_ub, 0.0f);
    pipe_barrier(PIPE_V);
    TCVT(s_ub_half, s_ub, pto::RoundMode::CAST_NONE);
    GdnSetFlag<PIPE_V, PIPE_MTE3>(0);
    GdnWaitFlag<PIPE_V, PIPE_MTE3>(0);

    PackedHiddenHalf state_ws_init(
        workspace + state_base +
        static_cast<int32_t>(vid) * HalfChunk * HiddenSize);
    TSTORE(state_ws_init, s_ub_half);

    if (chunk_num > 0) {
      const int32_t first_cb = static_cast<int32_t>(
          seq.chunk_offset * NumHeads + head_idx);
      PackedHiddenHalf s_out_init(
          s_out + first_cb * HiddenSquareElems +
          static_cast<int32_t>(vid) * HalfChunk * HiddenSize);
      TSTORE(s_out_init, s_ub_half);
    }
    pipe_barrier(PIPE_ALL);
    GdnSetCrossFlag<PIPE_MTE3>(3, 2);

    for (uint32_t chunk_idx = 0; chunk_idx < chunk_num; ++chunk_idx) {
      const uint32_t row_start = chunk_idx * ChunkSize;
      const uint32_t valid_rows = GdnMinU32(
          static_cast<uint32_t>(seq.seq_len - row_start),
          static_cast<uint32_t>(ChunkSize));
      const uint32_t row_offset = static_cast<uint32_t>(vid) * HalfChunk;
      const uint32_t local_rows =
          valid_rows > row_offset
              ? GdnMinU32(static_cast<uint32_t>(valid_rows - row_offset),
                          static_cast<uint32_t>(HalfChunk))
              : 0;
      const int32_t chunk_base = static_cast<int32_t>(
          (seq.chunk_offset + chunk_idx) * NumHeads + head_idx);

      PackedGGlobal g_global(g_packed + chunk_base * ChunkSize);
      TLOAD(g_ub, g_global);

      if (local_rows > 0) {
        const int32_t token_offset = static_cast<int32_t>(
            seq.token_base_offset +
            (row_start + row_offset) * seq.row_stride);
        KHalfUbDyn k_dyn_ub(local_rows, HiddenSize);
        TASSIGN(k_dyn_ub, KHalfUbAddr);
        DynGlobalHalf k_bsnd_global(
            k_bsnd + token_offset,
            {1, 1, 1, static_cast<int32_t>(local_rows), HiddenSize},
            {1, 1, 1, static_cast<int32_t>(seq.row_stride), 1});
        TLOAD(k_dyn_ub, k_bsnd_global);

        PackedHiddenHalf u_global(
            u_packed + chunk_base * ChunkHiddenElems +
            static_cast<int32_t>(row_offset) * HiddenSize);
        TLOAD(u_ub_half, u_global);
      }
      pipe_barrier(PIPE_ALL);

      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      float g_last_raw =
          g_ub.GetValue(static_cast<int32_t>(valid_rows) - 1);

      if (local_rows > 0) {
        GvUb g_slice;
        TASSIGN(g_slice, GUbAddr + static_cast<int32_t>(row_offset) *
                                       static_cast<int32_t>(sizeof(float)));
        TMOV(g_v_ub, g_slice);
        pipe_barrier(PIPE_V);

        TEXPANDS(coeff_ub, g_last_raw);
        pipe_barrier(PIPE_V);
        TSUB(coeff_ub, coeff_ub, g_v_ub);
        pipe_barrier(PIPE_V);
        TEXP(coeff_ub, coeff_ub);
        pipe_barrier(PIPE_V);

        TCVT(k_ub, k_ub_half, pto::RoundMode::CAST_NONE);
        pipe_barrier(PIPE_V);
        TROWEXPANDMUL(k_ub, k_ub, coeff_col_ub);
        pipe_barrier(PIPE_V);
        TCVT(k_ub_half, k_ub, pto::RoundMode::CAST_NONE);

        TCVT(u_ub, u_ub_half, pto::RoundMode::CAST_NONE);
        pipe_barrier(PIPE_V);
      }

      TEXP(g_ub, g_ub);
      pipe_barrier(PIPE_V);

      GdnWaitCrossFlag(0);
      pipe_barrier(PIPE_ALL);

      if (local_rows > 0) {
        PackedHiddenHalf ws_half_global(
            workspace + ws_kv_base +
            static_cast<int32_t>(row_offset) * HiddenSize);
        TLOAD(u_ub_half, ws_half_global);
        pipe_barrier(PIPE_ALL);
        GdnSetFlag<PIPE_MTE2, PIPE_V>(0);
        GdnWaitFlag<PIPE_MTE2, PIPE_V>(0);

        TCVT(ws_ub, u_ub_half, pto::RoundMode::CAST_NONE);
        pipe_barrier(PIPE_V);
        TSUB(u_ub, u_ub, ws_ub);
        pipe_barrier(PIPE_V);
        TCVT(u_ub_half, u_ub, pto::RoundMode::CAST_NONE);

        GdnSetFlag<PIPE_V, PIPE_MTE3>(0);
        GdnWaitFlag<PIPE_V, PIPE_MTE3>(0);
        PackedHiddenHalf kscaled_ws(
            workspace + kscaled_base +
            static_cast<int32_t>(row_offset) * HiddenSize);
        TSTORE(kscaled_ws, k_ub_half);

        DynGlobalHalf nv_global(
            nv_out + chunk_base * ChunkHiddenElems +
            static_cast<int32_t>(row_offset) * HiddenSize,
            {1, 1, 1, static_cast<int32_t>(local_rows), HiddenSize},
            {1, 1, 1, HiddenSize, 1});
        UHalfUbDyn nv_dyn_ub(local_rows, HiddenSize);
        TASSIGN(nv_dyn_ub, UHalfUbAddr);
        TSTORE(nv_global, nv_dyn_ub);
      }

      pipe_barrier(PIPE_ALL);
      GdnSetCrossFlag<PIPE_MTE3>(1, 2);

      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      float exp_g_last =
          g_ub.GetValue(static_cast<int32_t>(valid_rows) - 1);
      TMULS(s_ub, s_ub, exp_g_last);
      pipe_barrier(PIPE_V);

      GdnWaitCrossFlag(2);
      pipe_barrier(PIPE_ALL);

      PackedHiddenHalf kv_half_global(
          workspace + ws_kv_base +
          static_cast<int32_t>(vid) * HalfChunk * HiddenSize);
      TLOAD(s_ub_half, kv_half_global);
      pipe_barrier(PIPE_ALL);
      GdnSetFlag<PIPE_MTE2, PIPE_V>(1);
      GdnWaitFlag<PIPE_MTE2, PIPE_V>(1);

      TCVT(kv_ub, s_ub_half, pto::RoundMode::CAST_NONE);
      pipe_barrier(PIPE_V);
      TADD(s_ub, s_ub, kv_ub);
      pipe_barrier(PIPE_V);

      TCVT(s_ub_half, s_ub, pto::RoundMode::CAST_NONE);
      GdnSetFlag<PIPE_V, PIPE_MTE3>(1);
      GdnWaitFlag<PIPE_V, PIPE_MTE3>(1);

      if (chunk_idx + 1 < chunk_num) {
        PackedHiddenHalf state_ws(
            workspace + state_base +
            static_cast<int32_t>(vid) * HalfChunk * HiddenSize);
        TSTORE(state_ws, s_ub_half);
        const int32_t next_cb = static_cast<int32_t>(
            (seq.chunk_offset + chunk_idx + 1) * NumHeads + head_idx);
        PackedHiddenHalf s_out_next(
            s_out + next_cb * HiddenSquareElems +
            static_cast<int32_t>(vid) * HalfChunk * HiddenSize);
        TSTORE(s_out_next, s_ub_half);
        pipe_barrier(PIPE_ALL);
        GdnSetCrossFlag<PIPE_MTE3>(3, 2);
      }
    }

    GdnSetFlag<PIPE_V, PIPE_MTE3>(0);
    GdnWaitFlag<PIPE_V, PIPE_MTE3>(0);
    const int32_t fs_base =
        static_cast<int32_t>(seq_idx * NumHeads + head_idx);
    PackedHiddenHalf fs_global(
        fs_out + fs_base * HiddenSquareElems +
        static_cast<int32_t>(vid) * HalfChunk * HiddenSize);
    TSTORE(fs_global, s_ub_half);
    pipe_barrier(PIPE_ALL);
  }
#endif
}

extern "C" __global__ AICORE void launch_chunk_h(
    __gm__ uint8_t *k_bsnd, __gm__ uint8_t *w_packed,
    __gm__ uint8_t *u_packed, __gm__ uint8_t *g_packed,
    __gm__ uint8_t *s_out, __gm__ uint8_t *nv_out,
    __gm__ uint8_t *fs_out, __gm__ uint8_t *workspace,
    __gm__ int32_t *cu_seqlens, int64_t batch_size,
    int64_t fixed_seq_len, uint64_t ffts_addr) {
  chunk_h_main_kernel<GDN_H, GDN_D, GDN_C>(
      reinterpret_cast<__gm__ half *>(k_bsnd),
      reinterpret_cast<__gm__ half *>(w_packed),
      reinterpret_cast<__gm__ half *>(u_packed),
      reinterpret_cast<__gm__ float *>(g_packed),
      reinterpret_cast<__gm__ half *>(s_out),
      reinterpret_cast<__gm__ half *>(nv_out),
      reinterpret_cast<__gm__ half *>(fs_out),
      reinterpret_cast<__gm__ half *>(workspace),
      cu_seqlens, batch_size, fixed_seq_len, ffts_addr);
}

extern "C" void call_kernel(uint32_t blockDim, void *stream,
                            uint8_t *k_bsnd, uint8_t *w_packed,
                            uint8_t *u_packed, uint8_t *g_packed,
                            uint8_t *s_out, uint8_t *nv_out,
                            uint8_t *fs_out, uint8_t *workspace,
                            int32_t *cu_seqlens, int64_t batch_size,
                            int64_t fixed_seq_len) {
  uint32_t ffts_len = 0;
  uint64_t ffts_addr = 0;
  rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
  launch_chunk_h<<<blockDim, nullptr, stream>>>(
      k_bsnd, w_packed, u_packed, g_packed, s_out, nv_out, fs_out,
      workspace, cu_seqlens, batch_size, fixed_seq_len, ffts_addr);
}
