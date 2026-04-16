// ============================================================================
// chunk_cumsum_kernel.cpp — Prefix sum of gate values G along time dimension
//
// Mathematical operation (per chunk of C tokens, independently per head h):
//   g_sum[t, h] = Σ_{i=0}^{t} g[i, h]    for t = 0 .. valid-1
//
// Input:  g     [total_tokens, H]  float, BSND layout  — raw gate values
// Output: g_sum [total_tokens, H]  float               — cumulative sums
//
// The prefix sum enables downstream kernels to compute exponential decay
// coefficients:  exp(g_sum[i] - g_sum[j])  gives the cumulative gate
// from token j to token i within a chunk.
//
// Architecture: Vec-only kernel (no Cube/GEMM). Single Vec sub-block.
// Pipeline: MTE2(load) → Vec(compute) → MTE3(store), serialized per chunk.
//
// NPU memory hierarchy used:
//   GM (Global Memory) → UB (Unified Buffer, on-chip SRAM, Vec-accessible)
// ============================================================================

#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace pto;

#ifndef GDN_H
#define GDN_H 16
#endif

#ifndef GDN_C
#define GDN_C 128
#endif

// ── PTO type aliases (device-only, guarded by __CCE_AICORE__) ───────────────
// UB tile in row-major (ND) layout, used by Vec engine.
// T=dtype, R×C=static shape, RV×CV=valid region, P=pad value for TLOAD.
#ifdef __CCE_AICORE__
template <typename T, int R, int C, int RV = R, int CV = C,
          pto::PadValue P = pto::PadValue::Null>
using UbND = pto::Tile<pto::TileType::Vec, T, R, C, pto::BLayout::RowMajor,
                       RV, CV, pto::SLayout::NoneBox, 512, P>;
#endif

template <int32_t NumHeads, int32_t ChunkSize>
AICORE void cumsum_kernel(
    __gm__ float *g_ptr, __gm__ float *g_sum_ptr,
    __gm__ int32_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len,
    uint64_t ffts_addr)
{
  auto cid = get_block_idx();
  auto block_num = get_block_num();
  auto vid = get_subblockid();
  set_ffts_base_addr(ffts_addr);

#if defined(__DAV_C220_VEC__)
  if (vid != 0) return;

  set_mask_norm();
  set_vector_mask(-1, -1);

  // HeadTileCols: NumHeads rounded up to 8-element alignment (32B for float)
  constexpr int32_t HTC = ((NumHeads + 7) / 8) * 8;
  constexpr int32_t BlockBytes = ChunkSize * HTC *
                                 static_cast<int32_t>(sizeof(float));
  constexpr int32_t RowBytes = HTC * static_cast<int32_t>(sizeof(float));

  // ── UB memory layout ──────────────────────────────────────────────────
  //  [0            .. BlockBytes)     = g input  (ChunkSize × HTC floats)
  //  [BlockBytes   .. 2*BlockBytes)   = g_sum output
  //  [2*BlockBytes .. 2*BlockBytes+RowBytes) = row accumulator (1 × HTC)
  constexpr int32_t GUbAddr   = 0;
  constexpr int32_t SUbAddr   = BlockBytes;
  constexpr int32_t AccUbAddr = BlockBytes * 2;

  // GlobalTensor types for g/g_sum in [total_tokens, NumHeads] layout.
  // 5D shape with last two dims dynamic; stride encodes row pitch.
  using GmShape  = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using GmStride = Stride<1, 1, 1, NumHeads, 1>;
  using GmFloat  = GlobalTensor<float, GmShape, GmStride>;

  // Pre-assign row accumulator at fixed UB address
  UbND<float, 1, HTC> acc_ub;
  TASSIGN(acc_ub, AccUbAddr);

  int64_t num_seqs = batch_size;

  // ── Fixed-length sequence path (cu_seqlens == nullptr) ────────────────
  if (cu_seqlens == nullptr) {
    int64_t chunks_per_seq = (seq_len + ChunkSize - 1) / ChunkSize;
    int64_t total_chunks = num_seqs * chunks_per_seq;

    for (int64_t gi = static_cast<int64_t>(cid); gi < total_chunks;
         gi += static_cast<int64_t>(block_num)) {
      int64_t seq_idx = gi / chunks_per_seq;
      int64_t local_chunk = gi % chunks_per_seq;
      int64_t bos = seq_idx * seq_len;
      int64_t chunk_start = bos + local_chunk * ChunkSize;
      int64_t remaining = seq_len - local_chunk * ChunkSize;
      int32_t valid = static_cast<int32_t>(
          remaining < ChunkSize ? remaining : ChunkSize);

      // ── DMA: load g[chunk_start .. +valid] from GM → UB (MTE2 pipe) ──
      // Constructs a GlobalTensor view over the g array, loads into UB,
      // then zero-pads the tail region (rows beyond `valid`, cols beyond
      // NumHeads up to the 8-aligned HTC) so downstream Vec ops see zeros.
      {
        GmShape gs; gs.shape[3] = valid; gs.shape[4] = NumHeads;
        GmFloat g_gm(g_ptr + chunk_start * NumHeads, gs);
        UbND<float, ChunkSize, HTC, DYNAMIC, DYNAMIC, PadValue::Zero>
            g_load(valid, NumHeads);
        TASSIGN(g_load, GUbAddr);
        TLOAD(g_load, g_gm);
        if (valid != ChunkSize || NumHeads != HTC) {
          UbND<float, ChunkSize, HTC, ChunkSize, HTC, PadValue::Zero> g_pad;
          TASSIGN(g_pad, GUbAddr);
          TFILLPAD_INPLACE(g_pad, g_load);
        }
      }
      // MTE2 → Vec sync: wait for DMA load to finish before Vec reads UB
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

      // ── Vec compute: prefix sum over rows (all H heads in parallel) ───
      // Row 0: acc[h] = g[0,h];  g_sum[0,h] = acc[h]
      UbND<float, 1, HTC> g_row_0;
      TASSIGN(g_row_0, GUbAddr);
      TMOV(acc_ub, g_row_0);
      pipe_barrier(PIPE_V);

      UbND<float, 1, HTC> s_row_0;
      TASSIGN(s_row_0, SUbAddr);
      TMOV(s_row_0, acc_ub);
      pipe_barrier(PIPE_V);

      // Rows 1..valid-1:  acc[h] += g[i,h];  g_sum[i,h] = acc[h]
      for (int32_t i = 1; i < valid; ++i) {
        UbND<float, 1, HTC> g_row_i;
        TASSIGN(g_row_i, GUbAddr + i * RowBytes);
        TADD(acc_ub, acc_ub, g_row_i);
        pipe_barrier(PIPE_V);

        UbND<float, 1, HTC> s_row_i;
        TASSIGN(s_row_i, SUbAddr + i * RowBytes);
        TMOV(s_row_i, acc_ub);
        pipe_barrier(PIPE_V);
      }

      // Zero-fill rows beyond valid (tail padding for downstream kernels)
      TEXPANDS(acc_ub, 0.0f);
      pipe_barrier(PIPE_V);
      for (int32_t i = valid; i < ChunkSize; ++i) {
        UbND<float, 1, HTC> s_row_i;
        TASSIGN(s_row_i, SUbAddr + i * RowBytes);
        TMOV(s_row_i, acc_ub);
        pipe_barrier(PIPE_V);
      }

      // ── DMA: store g_sum from UB → GM (MTE3 pipe) ────────────────────
      // Vec → MTE3 sync: ensure Vec writes to UB are visible before DMA
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

      {
        GmShape ss; ss.shape[3] = valid; ss.shape[4] = NumHeads;
        GmFloat gs_gm(g_sum_ptr + chunk_start * NumHeads, ss);
        UbND<float, ChunkSize, HTC, DYNAMIC, DYNAMIC>
            s_store(valid, NumHeads);
        TASSIGN(s_store, SUbAddr);
        TSTORE(gs_gm, s_store);
      }
      // MTE3 → Vec sync: wait for DMA store before reusing UB next iter
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }
  }
  // ── Variable-length sequence path (cu_seqlens != nullptr) ─────────────
  else {
    int64_t gi = 0;
    for (int64_t si = 0; si < num_seqs; ++si) {
      int64_t bos = static_cast<int64_t>(cu_seqlens[si]);
      int64_t eos = static_cast<int64_t>(cu_seqlens[si + 1]);
      int64_t slen = eos - bos;
      int64_t nc = (slen + ChunkSize - 1) / ChunkSize;

      for (int64_t c = 0; c < nc; ++c) {
        if (gi % static_cast<int64_t>(block_num) ==
            static_cast<int64_t>(cid)) {
          int64_t chunk_start = bos + c * ChunkSize;
          int64_t remaining = slen - c * ChunkSize;
          int32_t valid = static_cast<int32_t>(
              remaining < ChunkSize ? remaining : ChunkSize);

          // Load g chunk from GM → UB, zero-padded
          {
            GmShape gs; gs.shape[3] = valid; gs.shape[4] = NumHeads;
            GmFloat g_gm(g_ptr + chunk_start * NumHeads, gs);
            UbND<float, ChunkSize, HTC, DYNAMIC, DYNAMIC, PadValue::Zero>
                g_load(valid, NumHeads);
            TASSIGN(g_load, GUbAddr);
            TLOAD(g_load, g_gm);
            if (valid != ChunkSize || NumHeads != HTC) {
              UbND<float, ChunkSize, HTC, ChunkSize, HTC, PadValue::Zero>
                  g_pad;
              TASSIGN(g_pad, GUbAddr);
              TFILLPAD_INPLACE(g_pad, g_load);
            }
          }
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

          // Prefix sum: acc = g[0]; g_sum[0] = acc
          UbND<float, 1, HTC> g_row_0;
          TASSIGN(g_row_0, GUbAddr);
          TMOV(acc_ub, g_row_0);
          pipe_barrier(PIPE_V);

          UbND<float, 1, HTC> s_row_0;
          TASSIGN(s_row_0, SUbAddr);
          TMOV(s_row_0, acc_ub);
          pipe_barrier(PIPE_V);

          // acc += g[i]; g_sum[i] = acc
          for (int32_t i = 1; i < valid; ++i) {
            UbND<float, 1, HTC> g_row_i;
            TASSIGN(g_row_i, GUbAddr + i * RowBytes);
            TADD(acc_ub, acc_ub, g_row_i);
            pipe_barrier(PIPE_V);

            UbND<float, 1, HTC> s_row_i;
            TASSIGN(s_row_i, SUbAddr + i * RowBytes);
            TMOV(s_row_i, acc_ub);
            pipe_barrier(PIPE_V);
          }

          // Zero-fill padding rows
          TEXPANDS(acc_ub, 0.0f);
          pipe_barrier(PIPE_V);
          for (int32_t i = valid; i < ChunkSize; ++i) {
            UbND<float, 1, HTC> s_row_i;
            TASSIGN(s_row_i, SUbAddr + i * RowBytes);
            TMOV(s_row_i, acc_ub);
            pipe_barrier(PIPE_V);
          }

          // Store g_sum to GM
          set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
          wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

          {
            GmShape ss; ss.shape[3] = valid; ss.shape[4] = NumHeads;
            GmFloat gs_gm(g_sum_ptr + chunk_start * NumHeads, ss);
            UbND<float, ChunkSize, HTC, DYNAMIC, DYNAMIC>
                s_store(valid, NumHeads);
            TASSIGN(s_store, SUbAddr);
            TSTORE(gs_gm, s_store);
          }
          set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        }
        gi++;
      }
    }
  }
#endif
}

extern "C" __global__ AICORE void launch_cumsum(
    __gm__ uint8_t *g_ptr, __gm__ uint8_t *g_sum_ptr,
    __gm__ uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len,
    uint64_t ffts_addr)
{
  cumsum_kernel<GDN_H, GDN_C>(
      reinterpret_cast<__gm__ float *>(g_ptr),
      reinterpret_cast<__gm__ float *>(g_sum_ptr),
      reinterpret_cast<__gm__ int32_t *>(cu_seqlens),
      batch_size, seq_len, ffts_addr);
}

extern "C" void call_kernel(
    uint32_t block_dim, void *stream,
    uint8_t *g_ptr, uint8_t *g_sum_ptr, uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len)
{
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  launch_cumsum<<<block_dim, nullptr, stream>>>(
      g_ptr, g_sum_ptr, cu_seqlens, batch_size, seq_len, fftsAddr);
}
