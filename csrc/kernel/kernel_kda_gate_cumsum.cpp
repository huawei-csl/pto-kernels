/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/

// ============================================================================
// kernel_kda_gate_cumsum.cpp — Within-chunk prefix sum of KDA gate vectors
//
// Ported and adapted from
//   huawei-csl/megagdn-pto @ f10b9f2  kernels/pto/gate_cumsum_kda.cpp
//
// Mathematical operation (per chunk of C tokens, per head h, per key-dim d):
//   g_sum[t, h, d] = Σ_{i=0}^{t} g[i, h, d]    for t = 0 .. valid-1
//
// Input:  g     [total_tokens, HV, D]  half    — raw per-dim gate values, BSND
// Output: g_sum [total_tokens, HV, D]  float32 — cumulative sums, BSND
//
// Accumulation is done in fp32 (input g stays fp16 and is cast up before
// accumulating): the per-chunk cumulative sum reaches ~-64 and fp16's
// limited step (~0.06 at that magnitude) would corrupt exp(g_cs) downstream.
//
// Difference from GDN chunk_cumsum (kernel_chunk_cumsum.cpp):
//   GDN: gate shape [T, H], row width = H (~16-64).
//   KDA: gate shape [T, HV, D], re-viewed as [T, HV*D].  Row width = HV*D is
//        ~512-2048, an order of magnitude larger, so we tile along the
//        column (HV*D) dimension.
//
// Template parameters (injected by bisheng at compile time):
//   GDN_H = HV   (number of value/gate heads)
//   GDN_D = D    (key/gate vector dimension per head)
//   GDN_C = C    (chunk size in tokens)
// ============================================================================

#include "kernel_utils.h"

using namespace pto;

#ifndef GDN_H
#define GDN_H 16
#endif

#ifndef GDN_D
#define GDN_D 128
#endif

#ifndef GDN_C
#define GDN_C 128
#endif

// UB tile alias — same convention used by all kernels in this repo.
#ifdef __CCE_AICORE__
template <typename T, int R, int C, int RV = R, int CV = C,
          pto::PadValue P = pto::PadValue::Null>
using UbND = pto::Tile<pto::TileType::Vec, T, R, C, pto::BLayout::RowMajor, RV,
                       CV, pto::SLayout::NoneBox, 512, P>;
#endif

template <int32_t NumHeads, int32_t KDim, int32_t ChunkSize>
AICORE inline void kda_gate_cumsum_kernel(__gm__ half* g_ptr,
                                          __gm__ float* g_sum_ptr,
                                          __gm__ int32_t* cu_seqlens,
                                          int64_t batch_size, int64_t seq_len) {
  using kernel_utils::PipeBarrierVec;
  using pto::Stride;
  auto cid = get_block_idx();
  auto block_num = get_block_num();
  auto vid = get_subblockid();

#if defined(__DAV_VEC__)
  if (vid != 0) return;

  set_mask_norm();
  set_vector_mask(-1, -1);

  // Flat 2D view of g: [total_tokens, HV*D] (row-major, contiguous).
  constexpr int32_t RowWidth = NumHeads * KDim;

  // ColTile: number of HV*D columns processed in one UB-resident slice.
  // 128 is safe for ChunkSize up to 128 (per-tile UB ≈ 66 KB) and divides
  // RowWidth = HV*KDim whenever KDim is a multiple of 128 (the typical case).
  constexpr int32_t ColTileTarget = 128;
  constexpr int32_t ColTile =
      (RowWidth < ColTileTarget) ? RowWidth : ColTileTarget;
  constexpr int32_t CTC = ((ColTile + 7) / 8) * 8;  // 8-elem alignment
  static_assert(RowWidth % ColTile == 0,
                "RowWidth (HV*KDim) must be a multiple of ColTile (128). "
                "Reduce ColTileTarget or pick HV/KDim values whose product "
                "is divisible by it.");
  constexpr int32_t NumColTiles = RowWidth / ColTile;

  // Per-tile UB layout:
  //   [0         .. HalfBlockBytes)   fp16 staging  (ChunkSize × CTC half)
  //   [GUbAddr   .. +BlockBytes)      fp32 input    (ChunkSize × CTC float)
  //   [SUbAddr   .. +BlockBytes)      fp32 output   (ChunkSize × CTC float)
  //   [AccUbAddr .. +CTC*4)           row accumulator (1 × CTC float)
  constexpr int32_t HalfBlockBytes =
      ChunkSize * CTC * static_cast<int32_t>(sizeof(half));
  constexpr int32_t BlockBytes =
      ChunkSize * CTC * static_cast<int32_t>(sizeof(float));
  constexpr int32_t RowBytes = CTC * static_cast<int32_t>(sizeof(float));
  constexpr int32_t GHalfAddr = 0;
  constexpr int32_t GUbAddr = HalfBlockBytes;
  constexpr int32_t SUbAddr = GUbAddr + BlockBytes;
  constexpr int32_t AccUbAddr = SUbAddr + BlockBytes;

  // Strided 2D GM views — row stride = RowWidth (full BSND row), column
  // window = ColTile per load.
  using GmShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using GmStride = Stride<1, 1, 1, RowWidth, 1>;
  using GmHalf = GlobalTensor<half, GmShape, GmStride>;
  using GmFloat = GlobalTensor<float, GmShape, GmStride>;

  UbND<float, 1, CTC> acc_ub;
  TASSIGN(acc_ub, AccUbAddr);

  int64_t num_seqs = batch_size;

  // ── Fixed-length sequence path (cu_seqlens == nullptr) ────────────────────
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
      int32_t valid =
          static_cast<int32_t>(remaining < ChunkSize ? remaining : ChunkSize);

      for (int32_t ct = 0; ct < NumColTiles; ++ct) {
        int32_t col_off = ct * ColTile;

        // MTE2: load g[chunk_start..+valid, col_off..+ColTile] (fp16)
        {
          GmShape gs;
          gs.shape[3] = valid;
          gs.shape[4] = ColTile;
          GmHalf g_gm(g_ptr + chunk_start * RowWidth + col_off, gs);
          UbND<half, ChunkSize, CTC, DYNAMIC, DYNAMIC, PadValue::Zero> g_load(
              valid, ColTile);
          TASSIGN(g_load, GHalfAddr);
          TLOAD(g_load, g_gm);
          if (valid != ChunkSize || ColTile != CTC) {
            UbND<half, ChunkSize, CTC, ChunkSize, CTC, PadValue::Zero> g_pad;
            TASSIGN(g_pad, GHalfAddr);
            TFILLPAD_INPLACE(g_pad, g_load);
          }
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        // Cast g fp16 → fp32 (accumulate in fp32 to avoid precision loss).
        {
          UbND<half, ChunkSize, CTC> g_h;
          TASSIGN(g_h, GHalfAddr);
          UbND<float, ChunkSize, CTC> g_f;
          TASSIGN(g_f, GUbAddr);
          TCVT(g_f, g_h, pto::RoundMode::CAST_NONE);
          PipeBarrierVec();
        }

        // Vec: prefix sum (all ColTile cols in parallel).
        // Row 0: acc = g[0]; g_sum[0] = acc
        UbND<float, 1, CTC> g_row_0;
        TASSIGN(g_row_0, GUbAddr);
        TMOV(acc_ub, g_row_0);
        PipeBarrierVec();

        UbND<float, 1, CTC> s_row_0;
        TASSIGN(s_row_0, SUbAddr);
        TMOV(s_row_0, acc_ub);
        PipeBarrierVec();

        // Rows 1..valid-1: acc += g[i]; g_sum[i] = acc
        for (int32_t i = 1; i < valid; ++i) {
          UbND<float, 1, CTC> g_row_i;
          TASSIGN(g_row_i, GUbAddr + i * RowBytes);
          TADD(acc_ub, acc_ub, g_row_i);
          PipeBarrierVec();

          UbND<float, 1, CTC> s_row_i;
          TASSIGN(s_row_i, SUbAddr + i * RowBytes);
          TMOV(s_row_i, acc_ub);
          PipeBarrierVec();
        }

        // V → MTE2: prevent next iteration's TLOAD from clobbering UB before
        // Vec has finished reading the current iteration's input.
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

        // MTE3: store g_sum (rows 0..valid-1 only)
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        {
          GmShape ss;
          ss.shape[3] = valid;
          ss.shape[4] = ColTile;
          GmFloat gs_gm(g_sum_ptr + chunk_start * RowWidth + col_off, ss);
          UbND<float, ChunkSize, CTC, DYNAMIC, DYNAMIC> s_store(valid, ColTile);
          TASSIGN(s_store, SUbAddr);
          TSTORE(gs_gm, s_store);
        }
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      }
    }
  }
  // ── Variable-length sequence path (cu_seqlens != nullptr) ─────────────────
  else {
    int64_t gi = 0;
    for (int64_t si = 0; si < num_seqs; ++si) {
      int64_t bos = static_cast<int64_t>(cu_seqlens[si]);
      int64_t eos = static_cast<int64_t>(cu_seqlens[si + 1]);
      int64_t slen = eos - bos;
      int64_t nc = (slen + ChunkSize - 1) / ChunkSize;

      for (int64_t c = 0; c < nc; ++c) {
        if (gi % static_cast<int64_t>(block_num) == static_cast<int64_t>(cid)) {
          int64_t chunk_start = bos + c * ChunkSize;
          int64_t remaining = slen - c * ChunkSize;
          int32_t valid = static_cast<int32_t>(
              remaining < ChunkSize ? remaining : ChunkSize);

          for (int32_t ct = 0; ct < NumColTiles; ++ct) {
            int32_t col_off = ct * ColTile;

            {
              GmShape gs;
              gs.shape[3] = valid;
              gs.shape[4] = ColTile;
              GmHalf g_gm(g_ptr + chunk_start * RowWidth + col_off, gs);
              UbND<half, ChunkSize, CTC, DYNAMIC, DYNAMIC, PadValue::Zero>
                  g_load(valid, ColTile);
              TASSIGN(g_load, GHalfAddr);
              TLOAD(g_load, g_gm);
              if (valid != ChunkSize || ColTile != CTC) {
                UbND<half, ChunkSize, CTC, ChunkSize, CTC, PadValue::Zero>
                    g_pad;
                TASSIGN(g_pad, GHalfAddr);
                TFILLPAD_INPLACE(g_pad, g_load);
              }
            }
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            {
              UbND<half, ChunkSize, CTC> g_h;
              TASSIGN(g_h, GHalfAddr);
              UbND<float, ChunkSize, CTC> g_f;
              TASSIGN(g_f, GUbAddr);
              TCVT(g_f, g_h, pto::RoundMode::CAST_NONE);
              PipeBarrierVec();
            }

            UbND<float, 1, CTC> g_row_0;
            TASSIGN(g_row_0, GUbAddr);
            TMOV(acc_ub, g_row_0);
            PipeBarrierVec();

            UbND<float, 1, CTC> s_row_0;
            TASSIGN(s_row_0, SUbAddr);
            TMOV(s_row_0, acc_ub);
            PipeBarrierVec();

            for (int32_t i = 1; i < valid; ++i) {
              UbND<float, 1, CTC> g_row_i;
              TASSIGN(g_row_i, GUbAddr + i * RowBytes);
              TADD(acc_ub, acc_ub, g_row_i);
              PipeBarrierVec();

              UbND<float, 1, CTC> s_row_i;
              TASSIGN(s_row_i, SUbAddr + i * RowBytes);
              TMOV(s_row_i, acc_ub);
              PipeBarrierVec();
            }

            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

            {
              GmShape ss;
              ss.shape[3] = valid;
              ss.shape[4] = ColTile;
              GmFloat gs_gm(g_sum_ptr + chunk_start * RowWidth + col_off, ss);
              UbND<float, ChunkSize, CTC, DYNAMIC, DYNAMIC> s_store(valid,
                                                                    ColTile);
              TASSIGN(s_store, SUbAddr);
              TSTORE(gs_gm, s_store);
            }
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
          }
        }
        gi++;
      }
    }
  }
#endif  // __DAV_VEC__
}

extern "C" __global__ AICORE void kda_gate_cumsum(__gm__ uint8_t* g_ptr,
                                                  __gm__ uint8_t* g_sum_ptr,
                                                  __gm__ uint8_t* cu_seqlens,
                                                  int64_t batch_size,
                                                  int64_t seq_len) {
#if defined(__DAV_VEC__)
  kda_gate_cumsum_kernel<GDN_H, GDN_D, GDN_C>(
      reinterpret_cast<__gm__ half*>(g_ptr),
      reinterpret_cast<__gm__ float*>(g_sum_ptr),
      reinterpret_cast<__gm__ int32_t*>(cu_seqlens), batch_size, seq_len);
#endif
}
