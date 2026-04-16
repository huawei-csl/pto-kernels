// ============================================================================
// wy_fast_kernel.cpp — WY representation for GatedDeltaNet chunk recurrence
//
// Computes the WY update matrices U and W for each chunk of C tokens:
//   U = A2 @ V     where A2 = A * beta_2d        (beta-scaled attention)
//   W = A1 @ K     where A1 = A * (exp(g)*beta)_2d (gate+beta-scaled attention)
//
// beta is the decay factor, g is the gate value, A is the triangular attention
// matrix (from the kkt kernel).  The column-broadcast notation x_2d means
// expanding a 1xC vector into a C/2 x C matrix by replicating across rows.
//
// Architecture: Vec+Cube cooperative kernel using cross-core synchronization.
//
//  Vec core (two sub-blocks for upper/lower C/2 rows):
//    For each chunk:
//      1. Load beta [H,T] and A [B,S,H,C], compute A2 = A * beta_2d -> ws
//      2. Load G [H,T], compute A1 = A * (exp(g)*beta)_2d -> ws
//      3. Signal Cube via cross-core flags when workspaces are ready
//
//  Cube core (waits for Vec signals):
//    For each chunk:
//      1. Load K, V from BSND layout into L1
//      2. Load A2 from workspace -> GEMM: U = A2 @ V
//      3. Load A1 from workspace -> GEMM: W = A1 @ K
//      4. Store U, W back to BSND layout
//
// NPU memory hierarchy used:
//   GM -> UB (Vec), GM -> L1 -> L0A/L0B -> L0C -> GM (Cube)
// ============================================================================

#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
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

// ── PTO type aliases (device-only, guarded by __CCE_AICORE__) ───────────────
// UB tile in row-major (ND) layout, used by Vec engine.
// T=dtype, R×C=static shape, RV×CV=valid region, P=pad value for TLOAD.
#ifdef __CCE_AICORE__
template <typename T, int R, int C, int RV = R, int CV = C,
          pto::PadValue P = pto::PadValue::Null>
using UbND = pto::Tile<pto::TileType::Vec, T, R, C, pto::BLayout::RowMajor,
                       RV, CV, pto::SLayout::NoneBox, 512, P>;

// L1 tile in column-major (NZ) layout, used as input to Cube engine.
// T=dtype, R×C=static shape, RV×CV=valid region. Zero-padded on TLOAD.
template <typename T, int R, int C, int RV = R, int CV = C>
using L1Mat = pto::Tile<pto::TileType::Mat, T, R, C, pto::BLayout::ColMajor,
                        RV, CV, pto::SLayout::RowMajor, 512, pto::PadValue::Zero>;
#endif

template <int32_t NumHeads, int32_t HiddenSize, int32_t ChunkSize>
AICORE void wy_fast_kernel(
    __gm__ half *K_handle, __gm__ half *V_handle,
    __gm__ half *Beta_handle, __gm__ float *G_handle,
    __gm__ half *A_handle,
    __gm__ half *workspace_a1_handle, __gm__ half *workspace_a2_handle,
    __gm__ half *W_handle, __gm__ half *U_handle,
    __gm__ int32_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len,
    int64_t total_tokens,
    uint64_t ffts_addr)
{
  constexpr int32_t HalfChunk = ChunkSize / 2;
  constexpr uint32_t KTail =
      (HiddenSize % 128 == 0) ? 128 : (HiddenSize % 128);

  // ── UB memory layout (byte addresses, Vec engine) ─────────────────────
  constexpr int32_t BetaHalfUbAddr = 0;
  constexpr int32_t A1HalfUbAddr   = 256;
  constexpr int32_t BetaUbAddr     = 16640;
  constexpr int32_t BetaRUbAddr    = 17152;
  constexpr int32_t Beta2dUbAddr   = 17664;
  constexpr int32_t TmpUbAddr      = 50432;
  constexpr int32_t A1UbAddr       = 75008;
  constexpr int32_t A2UbAddr       = 107776;
  constexpr int32_t A2HalfUbAddr   = 140544;
  constexpr int32_t GUbAddr        = 156928;
  constexpr int32_t GRUbAddr       = 157440;
  constexpr int32_t G2dUbAddr      = 157952;

  constexpr int32_t WsA1Size = ChunkSize * ChunkSize;
  constexpr int32_t WsA2Size = ChunkSize * ChunkSize;

  set_ffts_base_addr(ffts_addr);
  auto cid = get_block_idx();
  auto block_num = get_block_num();
  auto vid = get_subblockid();

  int64_t num_seqs = batch_size;

  // ── UB tile declarations (Vec sub-blocks) ─────────────────────────────
  UbND<half, 1, ChunkSize> beta_ub_half;
  TASSIGN(beta_ub_half, BetaHalfUbAddr);
  UbND<half, HalfChunk, ChunkSize> a1_ub_half;
  TASSIGN(a1_ub_half, A1HalfUbAddr);
  UbND<float, 1, ChunkSize> beta_ub;
  TASSIGN(beta_ub, BetaUbAddr);
  UbND<float, 1, ChunkSize> beta_r_ub;
  TASSIGN(beta_r_ub, BetaRUbAddr);
  UbND<float, HalfChunk, ChunkSize> beta_2d_ub;
  TASSIGN(beta_2d_ub, Beta2dUbAddr);
  UbND<uint8_t, 1, 24576> tmp_ub;
  TASSIGN(tmp_ub, TmpUbAddr);
  UbND<float, HalfChunk, ChunkSize> a1_ub;
  TASSIGN(a1_ub, A1UbAddr);
  UbND<float, HalfChunk, ChunkSize> a2_ub;
  TASSIGN(a2_ub, A2UbAddr);
  UbND<half, HalfChunk, ChunkSize> a2_ub_half;
  TASSIGN(a2_ub_half, A2HalfUbAddr);
  UbND<float, 1, ChunkSize> g_ub;
  TASSIGN(g_ub, GUbAddr);
  UbND<float, 1, ChunkSize> g_r_ub;
  TASSIGN(g_r_ub, GRUbAddr);
  UbND<float, HalfChunk, ChunkSize> g_2d_ub;
  TASSIGN(g_2d_ub, G2dUbAddr);

  // ── L1 / L0C tile declarations (Cube engine) ─────────────────────────
  L1Mat<half, ChunkSize, HiddenSize> k_l1;
  TASSIGN(k_l1, 0);
  L1Mat<half, ChunkSize, HiddenSize> v_l1;
  TASSIGN(v_l1, 32768);
  L1Mat<half, ChunkSize, ChunkSize> a2_l1;
  TASSIGN(a2_l1, 65536);
  TileAcc<float, ChunkSize, HiddenSize,
          ChunkSize, HiddenSize> u_l0;
  TASSIGN(u_l0, 0);
  L1Mat<half, ChunkSize, ChunkSize> a1_l1;
  TASSIGN(a1_l1, 98304);
  TileAcc<float, ChunkSize, HiddenSize,
          ChunkSize, HiddenSize> w_l0;
  TASSIGN(w_l0, 65536);

  int64_t total_work = 0;
  if (cu_seqlens == nullptr) {
    int64_t chunks_per_seq = (seq_len + ChunkSize - 1) / ChunkSize;
    total_work = num_seqs * chunks_per_seq * NumHeads;
  }

  // ════════════════════════════════════════════════════════════════════════
  // Vec phase: compute A2 = A*beta_2d and A1 = A*(exp(g)*beta)_2d
  // Two Vec sub-blocks (vid=0,1) handle upper/lower C/2 rows in parallel.
  // ════════════════════════════════════════════════════════════════════════
#if defined(__DAV_C220_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);

  // ── Fixed-length sequence path ────────────────────────────────────────
  if (cu_seqlens == nullptr) {
    int64_t chunks_per_seq = (seq_len + ChunkSize - 1) / ChunkSize;
    bool first_iter = true;
    for (int64_t work_idx = static_cast<int64_t>(cid);
         work_idx < total_work;
         work_idx += static_cast<int64_t>(block_num)) {
      int32_t head_idx = static_cast<int32_t>(work_idx % NumHeads);
      int64_t chunk_head_idx = work_idx / NumHeads;
      int64_t seq_idx = chunk_head_idx / chunks_per_seq;
      int64_t ci = chunk_head_idx % chunks_per_seq;

      int64_t bos = seq_idx * seq_len;
      int64_t slen = seq_len;
      int64_t chunk_start = ci * ChunkSize;
      int64_t remaining = slen - chunk_start;
      int32_t valid_rows = static_cast<int32_t>(
          remaining < ChunkSize ? remaining : ChunkSize);
      int64_t chunk_token_start = bos + chunk_start;

      // Load beta (pre-transposed [H, total_tokens]) -> UB, zero-pad tail
      {
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = 1; _gs.shape[4] = valid_rows;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, 1, 1>> _gm(
            Beta_handle + static_cast<int64_t>(head_idx) * total_tokens
                        + chunk_token_start, _gs);
        UbND<half, 1, ChunkSize, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(1, valid_rows);
        TASSIGN(_ld, BetaHalfUbAddr);
        TLOAD(_ld, _gm);
        if (valid_rows != ChunkSize) {
          UbND<half, 1, ChunkSize, 1, ChunkSize, PadValue::Zero> _pd;
          TASSIGN(_pd, BetaHalfUbAddr);
          TFILLPAD_INPLACE(_pd, _ld);
        }
      }

      // Load A [B,S,H,C] — this sub-block's C/2 rows
      int64_t a_gm_offset =
          ((chunk_token_start +
            static_cast<int64_t>(vid) * HalfChunk) *
           NumHeads + head_idx) *
          static_cast<int64_t>(ChunkSize);
      {
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = HalfChunk; _gs.shape[4] = ChunkSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, NumHeads * ChunkSize, 1>> _gm(
            A_handle + a_gm_offset, _gs);
        UbND<half, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(HalfChunk, ChunkSize);
        TASSIGN(_ld, A1HalfUbAddr);
        TLOAD(_ld, _gm);
      }

      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

      // A2 = A * beta_2d: column-broadcast beta then elementwise multiply
      TCVT(beta_ub, beta_ub_half, pto::RoundMode::CAST_NONE);
      pipe_barrier(PIPE_V);
      TMOV(beta_r_ub, beta_ub);
      pipe_barrier(PIPE_V);
      TCOLEXPAND(beta_2d_ub, beta_r_ub);

      TCVT(a1_ub, a1_ub_half, pto::RoundMode::CAST_NONE);
      TMUL(a2_ub, a1_ub, beta_2d_ub);
      TCVT(a2_ub_half, a2_ub, pto::RoundMode::CAST_NONE);

      // Store A2 -> workspace GM, signal Cube (cross-core flag 2)
      if (!first_iter) wait_flag_dev(3);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      {
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = HalfChunk; _gs.shape[4] = ChunkSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>> _gm(
            workspace_a2_handle +
                static_cast<int64_t>(cid) * WsA2Size +
                static_cast<int64_t>(vid) * HalfChunk * ChunkSize, _gs);
        UbND<half, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC> _st(HalfChunk, ChunkSize);
        TASSIGN(_st, A2HalfUbAddr);
        TSTORE(_gm, _st);
      }
      ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (2 << 8));

      // Load G (pre-transposed [H, total_tokens]) -> UB, zero-pad tail
      {
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = 1; _gs.shape[4] = valid_rows;
        GlobalTensor<float, decltype(_gs), Stride<1, 1, 1, 1, 1>> _gm(
            G_handle + static_cast<int64_t>(head_idx) * total_tokens
                     + chunk_token_start, _gs);
        UbND<float, 1, ChunkSize, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(1, valid_rows);
        TASSIGN(_ld, GUbAddr);
        TLOAD(_ld, _gm);
        if (valid_rows != ChunkSize) {
          UbND<float, 1, ChunkSize, 1, ChunkSize, PadValue::Zero> _pd;
          TASSIGN(_pd, GUbAddr);
          TFILLPAD_INPLACE(_pd, _ld);
        }
      }

      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

      // A1 = A * (exp(g) * beta)_2d: gate modulation before column-broadcast
      TEXP(g_ub, g_ub);
      pipe_barrier(PIPE_V);
      TMUL(g_ub, g_ub, beta_ub);
      pipe_barrier(PIPE_V);
      TMOV(g_r_ub, g_ub);
      pipe_barrier(PIPE_V);
      TCOLEXPAND(g_2d_ub, g_r_ub);
      TMUL(a1_ub, a1_ub, g_2d_ub);
      TCVT(a1_ub_half, a1_ub, pto::RoundMode::CAST_NONE);

      // Store A1 -> workspace GM, signal Cube (cross-core flag 1)
      if (!first_iter) wait_flag_dev(4);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      {
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = HalfChunk; _gs.shape[4] = ChunkSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>> _gm(
            workspace_a1_handle +
                static_cast<int64_t>(cid) * WsA1Size +
                static_cast<int64_t>(vid) * HalfChunk * ChunkSize, _gs);
        UbND<half, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC> _st(HalfChunk, ChunkSize);
        TASSIGN(_st, A1HalfUbAddr);
        TSTORE(_gm, _st);
      }
      ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (1 << 8));
      first_iter = false;
    }
  }
  // ── Variable-length sequence path (Vec) ───────────────────────────────
  else {
    int64_t gi = 0;
    bool first_iter_v = true;
    for (int64_t si = 0; si < num_seqs; ++si) {
      int64_t bos = static_cast<int64_t>(cu_seqlens[si]);
      int64_t eos = static_cast<int64_t>(cu_seqlens[si + 1]);
      int64_t slen = eos - bos;
      int64_t nc = (slen + ChunkSize - 1) / ChunkSize;

      for (int64_t ci = 0; ci < nc; ++ci) {
        for (int32_t h = 0; h < NumHeads; ++h) {
          if (gi % static_cast<int64_t>(block_num) ==
              static_cast<int64_t>(cid)) {
            int64_t chunk_start = ci * ChunkSize;
            int64_t remaining = slen - chunk_start;
            int32_t valid_rows = static_cast<int32_t>(
                remaining < ChunkSize ? remaining : ChunkSize);
            int64_t chunk_token_start = bos + chunk_start;
            int32_t head_idx = h;

            // Load beta -> UB
            {
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = 1; _gs.shape[4] = valid_rows;
              GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, 1, 1>> _gm(
                  Beta_handle + static_cast<int64_t>(head_idx) * total_tokens
                              + chunk_token_start, _gs);
              UbND<half, 1, ChunkSize, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(1, valid_rows);
              TASSIGN(_ld, BetaHalfUbAddr);
              TLOAD(_ld, _gm);
              if (valid_rows != ChunkSize) {
                UbND<half, 1, ChunkSize, 1, ChunkSize, PadValue::Zero> _pd;
                TASSIGN(_pd, BetaHalfUbAddr);
                TFILLPAD_INPLACE(_pd, _ld);
              }
            }

            // Load A -> UB
            int64_t a_gm_offset =
                ((chunk_token_start +
                  static_cast<int64_t>(vid) * HalfChunk) *
                 NumHeads + head_idx) *
                static_cast<int64_t>(ChunkSize);
            {
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = HalfChunk; _gs.shape[4] = ChunkSize;
              GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, NumHeads * ChunkSize, 1>> _gm(
                  A_handle + a_gm_offset, _gs);
              UbND<half, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(HalfChunk, ChunkSize);
              TASSIGN(_ld, A1HalfUbAddr);
              TLOAD(_ld, _gm);
            }

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            // A2 = A * beta_2d
            TCVT(beta_ub, beta_ub_half, pto::RoundMode::CAST_NONE);
            pipe_barrier(PIPE_V);
            TMOV(beta_r_ub, beta_ub);
            pipe_barrier(PIPE_V);
            TCOLEXPAND(beta_2d_ub, beta_r_ub);

            TCVT(a1_ub, a1_ub_half, pto::RoundMode::CAST_NONE);
            TMUL(a2_ub, a1_ub, beta_2d_ub);
            TCVT(a2_ub_half, a2_ub, pto::RoundMode::CAST_NONE);

            // Store A2 -> workspace, signal Cube (flag 2)
            if (!first_iter_v) wait_flag_dev(3);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            {
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = HalfChunk; _gs.shape[4] = ChunkSize;
              GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>> _gm(
                  workspace_a2_handle +
                      static_cast<int64_t>(cid) * WsA2Size +
                      static_cast<int64_t>(vid) * HalfChunk * ChunkSize, _gs);
              UbND<half, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC> _st(HalfChunk, ChunkSize);
              TASSIGN(_st, A2HalfUbAddr);
              TSTORE(_gm, _st);
            }
            ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (2 << 8));

            // Load G -> UB
            {
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = 1; _gs.shape[4] = valid_rows;
              GlobalTensor<float, decltype(_gs), Stride<1, 1, 1, 1, 1>> _gm(
                  G_handle + static_cast<int64_t>(head_idx) * total_tokens
                           + chunk_token_start, _gs);
              UbND<float, 1, ChunkSize, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(1, valid_rows);
              TASSIGN(_ld, GUbAddr);
              TLOAD(_ld, _gm);
              if (valid_rows != ChunkSize) {
                UbND<float, 1, ChunkSize, 1, ChunkSize, PadValue::Zero> _pd;
                TASSIGN(_pd, GUbAddr);
                TFILLPAD_INPLACE(_pd, _ld);
              }
            }

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            // A1 = A * (exp(g) * beta)_2d
            TEXP(g_ub, g_ub);
            pipe_barrier(PIPE_V);
            TMUL(g_ub, g_ub, beta_ub);
            pipe_barrier(PIPE_V);
            TMOV(g_r_ub, g_ub);
            pipe_barrier(PIPE_V);
            TCOLEXPAND(g_2d_ub, g_r_ub);
            TMUL(a1_ub, a1_ub, g_2d_ub);
            TCVT(a1_ub_half, a1_ub, pto::RoundMode::CAST_NONE);

            // Store A1 -> workspace, signal Cube (flag 1)
            if (!first_iter_v) wait_flag_dev(4);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            {
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = HalfChunk; _gs.shape[4] = ChunkSize;
              GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>> _gm(
                  workspace_a1_handle +
                      static_cast<int64_t>(cid) * WsA1Size +
                      static_cast<int64_t>(vid) * HalfChunk * ChunkSize, _gs);
              UbND<half, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC> _st(HalfChunk, ChunkSize);
              TASSIGN(_st, A1HalfUbAddr);
              TSTORE(_gm, _st);
            }
            ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (1 << 8));
            first_iter_v = false;
          }
          gi++;
        }
      }
    }
  }
#endif

  // ════════════════════════════════════════════════════════════════════════
  // Cube phase: GEMM  U = A2 @ V  and  W = A1 @ K
  // Waits for Vec cross-core flags before loading workspace matrices.
  // Single L0 split (K=ChunkSize=128 fits in one 64KB L0 block).
  // ════════════════════════════════════════════════════════════════════════
#if defined(__DAV_C220_CUBE__)
  // ── Fixed-length sequence path (Cube) ─────────────────────────────────
  if (cu_seqlens == nullptr) {
    int64_t chunks_per_seq = (seq_len + ChunkSize - 1) / ChunkSize;
    for (int64_t work_idx = static_cast<int64_t>(cid);
         work_idx < total_work;
         work_idx += static_cast<int64_t>(block_num)) {
      int32_t head_idx = static_cast<int32_t>(work_idx % NumHeads);
      int64_t chunk_head_idx = work_idx / NumHeads;
      int64_t seq_idx = chunk_head_idx / chunks_per_seq;
      int64_t ci = chunk_head_idx % chunks_per_seq;

      int64_t bos = seq_idx * seq_len;
      int64_t slen = seq_len;
      int64_t chunk_start = ci * ChunkSize;
      int64_t remaining = slen - chunk_start;
      int32_t valid_rows = static_cast<int32_t>(
          remaining < ChunkSize ? remaining : ChunkSize);
      int64_t chunk_token_start = bos + chunk_start;

      int64_t kv_offset =
          (chunk_token_start * NumHeads + head_idx) *
          static_cast<int64_t>(HiddenSize);

      // Load K [B,S,N,D] -> L1, zero-pad if tail chunk
      {
        L1Mat<half, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l1(valid_rows, HiddenSize);
        TASSIGN(_l1, 0);
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = valid_rows; _gs.shape[4] = HiddenSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, NumHeads * HiddenSize, 1>> _gm(
            K_handle + kv_offset, _gs);
        TLOAD(_l1, _gm);
        if (valid_rows != ChunkSize) TFILLPAD(_l1, _l1);
      }
      // Load V [B,S,N,D] -> L1
      {
        L1Mat<half, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l1(valid_rows, HiddenSize);
        TASSIGN(_l1, 32768);
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = valid_rows; _gs.shape[4] = HiddenSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, NumHeads * HiddenSize, 1>> _gm(
            V_handle + kv_offset, _gs);
        TLOAD(_l1, _gm);
        if (valid_rows != ChunkSize) TFILLPAD(_l1, _l1);
      }

      // Wait for Vec's A2 workspace (cross-core flag 2) -> load A2 -> L1
      wait_flag_dev(2);
      {
        L1Mat<half, ChunkSize, ChunkSize, DYNAMIC, DYNAMIC> _l1(ChunkSize, ChunkSize);
        TASSIGN(_l1, 65536);
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = ChunkSize; _gs.shape[4] = ChunkSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>> _gm(
            workspace_a2_handle +
                static_cast<int64_t>(cid) * WsA2Size, _gs);
        TLOAD(_l1, _gm);
      }

      // GEMM: U = A2 @ V  (L1 -> L0A/L0B -> L0C)
      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      {
        TileLeft<half, ChunkSize, ChunkSize, ChunkSize, ChunkSize> _l0a;
        TileRight<half, ChunkSize, HiddenSize, ChunkSize, HiddenSize> _l0b;
        TASSIGN(_l0a, 0x0);
        TASSIGN(_l0b, 0x0);
        auto _we = EVENT_ID1;
        set_flag(PIPE_MTE2, PIPE_MTE1, _we);
        wait_flag(PIPE_MTE2, PIPE_MTE1, _we);
        set_flag(PIPE_M, PIPE_MTE1, _we);
        wait_flag(PIPE_M, PIPE_MTE1, _we);
        TEXTRACT(_l0a, a2_l1, 0, 0);
        TEXTRACT(_l0b, v_l1, 0, 0);
        set_flag(PIPE_MTE1, PIPE_M, _we);
        wait_flag(PIPE_MTE1, PIPE_M, _we);
        TMATMUL(u_l0, _l0a, _l0b);
        set_flag(PIPE_MTE1, PIPE_MTE2, _we);
        wait_flag(PIPE_MTE1, PIPE_MTE2, _we);
        set_flag(PIPE_M, PIPE_FIX, _we);
        wait_flag(PIPE_M, PIPE_FIX, _we);
      }

      // Store U from L0C -> GM (fp32->fp16 cast handled by TSTORE)
      {
        TileAcc<float, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l0(valid_rows, HiddenSize);
        TASSIGN(_l0, 0);
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = valid_rows; _gs.shape[4] = HiddenSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, NumHeads * HiddenSize, 1>> _gm(
            U_handle + kv_offset, _gs);
        TSTORE(_gm, _l0);
      }
      ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (3 << 8));

      // Wait for Vec's A1 workspace (cross-core flag 1) -> load A1 -> L1
      wait_flag_dev(1);
      {
        L1Mat<half, ChunkSize, ChunkSize, DYNAMIC, DYNAMIC> _l1(ChunkSize, ChunkSize);
        TASSIGN(_l1, 98304);
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = ChunkSize; _gs.shape[4] = ChunkSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>> _gm(
            workspace_a1_handle +
                static_cast<int64_t>(cid) * WsA1Size, _gs);
        TLOAD(_l1, _gm);
      }

      // GEMM: W = A1 @ K  (L1 -> L0A/L0B -> L0C)
      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      {
        TileLeft<half, ChunkSize, ChunkSize, ChunkSize, ChunkSize> _l0a;
        TileRight<half, ChunkSize, HiddenSize, ChunkSize, HiddenSize> _l0b;
        TASSIGN(_l0a, 0x0);
        TASSIGN(_l0b, 0x0);
        auto _we = EVENT_ID1;
        set_flag(PIPE_MTE2, PIPE_MTE1, _we);
        wait_flag(PIPE_MTE2, PIPE_MTE1, _we);
        set_flag(PIPE_M, PIPE_MTE1, _we);
        wait_flag(PIPE_M, PIPE_MTE1, _we);
        TEXTRACT(_l0a, a1_l1, 0, 0);
        TEXTRACT(_l0b, k_l1, 0, 0);
        set_flag(PIPE_MTE1, PIPE_M, _we);
        wait_flag(PIPE_MTE1, PIPE_M, _we);
        TMATMUL(w_l0, _l0a, _l0b);
        set_flag(PIPE_MTE1, PIPE_MTE2, _we);
        wait_flag(PIPE_MTE1, PIPE_MTE2, _we);
        set_flag(PIPE_M, PIPE_FIX, _we);
        wait_flag(PIPE_M, PIPE_FIX, _we);
      }

      // Store W from L0C -> GM
      {
        TileAcc<float, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l0(valid_rows, HiddenSize);
        TASSIGN(_l0, 65536);
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = valid_rows; _gs.shape[4] = HiddenSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, NumHeads * HiddenSize, 1>> _gm(
            W_handle + kv_offset, _gs);
        TSTORE(_gm, _l0);
      }
      ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (4 << 8));
    }
  }
  // ── Variable-length sequence path (Cube) ──────────────────────────────
  else {
    int64_t gi = 0;
    for (int64_t si = 0; si < num_seqs; ++si) {
      int64_t bos = static_cast<int64_t>(cu_seqlens[si]);
      int64_t eos = static_cast<int64_t>(cu_seqlens[si + 1]);
      int64_t slen = eos - bos;
      int64_t nc = (slen + ChunkSize - 1) / ChunkSize;

      for (int64_t ci = 0; ci < nc; ++ci) {
        for (int32_t h = 0; h < NumHeads; ++h) {
          if (gi % static_cast<int64_t>(block_num) ==
              static_cast<int64_t>(cid)) {
            int64_t chunk_start = ci * ChunkSize;
            int64_t remaining = slen - chunk_start;
            int32_t valid_rows = static_cast<int32_t>(
                remaining < ChunkSize ? remaining : ChunkSize);
            int64_t chunk_token_start = bos + chunk_start;
            int32_t head_idx = h;

            int64_t kv_offset =
                (chunk_token_start * NumHeads + head_idx) *
                static_cast<int64_t>(HiddenSize);

            // Load K -> L1
            {
              L1Mat<half, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l1(valid_rows, HiddenSize);
              TASSIGN(_l1, 0);
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = valid_rows; _gs.shape[4] = HiddenSize;
              GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, NumHeads * HiddenSize, 1>> _gm(
                  K_handle + kv_offset, _gs);
              TLOAD(_l1, _gm);
              if (valid_rows != ChunkSize) TFILLPAD(_l1, _l1);
            }
            // Load V -> L1
            {
              L1Mat<half, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l1(valid_rows, HiddenSize);
              TASSIGN(_l1, 32768);
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = valid_rows; _gs.shape[4] = HiddenSize;
              GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, NumHeads * HiddenSize, 1>> _gm(
                  V_handle + kv_offset, _gs);
              TLOAD(_l1, _gm);
              if (valid_rows != ChunkSize) TFILLPAD(_l1, _l1);
            }

            // Wait for A2, load -> L1
            wait_flag_dev(2);
            {
              L1Mat<half, ChunkSize, ChunkSize, DYNAMIC, DYNAMIC> _l1(ChunkSize, ChunkSize);
              TASSIGN(_l1, 65536);
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = ChunkSize; _gs.shape[4] = ChunkSize;
              GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>> _gm(
                  workspace_a2_handle +
                      static_cast<int64_t>(cid) * WsA2Size, _gs);
              TLOAD(_l1, _gm);
            }

            // GEMM: U = A2 @ V
            set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            {
              TileLeft<half, ChunkSize, ChunkSize, ChunkSize, ChunkSize> _l0a;
              TileRight<half, ChunkSize, HiddenSize, ChunkSize, HiddenSize> _l0b;
              TASSIGN(_l0a, 0x0);
              TASSIGN(_l0b, 0x0);
              auto _we = EVENT_ID1;
              set_flag(PIPE_MTE2, PIPE_MTE1, _we);
              wait_flag(PIPE_MTE2, PIPE_MTE1, _we);
              set_flag(PIPE_M, PIPE_MTE1, _we);
              wait_flag(PIPE_M, PIPE_MTE1, _we);
              TEXTRACT(_l0a, a2_l1, 0, 0);
              TEXTRACT(_l0b, v_l1, 0, 0);
              set_flag(PIPE_MTE1, PIPE_M, _we);
              wait_flag(PIPE_MTE1, PIPE_M, _we);
              TMATMUL(u_l0, _l0a, _l0b);
              set_flag(PIPE_MTE1, PIPE_MTE2, _we);
              wait_flag(PIPE_MTE1, PIPE_MTE2, _we);
              set_flag(PIPE_M, PIPE_FIX, _we);
              wait_flag(PIPE_M, PIPE_FIX, _we);
            }

            // Store U
            {
              TileAcc<float, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l0(valid_rows, HiddenSize);
              TASSIGN(_l0, 0);
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = valid_rows; _gs.shape[4] = HiddenSize;
              GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, NumHeads * HiddenSize, 1>> _gm(
                  U_handle + kv_offset, _gs);
              TSTORE(_gm, _l0);
            }
            ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (3 << 8));

            // Wait for A1, load -> L1
            wait_flag_dev(1);
            {
              L1Mat<half, ChunkSize, ChunkSize, DYNAMIC, DYNAMIC> _l1(ChunkSize, ChunkSize);
              TASSIGN(_l1, 98304);
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = ChunkSize; _gs.shape[4] = ChunkSize;
              GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>> _gm(
                  workspace_a1_handle +
                      static_cast<int64_t>(cid) * WsA1Size, _gs);
              TLOAD(_l1, _gm);
            }

            // GEMM: W = A1 @ K
            set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            {
              TileLeft<half, ChunkSize, ChunkSize, ChunkSize, ChunkSize> _l0a;
              TileRight<half, ChunkSize, HiddenSize, ChunkSize, HiddenSize> _l0b;
              TASSIGN(_l0a, 0x0);
              TASSIGN(_l0b, 0x0);
              auto _we = EVENT_ID1;
              set_flag(PIPE_MTE2, PIPE_MTE1, _we);
              wait_flag(PIPE_MTE2, PIPE_MTE1, _we);
              set_flag(PIPE_M, PIPE_MTE1, _we);
              wait_flag(PIPE_M, PIPE_MTE1, _we);
              TEXTRACT(_l0a, a1_l1, 0, 0);
              TEXTRACT(_l0b, k_l1, 0, 0);
              set_flag(PIPE_MTE1, PIPE_M, _we);
              wait_flag(PIPE_MTE1, PIPE_M, _we);
              TMATMUL(w_l0, _l0a, _l0b);
              set_flag(PIPE_MTE1, PIPE_MTE2, _we);
              wait_flag(PIPE_MTE1, PIPE_MTE2, _we);
              set_flag(PIPE_M, PIPE_FIX, _we);
              wait_flag(PIPE_M, PIPE_FIX, _we);
            }

            // Store W
            {
              TileAcc<float, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l0(valid_rows, HiddenSize);
              TASSIGN(_l0, 65536);
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = valid_rows; _gs.shape[4] = HiddenSize;
              GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, NumHeads * HiddenSize, 1>> _gm(
                  W_handle + kv_offset, _gs);
              TSTORE(_gm, _l0);
            }
            ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (4 << 8));
          }
          gi++;
        }
      }
    }
  }
#endif
}

extern "C" __global__ AICORE void launch_wy_fast(
    __gm__ uint8_t *K_handle, __gm__ uint8_t *V_handle,
    __gm__ uint8_t *Beta_handle, __gm__ uint8_t *G_handle,
    __gm__ uint8_t *A_handle,
    __gm__ uint8_t *workspace_a1_handle, __gm__ uint8_t *workspace_a2_handle,
    __gm__ uint8_t *W_handle, __gm__ uint8_t *U_handle,
    __gm__ uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len,
    int64_t total_tokens,
    uint64_t ffts_addr)
{
  wy_fast_kernel<GDN_H, GDN_D, GDN_C>(
      reinterpret_cast<__gm__ half *>(K_handle),
      reinterpret_cast<__gm__ half *>(V_handle),
      reinterpret_cast<__gm__ half *>(Beta_handle),
      reinterpret_cast<__gm__ float *>(G_handle),
      reinterpret_cast<__gm__ half *>(A_handle),
      reinterpret_cast<__gm__ half *>(workspace_a1_handle),
      reinterpret_cast<__gm__ half *>(workspace_a2_handle),
      reinterpret_cast<__gm__ half *>(W_handle),
      reinterpret_cast<__gm__ half *>(U_handle),
      reinterpret_cast<__gm__ int32_t *>(cu_seqlens),
      batch_size, seq_len, total_tokens, ffts_addr);
}

extern "C" void call_kernel(
    uint32_t block_dim, void *stream,
    uint8_t *k, uint8_t *v, uint8_t *beta, uint8_t *g_sum, uint8_t *A,
    uint8_t *workspace_a1, uint8_t *workspace_a2,
    uint8_t *w, uint8_t *u,
    uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len,
    int64_t total_tokens)
{
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  launch_wy_fast<<<block_dim, nullptr, stream>>>(
      k, v, beta, g_sum, A,
      workspace_a1, workspace_a2,
      w, u,
      cu_seqlens,
      batch_size, seq_len, total_tokens, fftsAddr);
}
