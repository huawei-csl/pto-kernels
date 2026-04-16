// ============================================================================
// scaled_dot_kkt_kernel.cpp — Intra-chunk attention matrix for GatedDeltaNet
//
// Computes A = mask(KK^T · gating_coeff) per chunk, where:
//   KK^T ∈ ℝ^{C×C} = K @ K^T                  (Cube engine, GEMM)
//   coeff[i,j] = exp(clamp(g[i]+log(β[i]) - g[j], max=0))  (Vec engine)
//   A[i,j] = KK^T[i,j] · coeff[i,j] · causal_mask[i,j]
//
// Inputs:
//   K       [total_tokens, H, D]  half  — key vectors in BSND layout
//   Beta    [H, total_tokens]     half  — gate bias (pre-transposed)
//   G       [H, total_tokens]     float — cumulative gate sum (pre-transposed)
//   Msk     [C, C]                float — lower-triangular causal mask
//
// Output:
//   A       [total_tokens, H, C]  half  — gated attention matrix in BSND
//
// Architecture: Cube + Vec cross-core kernel.
//   Cube phase: K→L1, GEMM K@K^T→L0C, store to workspace (GM)
//   Vec phase:  load workspace KK^T, compute gating coefficients, apply mask
//
// Cross-core sync: Cube signals Vec via FFTS flag after each chunk's KK^T
// is written to workspace. Vec signals back when workspace buffer is free.
// Two workspace slots alternate (double-buffering via slot = ci & 1).
//
// Vec sub-blocks: Two sub-blocks (vid=0,1) process upper/lower halves of
// the C×C attention matrix in parallel (HalfChunk rows each).
//
// NPU memory hierarchy:
//   GM → L1 (Cube-accessible) → L0A/L0B (GEMM operands) → L0C (accumulator)
//   GM → UB (Vec-accessible SRAM)
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
#ifdef __CCE_AICORE__
// UB tile in row-major (ND) layout
template <typename T, int R, int C, int RV = R, int CV = C,
          pto::PadValue P = pto::PadValue::Null>
using UbND = pto::Tile<pto::TileType::Vec, T, R, C, pto::BLayout::RowMajor,
                       RV, CV, pto::SLayout::NoneBox, 512, P>;

// UB tile in column-major (DN) layout
template <typename T, int R, int C, int RV = R, int CV = C>
using UbDN = pto::Tile<pto::TileType::Vec, T, R, C, pto::BLayout::ColMajor,
                       RV, CV, pto::SLayout::NoneBox, 512>;

// L1 matrix tile in NZ format
template <typename T, int R, int C, int RV = R, int CV = C>
using L1Mat = pto::Tile<pto::TileType::Mat, T, R, C, pto::BLayout::ColMajor,
                        RV, CV, pto::SLayout::RowMajor, 512, pto::PadValue::Zero>;

// L1 matrix tile in ZN format (for transposed views)
template <typename T, int R, int C, int RV = R, int CV = C>
using L1MatZN = pto::Tile<pto::TileType::Mat, T, R, C, pto::BLayout::RowMajor,
                          RV, CV, pto::SLayout::ColMajor, 512, pto::PadValue::Zero>;
#endif

template <int32_t NumHeads, int32_t HiddenSize, int32_t ChunkSize>
AICORE void kkt_kernel(
    __gm__ half *K_handle, __gm__ half *Beta_handle,
    __gm__ float *G_handle, __gm__ float *Msk_handle,
    __gm__ half *workspace_handle, __gm__ half *A_handle,
    __gm__ int32_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len,
    int64_t total_tokens,
    uint64_t ffts_addr)
{
  constexpr int32_t HalfChunk = ChunkSize / 2;
  constexpr int32_t ChunkSquare = ChunkSize * ChunkSize;
  constexpr uint32_t KTail =
      (HiddenSize % 128 == 0) ? 128 : (HiddenSize % 128);

  constexpr int32_t GUbAddr      = 0;
  constexpr int32_t BetaHalfUbAddr = 512;
  constexpr int32_t BetaUbAddr   = 640;
  constexpr int32_t GvUbAddr     = 896;
  constexpr int32_t AUbAddr      = 1152;
  constexpr int32_t GRUbAddr     = 33920;
  constexpr int32_t GCUbAddr     = 34176;
  constexpr int32_t MskUbAddr    = 34688;
  constexpr int32_t GR2dUbAddr   = 67456;
  constexpr int32_t GC2dUbAddr   = 124800;
  constexpr int32_t CoeffUbAddr  = 157568;
  constexpr int32_t AUbHalfAddr  = GR2dUbAddr;

  set_ffts_base_addr(ffts_addr);
  auto cid = get_block_idx();
  auto block_num = get_block_num();
  auto vid = get_subblockid();

  int64_t num_seqs = batch_size;
  int64_t total_work = num_seqs * NumHeads;

  // Cube-side tiles: K in L1 (NZ format), accumulator in L0C
  L1Mat<half, ChunkSize, HiddenSize,
        ChunkSize, HiddenSize> k_l1;
  TASSIGN(k_l1, 0);
  TileAcc<float, ChunkSize, ChunkSize, ChunkSize, ChunkSize> a_l0;
  TASSIGN(a_l0, 0);

  // Vec-side UB tiles for gating computation
  UbND<float, 1, ChunkSize, 1, ChunkSize> g_ub;
  TASSIGN(g_ub, GUbAddr);
  UbND<half, 1, HalfChunk, 1, HalfChunk> beta_ub_half;
  TASSIGN(beta_ub_half, BetaHalfUbAddr);
  UbND<float, 1, HalfChunk, 1, HalfChunk> beta_ub;
  TASSIGN(beta_ub, BetaUbAddr);
  UbND<float, 1, HalfChunk, 1, HalfChunk> g_v_ub;
  TASSIGN(g_v_ub, GvUbAddr);
  UbND<float, HalfChunk, ChunkSize,
       HalfChunk, ChunkSize> a_ub;
  TASSIGN(a_ub, AUbAddr);
  UbND<float, 1, HalfChunk, 1, HalfChunk> g_r_ub;
  TASSIGN(g_r_ub, GRUbAddr);
  UbND<float, 1, ChunkSize, 1, ChunkSize> g_c_ub;
  TASSIGN(g_c_ub, GCUbAddr);
  UbND<float, HalfChunk, ChunkSize,
       HalfChunk, ChunkSize> msk_ub;
  TASSIGN(msk_ub, MskUbAddr);
  UbND<float, HalfChunk, ChunkSize,
       HalfChunk, ChunkSize> g_r_2d_ub;
  TASSIGN(g_r_2d_ub, GR2dUbAddr);
  UbND<float, HalfChunk, ChunkSize,
       HalfChunk, ChunkSize> g_c_2d_ub;
  TASSIGN(g_c_2d_ub, GC2dUbAddr);
  UbND<float, HalfChunk, ChunkSize,
       HalfChunk, ChunkSize> coeff_ub;
  TASSIGN(coeff_ub, CoeffUbAddr);
  UbND<half, HalfChunk, ChunkSize,
       HalfChunk, ChunkSize> a_ub_half;
  TASSIGN(a_ub_half, AUbHalfAddr);

  // ========================================================================
  // CUBE PHASE: Compute KK^T = K @ K^T for each chunk via GEMM
  // ========================================================================
#if defined(__DAV_C220_CUBE__)
  for (int64_t work_idx = 0;
       work_idx < (total_work + block_num - 1) / block_num; ++work_idx) {
    int64_t pid = work_idx * static_cast<int64_t>(block_num) +
                  static_cast<int64_t>(cid);
    if (pid >= total_work) continue;

    int32_t head_idx = static_cast<int32_t>(pid % NumHeads);
    int64_t seq_idx = pid / NumHeads;

    int64_t bos, slen;
    if (cu_seqlens != nullptr) {
      bos = static_cast<int64_t>(cu_seqlens[seq_idx]);
      slen = static_cast<int64_t>(cu_seqlens[seq_idx + 1]) - bos;
    } else {
      bos = seq_idx * seq_len;
      slen = seq_len;
    }
    int64_t num_chunks = (slen + ChunkSize - 1) / ChunkSize;

    for (int64_t ci = 0; ci < num_chunks; ++ci) {
      int32_t slot = static_cast<int32_t>(ci & 1);
      // Wait for Vec to finish reading the previous KK^T from this slot
      wait_flag_dev(2 + slot);
      pipe_barrier(PIPE_ALL);

      int64_t chunk_start = ci * ChunkSize;
      int64_t remaining = slen - chunk_start;
      int32_t valid_rows = static_cast<int32_t>(
          remaining < ChunkSize ? remaining : ChunkSize);

      // K is in BSND layout: stride between tokens = NumHeads * HiddenSize
      int64_t k_offset =
          ((bos + chunk_start) * NumHeads + head_idx) *
          static_cast<int64_t>(HiddenSize);

      // ── Load K chunk from GM → L1 (MTE2 pipe) ──────────────────────
      {
        L1Mat<half, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l1(valid_rows, HiddenSize);
        TASSIGN(_l1, 0);
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = valid_rows; _gs.shape[4] = HiddenSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, NumHeads * HiddenSize, 1>> _gm(K_handle + k_offset, _gs);
        TLOAD(_l1, _gm);
        if (valid_rows != ChunkSize) TFILLPAD(_l1, _l1);
      }

      // ── GEMM: KK^T = K @ K^T (L1→L0A/L0B→L0C) ────────────────────
      // K is [C×D] in L1 NZ; K^T obtained via ZN reshape of same tile.
      // WAR sync: MTE2→MTE1, M→MTE1 before extract; MTE1→M before matmul.
      {
        TileLeft<half, ChunkSize, HiddenSize, ChunkSize, HiddenSize> _l0a;
        TileRight<half, HiddenSize, ChunkSize, HiddenSize, ChunkSize> _l0b;
        TASSIGN(_l0a, 0x0);
        TASSIGN(_l0b, 0x0);
        auto _we = EVENT_ID1;
        set_flag(PIPE_MTE2, PIPE_MTE1, _we);
        wait_flag(PIPE_MTE2, PIPE_MTE1, _we);
        set_flag(PIPE_M, PIPE_MTE1, _we);
        wait_flag(PIPE_M, PIPE_MTE1, _we);
        // Left operand: K in NZ format, extract directly to L0A
        TEXTRACT(_l0a, k_l1, 0, 0);
        // Right operand: K^T via ZN reshape of same L1 tile, extract to L0B
        L1MatZN<half, HiddenSize, ChunkSize> _bzn;
        TRESHAPE(_bzn, k_l1);
        TEXTRACT(_l0b, _bzn, 0, 0);
        set_flag(PIPE_MTE1, PIPE_M, _we);
        wait_flag(PIPE_MTE1, PIPE_M, _we);
        TMATMUL(a_l0, _l0a, _l0b);
        set_flag(PIPE_MTE1, PIPE_MTE2, _we);
        wait_flag(PIPE_MTE1, PIPE_MTE2, _we);
        set_flag(PIPE_M, PIPE_FIX, _we);
        wait_flag(PIPE_M, PIPE_FIX, _we);
      }

      // ── Store KK^T from L0C → workspace GM (with fp32→fp16 cast) ───
      {
        TileAcc<float, ChunkSize, ChunkSize, DYNAMIC, DYNAMIC> _l0(ChunkSize, ChunkSize);
        TASSIGN(_l0, 0);
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = ChunkSize; _gs.shape[4] = ChunkSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>> _gm(
            workspace_handle +
                (static_cast<int64_t>(cid) * 2 + slot) * ChunkSquare,
            _gs);
        TSTORE(_gm, _l0);
      }

      // Signal Vec that this slot's KK^T is ready
      ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (slot << 8));
    }
  }
#endif

  // ========================================================================
  // VEC PHASE: Apply gating and causal mask to KK^T
  //   coeff[i,j] = exp(min(g[i]+log(β[i]) - g[j], 0))
  //   A[i,j] = KK^T[i,j] · coeff[i,j] · mask[i,j]
  // Each sub-block (vid=0,1) handles HalfChunk rows of the C×C matrix.
  // ========================================================================
#if defined(__DAV_C220_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);

  // ── Load causal mask (lower triangular) once, reused across all chunks ──
  {
    Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
    _gs.shape[3] = HalfChunk; _gs.shape[4] = ChunkSize;
    GlobalTensor<float, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>> _gm(
        Msk_handle +
            static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
        _gs);
    UbND<float, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(HalfChunk, ChunkSize);
    TASSIGN(_ld, MskUbAddr);
    TLOAD(_ld, _gm);
  }
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

  // Initial cross-core sync: release both workspace slots so Cube can start
  ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (2 << 8));
  ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (3 << 8));

  for (int64_t work_idx = 0;
       work_idx < (total_work + block_num - 1) / block_num; ++work_idx) {
    int64_t pid = work_idx * static_cast<int64_t>(block_num) +
                  static_cast<int64_t>(cid);
    if (pid >= total_work) continue;

    int32_t head_idx = static_cast<int32_t>(pid % NumHeads);
    int64_t seq_idx = pid / NumHeads;

    int64_t bos, slen;
    if (cu_seqlens != nullptr) {
      bos = static_cast<int64_t>(cu_seqlens[seq_idx]);
      slen = static_cast<int64_t>(cu_seqlens[seq_idx + 1]) - bos;
    } else {
      bos = seq_idx * seq_len;
      slen = seq_len;
    }
    int64_t num_chunks = (slen + ChunkSize - 1) / ChunkSize;

    for (int64_t ci = 0; ci < num_chunks; ++ci) {
      int32_t slot = static_cast<int32_t>(ci & 1);

      int64_t chunk_start = ci * ChunkSize;
      int64_t remaining = slen - chunk_start;
      int32_t valid_rows = static_cast<int32_t>(
          remaining < ChunkSize ? remaining : ChunkSize);
      int32_t row_offset = static_cast<int32_t>(vid) * HalfChunk;
      int32_t local_valid =
          valid_rows > row_offset
              ? (valid_rows - row_offset < HalfChunk
                     ? valid_rows - row_offset
                     : HalfChunk)
              : 0;

      if (local_valid > 0) {
        // ── Load G (full chunk, 1×C) and Beta (sub-block rows, 1×HalfC) ──
        // G is [H, total_tokens] float — contiguous per head
        {
          Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
          _gs.shape[3] = 1; _gs.shape[4] = valid_rows;
          GlobalTensor<float, decltype(_gs), Stride<1, 1, 1, 1, 1>> _gm(
              G_handle + static_cast<int64_t>(head_idx) * total_tokens
                       + (bos + chunk_start),
              _gs);
          UbND<float, 1, ChunkSize, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(1, valid_rows);
          TASSIGN(_ld, GUbAddr);
          TLOAD(_ld, _gm);
          if (valid_rows != ChunkSize) {
            UbND<float, 1, ChunkSize, 1, ChunkSize, PadValue::Zero> _pd;
            TASSIGN(_pd, GUbAddr);
            TFILLPAD_INPLACE(_pd, _ld);
          }
        }

        // Beta is [H, total_tokens] half — contiguous per head
        {
          Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
          _gs.shape[3] = 1; _gs.shape[4] = local_valid;
          GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, 1, 1>> _gm(
              Beta_handle + static_cast<int64_t>(head_idx) * total_tokens
                          + (bos + chunk_start + row_offset),
              _gs);
          UbND<half, 1, HalfChunk, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(1, local_valid);
          TASSIGN(_ld, BetaHalfUbAddr);
          TLOAD(_ld, _gm);
          if (local_valid != HalfChunk) {
            UbND<half, 1, HalfChunk, 1, HalfChunk, PadValue::Zero> _pd;
            TASSIGN(_pd, BetaHalfUbAddr);
            TFILLPAD_INPLACE(_pd, _ld);
          }
        }
      }

      // Wait for Cube to finish writing KK^T for this slot
      wait_flag_dev(slot);
      pipe_barrier(PIPE_ALL);

      if (local_valid > 0) {
        // ── Compute gating coefficient ────────────────────────────────
        // g_v[i] = g[row_offset+i] + log(β[i])  — combined row gate
        TCVT(beta_ub, beta_ub_half, pto::RoundMode::CAST_NONE);
        UbND<float, 1, HalfChunk, 1, HalfChunk>
            g_ub_temp;
        TASSIGN(g_ub_temp,
                GUbAddr + row_offset *
                              static_cast<int32_t>(sizeof(float)));
        TMOV(g_v_ub, g_ub_temp);
        pipe_barrier(PIPE_V);

        TLOG(beta_ub, beta_ub);
        pipe_barrier(PIPE_V);
        TADD(g_v_ub, g_v_ub, beta_ub);
        pipe_barrier(PIPE_V);
        TMOV(g_r_ub, g_v_ub);
        TMOV(g_c_ub, g_ub);
        pipe_barrier(PIPE_V);

        // Broadcast g_v to rows, g to columns → 2D gating matrix
        // coeff[i,j] = exp(min(g_v[i] - g[j], 0))
        UbDN<float, HalfChunk, 1,
             HalfChunk, 1> g_r_ub_temp;
        TASSIGN(g_r_ub_temp, GRUbAddr);
        TROWEXPAND(g_r_2d_ub, g_r_ub_temp);
        TCOLEXPAND(g_c_2d_ub, g_c_ub);
        pipe_barrier(PIPE_V);
        TSUB(coeff_ub, g_r_2d_ub, g_c_2d_ub);
        pipe_barrier(PIPE_V);
        TMINS(coeff_ub, coeff_ub, 0.0f);
        pipe_barrier(PIPE_V);
        TEXP(coeff_ub, coeff_ub);

        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

        // ── Load KK^T sub-block from workspace (fp16) ────────────────
        {
          Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
          _gs.shape[3] = HalfChunk; _gs.shape[4] = ChunkSize;
          GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>> _gm(
              workspace_handle +
                  (static_cast<int64_t>(cid) * 2 + slot) * ChunkSquare +
                  static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
              _gs);
          UbND<half, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(HalfChunk, ChunkSize);
          TASSIGN(_ld, AUbHalfAddr);
          TLOAD(_ld, _gm);
        }

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        // ── Apply gating and mask: A = KK^T · coeff · mask ───────────
        TCVT(a_ub, a_ub_half, pto::RoundMode::CAST_NONE);
        TMUL(a_ub, a_ub, coeff_ub);
        TMUL(a_ub, a_ub, msk_ub);
        TCVT(a_ub_half, a_ub, pto::RoundMode::CAST_NONE);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        // ── Store A sub-block to output GM ────────────────────────────
        int64_t a_gm_offset =
            ((bos + chunk_start + row_offset) * NumHeads +
             head_idx) *
            static_cast<int64_t>(ChunkSize);

        {
          Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
          _gs.shape[3] = local_valid; _gs.shape[4] = ChunkSize;
          GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, NumHeads * ChunkSize, 1>> _gm(A_handle + a_gm_offset, _gs);
          UbND<half, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC> _st(local_valid, ChunkSize);
          TASSIGN(_st, AUbHalfAddr);
          TSTORE(_gm, _st);
        }
      }

      pipe_barrier(PIPE_ALL);
      // Signal Cube that this workspace slot is free for reuse
      ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | ((2 + slot) << 8));
    }
  }
#endif
}

extern "C" __global__ AICORE void launch_scaled_dot_kkt(
    __gm__ uint8_t *K_handle, __gm__ uint8_t *Beta_handle,
    __gm__ uint8_t *G_handle, __gm__ uint8_t *Msk_handle,
    __gm__ uint8_t *workspace_handle, __gm__ uint8_t *A_handle,
    __gm__ uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len,
    int64_t total_tokens,
    uint64_t ffts_addr)
{
  kkt_kernel<GDN_H, GDN_D, GDN_C>(
      reinterpret_cast<__gm__ half *>(K_handle),
      reinterpret_cast<__gm__ half *>(Beta_handle),
      reinterpret_cast<__gm__ float *>(G_handle),
      reinterpret_cast<__gm__ float *>(Msk_handle),
      reinterpret_cast<__gm__ half *>(workspace_handle),
      reinterpret_cast<__gm__ half *>(A_handle),
      reinterpret_cast<__gm__ int32_t *>(cu_seqlens),
      batch_size, seq_len, total_tokens, ffts_addr);
}

extern "C" void call_kernel(
    uint32_t block_dim, void *stream,
    uint8_t *K_handle, uint8_t *Beta_handle,
    uint8_t *G_handle, uint8_t *Msk_handle,
    uint8_t *workspace_handle, uint8_t *A_handle,
    uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len,
    int64_t total_tokens)
{
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  launch_scaled_dot_kkt<<<block_dim, nullptr, stream>>>(
      K_handle, Beta_handle, G_handle, Msk_handle,
      workspace_handle, A_handle, cu_seqlens,
      batch_size, seq_len, total_tokens, fftsAddr);
}
