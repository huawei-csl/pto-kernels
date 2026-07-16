// ============================================================================
// kda_chunk_o.cpp — Output stage for KDA (per-dim gate)
//
// Math (per chunk, matches ref_kda_chunk_o in
//   tests/test_kda_single_kernels.py:333-380):
//   q_eff = q * exp(g_cs)              # [c_len, K]
//   k_eff = k * exp(-g_cs)             # [c_len, K]
//   inter = q_eff @ S                  # [c_len, V]
//   Aqk   = tril(q_eff @ k_eff^T,      # [c_len, c_len], INCLUSIVE diagonal
//                diagonal=0)
//   o     = inter + Aqk @ v_corr       # [c_len, V]
//
// where S = s_snapshots[ci_base + ci, head] is the [K, V] state *entering*
// this chunk (already computed by kda_chunk_h), and v_corr = u - w @ S is
// the corrected values (also from kda_chunk_h).
//
// Differences from GDN chunk_o.cpp:
//   - Gate is per-DIMENSION (g_cs has shape [HV, T, K] head-major).
//   - Vec pre-scales Q and K element-wise (q*exp(g_cs), k*exp(-g_cs)) BEFORE
//     Cube sees them.  Cube does pure matmuls; there is no per-element gating
//     coefficient applied to QK on the Vec side.
//   - Causal mask is INCLUSIVE of the diagonal (rows >= cols), so the mask
//     tensor passed from Python differs from kkt_kda's strict-lower mask.
//   - No GQA: Q, K, V_corr, O all use HV heads.
//   - S is fp16 in GM (from kda_chunk_h's output) — Vec casts to fp32 into
//     workspace so Cube has fp32 sources for all three GEMMs.
//
// Chunks within a (seq, head) work item are fully independent (each reads
// its own s_snapshots entry).  Cube/Vec still process them sequentially per
// work item to keep the per-core 4-flag protocol simple.
//
// Cross-core sync: same data-flow flags as kda_chunk_h (0-3), plus a
// full mix-core barrier on entry/exit via SYNCALL<SyncCoreType::Mix>().
//
// Inputs:
//   Q       [HV, T, K]               fp16  — queries (head-major)
//   K       [HV, T, K]               fp16  — keys    (head-major)
//   V_corr  [T, HV, V]               fp16  — corrected values from kda_chunk_h
//   (BSND) S       [total_chunks, HV, K, V] fp16  — snapshots from kda_chunk_h
//   G_cs    [HV, T, K]               fp32  — per-dim cumulative gate
//   (head-major) Msk     [C, C]                   fp32  — inclusive lower-tri
//   mask (rows >= cols) workspace [per-core scratch]     fp32  — 7 slots × K*V
//   floats O       [T, HV, V]               fp16  — output (BSND)
//
// Workspace per AI core (7 slots, fp32; assumes K == V == HiddenSize):
//   WS_Q   [C, K]   Vec writes q*exp(g_cs)  → Cube reads (GEMM1 A, GEMM2 A)
//   WS_K   [C, K]   Vec writes k*exp(-g_cs) → Cube reads (GEMM1 B, transposed)
//   WS_V   [C, V]   Vec writes V_corr fp32  → Cube reads (GEMM3 B)
//   WS_S   [K, V]   Vec writes S fp32       → Cube reads (GEMM2 B)
//   WS_QK  [C, C]   Cube writes QK fp32     → Vec masks  → Cube reads (GEMM3 A)
//   WS_QS  [C, V]   Cube writes QS fp32     → Vec reads (final combine)
//   WS_QKV [C, V]   Cube writes QKV fp32    → Vec reads (final combine)
// ============================================================================

#include "kernel_utils.h"
using namespace pto;
using kernel_utils::GetOuterLayout;
using kernel_utils::PipeBarrierVec;
using kernel_utils::SetCrossFlag;
using kernel_utils::SignalBothVecOnA5;
using kernel_utils::WaitBothVecOnA5;

#ifndef GDN_H
#define GDN_H 16
#endif

#ifndef GDN_D
#define GDN_D 128
#endif

#ifndef GDN_C
#define GDN_C 128
#endif

#ifdef __CCE_AICORE__

namespace {

using GmShape2D = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;
using GmStride2D = pto::Stride<1, 1, 1, pto::DYNAMIC, 1>;

template <typename T>
using GmTensor2D = pto::GlobalTensor<T, GmShape2D, GmStride2D>;

template <typename T, int32_t Rows, int32_t Cols>
using DynMatL1 = pto::Tile<pto::TileType::Mat, T, Rows, Cols,
                           pto::BLayout::ColMajor, pto::DYNAMIC, pto::DYNAMIC,
                           pto::SLayout::RowMajor, 512, pto::PadValue::Zero>;

template <typename T, int32_t Rows, int32_t Cols,
          pto::PadValue PadVal = pto::PadValue::Null>
using DynVecTile =
    pto::Tile<pto::TileType::Vec, T, Rows, Cols, pto::BLayout::RowMajor,
              pto::DYNAMIC, pto::DYNAMIC, pto::SLayout::NoneBox, 512, PadVal>;

template <typename T, int32_t Rows, int32_t Cols, int32_t RowValid = Rows,
          int32_t ColValid = Cols>
using TileMatL1 = pto::Tile<pto::TileType::Mat, T, Rows, Cols,
                            pto::BLayout::ColMajor, RowValid, ColValid,
                            pto::SLayout::RowMajor, 512, pto::PadValue::Zero>;

template <typename T, int32_t Rows, int32_t Cols, int32_t RowValid = Rows,
          int32_t ColValid = Cols>
using TileMatL1ZN = pto::Tile<pto::TileType::Mat, T, Rows, Cols,
                              pto::BLayout::RowMajor, RowValid, ColValid,
                              pto::SLayout::ColMajor, 512, pto::PadValue::Zero>;

template <typename T, int32_t Rows, int32_t Cols, int32_t RowValid = Rows,
          int32_t ColValid = Cols>
using TileMatL0A =
    pto::Tile<pto::TileType::Left, T, Rows, Cols,
              GetOuterLayout(/* isLeft*/ true), RowValid, ColValid,
              pto::SLayout::RowMajor, 512, pto::PadValue::Zero>;

template <typename T, int32_t Rows, int32_t Cols, int32_t RowValid = Rows,
          int32_t ColValid = Cols>
using TileMatL0B =
    pto::Tile<pto::TileType::Right, T, Rows, Cols,
              GetOuterLayout(/* isLeft*/ false), RowValid, ColValid,
              pto::SLayout::ColMajor, 512, pto::PadValue::Zero>;

template <typename T, int32_t Rows, int32_t Cols, int32_t RowValid = Rows,
          int32_t ColValid = Cols, pto::PadValue PadVal = pto::PadValue::Null>
using TileUbDataND =
    pto::Tile<pto::TileType::Vec, T, Rows, Cols, pto::BLayout::RowMajor,
              RowValid, ColValid, pto::SLayout::NoneBox, 512, PadVal>;

// Single-shot dense GEMM via L0A/L0B — all three GEMMs have inner-dim == 128.
template <typename T1, typename T2, int32_t M, int32_t N, int32_t K,
          bool transpose_B = false>
AICORE PTO_INLINE void gemm_oneshot(
    TileMatL1<T1, M, K, M, K>& A,
    std::conditional_t<transpose_B, TileMatL1<T1, N, K, N, K>,
                       TileMatL1<T1, K, N, K, N>>& B,
    pto::TileAcc<T2, M, N, M, N>& C) {
  TileMatL0A<T1, M, K, M, K> l0a;
  TileMatL0B<T1, K, N, K, N> l0b;
  pto::TASSIGN(l0a, 0x0);
  pto::TASSIGN(l0b, 0x0);

  auto war_event_id = (event_t)(((int)EVENT_ID0 + 1) % 8);
  set_flag(PIPE_MTE2, PIPE_MTE1, war_event_id);
  wait_flag(PIPE_MTE2, PIPE_MTE1, war_event_id);
  set_flag(PIPE_M, PIPE_MTE1, war_event_id);
  wait_flag(PIPE_M, PIPE_MTE1, war_event_id);

  pto::TEXTRACT(l0a, A, 0, 0);
  if constexpr (!transpose_B) {
    pto::TEXTRACT(l0b, B, 0, 0);
  } else {
    TileMatL1ZN<T1, K, N, K, N> B_t;
    pto::TRESHAPE(B_t, B);
    pto::TEXTRACT(l0b, B_t, 0, 0);
  }

  set_flag(PIPE_MTE1, PIPE_M, war_event_id);
  wait_flag(PIPE_MTE1, PIPE_M, war_event_id);
  pto::TMATMUL(C, l0a, l0b);

  set_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);
  wait_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);
  set_flag(PIPE_M, PIPE_FIX, war_event_id);
  wait_flag(PIPE_M, PIPE_FIX, war_event_id);
}

}  // namespace

#endif

template <int32_t NumHeads, int32_t HiddenSize, int32_t ChunkSize>
AICORE void kda_chunk_o_kernel(__gm__ half* Q_handle, __gm__ half* K_handle,
                               __gm__ half* V_handle, __gm__ half* S_handle,
                               __gm__ float* G_handle,
                               __gm__ float* Mask_handle,
                               __gm__ float* workspace_handle,
                               __gm__ half* O_handle,
                               __gm__ int32_t* cu_seqlens, int64_t batch_size,
                               int64_t seq_len, int64_t total_tokens) {
  auto cid = get_block_idx();

  constexpr int32_t K_DIM = HiddenSize;
  constexpr int32_t V_DIM = HiddenSize;
  constexpr int32_t C = ChunkSize;
  constexpr int32_t H = NumHeads;
  constexpr int32_t HalfC = C / 2;
  constexpr int32_t BSND_STRIDE = H * HiddenSize;
  constexpr int32_t HM_STRIDE = HiddenSize;
  constexpr int32_t KV = K_DIM * V_DIM;

  // ── Workspace slots (fp32 elements, per AI core) ─────────────────────────
  constexpr int32_t WS_Q = 0;
  constexpr int32_t WS_K = WS_Q + C * K_DIM;
  constexpr int32_t WS_V = WS_K + C * K_DIM;
  constexpr int32_t WS_S = WS_V + C * V_DIM;
  constexpr int32_t WS_QK = WS_S + KV;
  constexpr int32_t WS_QS = WS_QK + C * C;
  constexpr int32_t WS_QKV = WS_QS + C * V_DIM;
  constexpr int32_t WS_PER_CORE = WS_QKV + C * V_DIM;

#if defined(__DAV_CUBE__)
  TileMatL1<float, C, K_DIM, C, K_DIM> q_l1;
  TASSIGN(q_l1, 0);
  TileMatL1<float, K_DIM, V_DIM, K_DIM, V_DIM> s_l1;
  TASSIGN(s_l1, (C * K_DIM + C * K_DIM) * sizeof(float));
  TileMatL1<float, C, C, C, C> qkm_l1;
  TASSIGN(qkm_l1, (C * K_DIM + C * K_DIM + KV) * sizeof(float));
  TileMatL1<float, C, V_DIM, C, V_DIM> v_l1;
  TASSIGN(v_l1, (C * K_DIM + C * K_DIM + KV + C * C) * sizeof(float));

  TileAcc<float, C, V_DIM, C, V_DIM> qkv_l0;
  TASSIGN(qkv_l0, 0);
  TileAcc<float, C, V_DIM, C, V_DIM> qs_l0;
  TASSIGN(qs_l0, C * C * sizeof(float));

#endif

#if defined(__DAV_VEC__)

#endif

  int64_t num_seqs = batch_size;
  int64_t total_work = num_seqs * H;

  SYNCALL<SyncCoreType::Mix>();

#if defined(__DAV_CUBE__)

  for (int64_t wi = 0; wi < (total_work + block_num - 1) / block_num; ++wi) {
    int64_t pid = wi * block_num + cid;
    if (pid >= total_work) break;

    int64_t head = pid % H;
    int64_t seq_idx = pid / H;

    int64_t bos, slen;
    if (cu_seqlens != nullptr) {
      bos = static_cast<int64_t>(cu_seqlens[seq_idx]);
      int64_t eos = static_cast<int64_t>(cu_seqlens[seq_idx + 1]);
      slen = eos - bos;
    } else {
      bos = seq_idx * seq_len;
      slen = seq_len;
    }
    int64_t num_chunks = (slen + C - 1) / C;
    int64_t ws_base = static_cast<int64_t>(cid) * WS_PER_CORE;

    for (int32_t ci = 0; ci < num_chunks; ++ci) {
      // Wait Vec phase A: q_eff, masked Aqk, V_corr, S all in workspace.
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
      wait_flag_dev(0);
#else
      WaitBothVecOnA5<PIPE_MTE3>(0);
#endif

      {
        GmShape2D q_shape(C, K_DIM);
        GmStride2D q_stride(K_DIM);
        GmTensor2D<float> q_global(workspace_handle + ws_base + WS_Q, q_shape,
                                   q_stride);
        DynMatL1<float, C, K_DIM> q_l1_load(C, K_DIM);
        TASSIGN(q_l1_load, 0);
        TLOAD(q_l1_load, q_global);
      }
      {
        GmShape2D s_shape(K_DIM, V_DIM);
        GmStride2D s_stride(V_DIM);
        GmTensor2D<float> s_global(workspace_handle + ws_base + WS_S, s_shape,
                                   s_stride);
        DynMatL1<float, K_DIM, V_DIM> s_l1_load(K_DIM, V_DIM);
        TASSIGN(s_l1_load, (C * K_DIM + C * K_DIM) * sizeof(float));
        TLOAD(s_l1_load, s_global);
      }
      {
        GmShape2D v_shape(C, V_DIM);
        GmStride2D v_stride(V_DIM);
        GmTensor2D<float> v_global(workspace_handle + ws_base + WS_V, v_shape,
                                   v_stride);
        DynMatL1<float, C, V_DIM> v_l1_load(C, V_DIM);
        TASSIGN(v_l1_load,
                (C * K_DIM + C * K_DIM + KV + C * C) * sizeof(float));
        TLOAD(v_l1_load, v_global);
      }
      {
        GmShape2D qkm_shape(C, C);
        GmStride2D qkm_stride(C);
        GmTensor2D<float> qkm_global(workspace_handle + ws_base + WS_QK,
                                     qkm_shape, qkm_stride);
        DynMatL1<float, C, C> qkm_l1_load(C, C);
        TASSIGN(qkm_l1_load, (C * K_DIM + C * K_DIM + KV) * sizeof(float));
        TLOAD(qkm_l1_load, qkm_global);
      }

      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

      // GEMM2: QS = q_eff @ S  [C, K] @ [K, V] → [C, V].
      gemm_oneshot<float, float, C, V_DIM, K_DIM, /*transpose_B=*/false>(
          q_l1, s_l1, qs_l0);

      {
        GmShape2D qs_shape(C, V_DIM);
        GmStride2D qs_stride(V_DIM);
        GmTensor2D<float> qs_global(workspace_handle + ws_base + WS_QS,
                                    qs_shape, qs_stride);
        TileAcc<float, C, V_DIM, C, V_DIM> qs_store;
        TASSIGN(qs_store, C * C * sizeof(float));
        TSTORE(qs_global, qs_store);
      }

      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

      // GEMM3: QKV = Aqk_masked @ V_corr  [C, C] @ [C, V] → [C, V].
      gemm_oneshot<float, float, C, V_DIM, C, /*transpose_B=*/false>(
          qkm_l1, v_l1, qkv_l0);

      {
        GmShape2D qkv_shape(C, V_DIM);
        GmStride2D qkv_stride(V_DIM);
        GmTensor2D<float> qkv_global(workspace_handle + ws_base + WS_QKV,
                                     qkv_shape, qkv_stride);
        TileAcc<float, C, V_DIM, C, V_DIM> qkv_store;
        TASSIGN(qkv_store, 0);
        TSTORE(qkv_global, qkv_store);
      }

      // ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (1 << 8));
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
      SetCrossFlag<PIPE_FIX>(1);
#else
      SignalBothVecOnA5<PIPE_FIX>(1);
#endif
    }
  }

  SYNCALL<SyncCoreType::Mix>();
#endif

#if defined(__DAV_VEC__)

  // ── Vec UB address plan (192 KB budget) ──────────────────────────────────
  // MASK_UB [HalfC, C] fp32 — loaded once; used in every chunk.
  constexpr int32_t MASK_UB_ADDR = 0;
  constexpr int32_t SLOT_A_ADDR = MASK_UB_ADDR + HalfC * C * sizeof(float);
  constexpr int32_t SLOT_B_ADDR = SLOT_A_ADDR + HalfC * K_DIM * sizeof(float);
  constexpr int32_t SLOT_C_ADDR = SLOT_B_ADDR + HalfC * K_DIM * sizeof(float);
  constexpr int32_t SLOT_D_ADDR = SLOT_C_ADDR + HalfC * K_DIM * sizeof(float);
  set_mask_norm();
  set_vector_mask(-1, -1);

  auto vid = get_subblockid();
  int32_t my_row_offset = static_cast<int32_t>(vid) * HalfC;

  // NOTE: mask is read directly from GM in the Aqk loop below (per column).

  for (int64_t wi = 0; wi < (total_work + block_num - 1) / block_num; ++wi) {
    int64_t pid = wi * block_num + cid;
    if (pid >= total_work) break;

    int64_t head = pid % H;
    int64_t seq_idx = pid / H;

    int64_t bos, slen;
    int64_t chunk_offset = 0;
    if (cu_seqlens != nullptr) {
      bos = static_cast<int64_t>(cu_seqlens[seq_idx]);
      int64_t eos = static_cast<int64_t>(cu_seqlens[seq_idx + 1]);
      slen = eos - bos;
      for (int64_t si = 0; si < seq_idx; ++si) {
        int64_t sb = static_cast<int64_t>(cu_seqlens[si]);
        int64_t se = static_cast<int64_t>(cu_seqlens[si + 1]);
        chunk_offset += (se - sb + C - 1) / C;
      }
    } else {
      bos = seq_idx * seq_len;
      slen = seq_len;
      chunk_offset = seq_idx * ((seq_len + C - 1) / C);
    }
    int64_t num_chunks = (slen + C - 1) / C;
    int64_t ws_base = static_cast<int64_t>(cid) * WS_PER_CORE;

    for (int32_t ci = 0; ci < static_cast<int32_t>(num_chunks); ++ci) {
      int64_t chunk_start = bos + static_cast<int64_t>(ci) * C;
      int64_t valid = slen - static_cast<int64_t>(ci) * C;
      if (valid > C) valid = C;
      int32_t valid_rows =
          static_cast<int32_t>(valid - static_cast<int64_t>(vid) * HalfC);
      if (valid_rows < 0) valid_rows = 0;
      if (valid_rows > HalfC) valid_rows = HalfC;

      // ====================================================================
      // PHASE A — load Q, K, G_cs; pre-scale q_eff/k_eff; write V_corr, S.
      // ====================================================================
      int64_t hk_base =
          static_cast<int64_t>(head) * total_tokens * K_DIM +
          (chunk_start + static_cast<int64_t>(vid) * HalfC) * K_DIM;

      TileUbDataND<float, HalfC, K_DIM, HalfC, K_DIM, pto::PadValue::Zero> g_ub;
      TASSIGN(g_ub, SLOT_A_ADDR);
      TileUbDataND<float, HalfC, K_DIM, HalfC, K_DIM, pto::PadValue::Zero> q_ub;
      TASSIGN(q_ub, SLOT_B_ADDR);
      TileUbDataND<float, HalfC, K_DIM, HalfC, K_DIM> exp_ub;
      TASSIGN(exp_ub, SLOT_C_ADDR);

      // (A.1) Load Q and G_cs (head-major fp16).
      if (valid_rows > 0) {
        {
          GmShape2D q_shape(valid_rows, K_DIM);
          GmStride2D q_stride(HM_STRIDE);
          GmTensor2D<half> q_global(Q_handle + hk_base, q_shape, q_stride);
          TileUbDataND<half, HalfC, K_DIM, HalfC, K_DIM, pto::PadValue::Zero>
              q_stg_full;
          TASSIGN(q_stg_full, SLOT_D_ADDR);
          DynVecTile<half, HalfC, K_DIM, pto::PadValue::Zero> q_load(valid_rows,
                                                                     K_DIM);
          TASSIGN(q_load, SLOT_D_ADDR);
          TLOAD(q_load, q_global);
          if (valid_rows != HalfC) {
            TFILLPAD_INPLACE(q_stg_full, q_load);
          }
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        {
          TileUbDataND<half, HalfC, K_DIM, HalfC, K_DIM> q_stg_cvt;
          TASSIGN(q_stg_cvt, SLOT_D_ADDR);
          TCVT(q_ub, q_stg_cvt, pto::RoundMode::CAST_NONE);
          PipeBarrierVec();
        }
        {
          GmShape2D g_shape(valid_rows, K_DIM);
          GmStride2D g_stride(HM_STRIDE);
          GmTensor2D<float> g_global(G_handle + hk_base, g_shape, g_stride);
          TileUbDataND<float, HalfC, K_DIM, HalfC, K_DIM, pto::PadValue::Zero>
              g_stg_full;
          TASSIGN(g_stg_full, SLOT_A_ADDR);
          DynVecTile<float, HalfC, K_DIM, pto::PadValue::Zero> g_load(
              valid_rows, K_DIM);
          TASSIGN(g_load, SLOT_A_ADDR);
          TLOAD(g_load, g_global);
          if (valid_rows != HalfC) {
            TFILLPAD_INPLACE(g_stg_full, g_load);
          }
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      } else {
        TEXPANDS(q_ub, 0.0f);
        TEXPANDS(g_ub, 0.0f);
      }

      // (A.2) q_eff = Q * exp(g_cs).
      TEXP(exp_ub, g_ub);
      PipeBarrierVec();
      TMUL(exp_ub, q_ub, exp_ub);
      PipeBarrierVec();

      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      {
        GmShape2D q_shape(HalfC, K_DIM);
        GmStride2D q_stride(K_DIM);
        GmTensor2D<float> q_global(
            workspace_handle + ws_base + WS_Q +
                static_cast<int64_t>(vid) * HalfC * K_DIM,
            q_shape, q_stride);
        DynVecTile<float, HalfC, K_DIM> q_store(HalfC, K_DIM);
        TASSIGN(q_store, SLOT_C_ADDR);
        TSTORE(q_global, q_store);
      }

      // (A.3) Aqk matrix (element-wise stable) → WS_QK (masked).
      pipe_barrier(PIPE_ALL);
      {
        constexpr int32_t AQK_GC = SLOT_D_ADDR + HalfC * K_DIM * 4;
        constexpr int32_t AQK_KC = AQK_GC + K_DIM * 4;
        constexpr int32_t AQK_KCH = AQK_KC + K_DIM * 4;
        constexpr int32_t AQK_COL = AQK_KCH + K_DIM * 2;
        constexpr int32_t AQK_MSK = AQK_COL + HalfC * 16 * 4;

        {
          TileUbDataND<float, HalfC, C, HalfC, C> zero_ub;
          TASSIGN(zero_ub, SLOT_C_ADDR);
          TEXPANDS(zero_ub, 0.0f);
          PipeBarrierVec();
          set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
          wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
          GmShape2D z_shape(HalfC, C);
          GmStride2D z_stride(C);
          GmTensor2D<float> z_global(
              workspace_handle + ws_base + WS_QK +
                  static_cast<int64_t>(my_row_offset) * C,
              z_shape, z_stride);
          DynVecTile<float, HalfC, C> z_store(HalfC, C);
          TASSIGN(z_store, SLOT_C_ADDR);
          TSTORE(z_global, z_store);
          set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        }

        for (int32_t c = 0; c < static_cast<int32_t>(valid); ++c) {
          int64_t col_base = static_cast<int64_t>(head) * total_tokens * K_DIM +
                             (chunk_start + static_cast<int64_t>(c)) * K_DIM;
          {
            GmShape2D cs(1, K_DIM);
            GmStride2D cst(K_DIM);
            GmTensor2D<float> gc_gm(G_handle + col_base, cs, cst);
            TileUbDataND<float, 1, K_DIM, 1, K_DIM> gc_ld;
            TASSIGN(gc_ld, AQK_GC);
            TLOAD(gc_ld, gc_gm);
            GmTensor2D<half> kc_gm(K_handle + col_base, cs, cst);
            TileUbDataND<half, 1, K_DIM, 1, K_DIM> kc_ld;
            TASSIGN(kc_ld, AQK_KCH);
            TLOAD(kc_ld, kc_gm);
          }
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          {
            TileUbDataND<half, 1, K_DIM, 1, K_DIM> kc_h;
            TASSIGN(kc_h, AQK_KCH);
            TileUbDataND<float, 1, K_DIM, 1, K_DIM> kc_f;
            TASSIGN(kc_f, AQK_KC);
            TCVT(kc_f, kc_h, pto::RoundMode::CAST_NONE);
            PipeBarrierVec();
          }
          TileUbDataND<float, 1, K_DIM, 1, K_DIM> gc;
          TASSIGN(gc, AQK_GC);
          TileUbDataND<float, 1, K_DIM, 1, K_DIM> kc;
          TASSIGN(kc, AQK_KC);
          TileUbDataND<float, HalfC, K_DIM, HalfC, K_DIM> diff;
          TASSIGN(diff, SLOT_C_ADDR);
          TileUbDataND<float, HalfC, K_DIM, HalfC, K_DIM> tmp;
          TASSIGN(tmp, SLOT_D_ADDR);
          TileUbDataND<float, HalfC, 16, HalfC, 1> colsum;
          TASSIGN(colsum, AQK_COL);

          TCOLEXPANDSUB(diff, g_ub, gc);
          PipeBarrierVec();
          TMINS(diff, diff, 0.0f);
          PipeBarrierVec();
          TEXP(diff, diff);
          PipeBarrierVec();
          TCOLEXPANDMUL(diff, diff, kc);
          PipeBarrierVec();
          TMUL(diff, diff, q_ub);
          PipeBarrierVec();
          TROWSUM(colsum, diff, tmp);
          PipeBarrierVec();
          {
            TileUbDataND<float, HalfC, 16, HalfC, 1> mk;
            TASSIGN(mk, AQK_MSK);
            GmShape2D ms(HalfC, 1);
            GmStride2D mst(C);
            GmTensor2D<float> mk_gm(
                Mask_handle + static_cast<int64_t>(my_row_offset) * C + c, ms,
                mst);
            TLOAD(mk, mk_gm);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            TMUL(colsum, colsum, mk);
            PipeBarrierVec();
          }
          set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
          wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
          {
            GmShape2D qs2(HalfC, 1);
            GmStride2D qst2(C);
            GmTensor2D<float> qk_col(
                workspace_handle + ws_base + WS_QK +
                    static_cast<int64_t>(my_row_offset) * C + c,
                qs2, qst2);
            TileUbDataND<float, HalfC, 16, HalfC, 1> col_st;
            TASSIGN(col_st, AQK_COL);
            TSTORE(qk_col, col_st);
          }
          set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
          pipe_barrier(PIPE_ALL);
        }
      }

      // (A.4) Load V_corr fp16 (BSND), cast to fp32, store to WS_V.
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      {
        TileUbDataND<half, HalfC, V_DIM, HalfC, V_DIM, pto::PadValue::Zero>
            vh_ub;
        TASSIGN(vh_ub, SLOT_D_ADDR);
        TileUbDataND<float, HalfC, V_DIM, HalfC, V_DIM> v_f_ub;
        TASSIGN(v_f_ub, SLOT_A_ADDR);

        int64_t v_offset = (chunk_start * H + head) * V_DIM +
                           static_cast<int64_t>(vid) * HalfC * BSND_STRIDE;
        if (valid_rows > 0) {
          GmShape2D v_shape(valid_rows, V_DIM);
          GmStride2D v_stride(BSND_STRIDE);
          GmTensor2D<half> v_global(V_handle + v_offset, v_shape, v_stride);
          DynVecTile<half, HalfC, V_DIM, pto::PadValue::Zero> v_load(valid_rows,
                                                                     V_DIM);
          TASSIGN(v_load, SLOT_D_ADDR);
          TLOAD(v_load, v_global);
          if (valid_rows != HalfC) {
            TFILLPAD_INPLACE(vh_ub, v_load);
          }
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          TCVT(v_f_ub, vh_ub, pto::RoundMode::CAST_NONE);
          PipeBarrierVec();
        } else {
          TEXPANDS(v_f_ub, 0.0f);
          PipeBarrierVec();
        }

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        GmShape2D vw_shape(HalfC, V_DIM);
        GmStride2D vw_stride(V_DIM);
        GmTensor2D<float> vw_global(
            workspace_handle + ws_base + WS_V +
                static_cast<int64_t>(vid) * HalfC * V_DIM,
            vw_shape, vw_stride);
        DynVecTile<float, HalfC, V_DIM> v_store(HalfC, V_DIM);
        TASSIGN(v_store, SLOT_A_ADDR);
        TSTORE(vw_global, v_store);
      }

      // (A.5) Load S fp16 from snapshots, cast to fp32, store to WS_S.
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      {
        TileUbDataND<half, HalfC, V_DIM, HalfC, V_DIM> sh_ub;
        TASSIGN(sh_ub, SLOT_D_ADDR);
        TileUbDataND<float, HalfC, V_DIM, HalfC, V_DIM> s_f_ub;
        TASSIGN(s_f_ub, SLOT_A_ADDR);

        int64_t s_in_offset =
            (chunk_offset + static_cast<int64_t>(ci)) * H * KV +
            static_cast<int64_t>(head) * KV +
            static_cast<int64_t>(vid) * HalfC * V_DIM;
        GmShape2D s_shape(HalfC, V_DIM);
        GmStride2D s_stride(V_DIM);
        GmTensor2D<half> s_global(S_handle + s_in_offset, s_shape, s_stride);
        DynVecTile<half, HalfC, V_DIM> s_load(HalfC, V_DIM);
        TASSIGN(s_load, SLOT_D_ADDR);
        TLOAD(s_load, s_global);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TCVT(s_f_ub, sh_ub, pto::RoundMode::CAST_NONE);
        PipeBarrierVec();

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        GmShape2D sw_shape(HalfC, V_DIM);
        GmStride2D sw_stride(V_DIM);
        GmTensor2D<float> sw_global(
            workspace_handle + ws_base + WS_S +
                static_cast<int64_t>(vid) * HalfC * V_DIM,
            sw_shape, sw_stride);
        DynVecTile<float, HalfC, V_DIM> s_store(HalfC, V_DIM);
        TASSIGN(s_store, SLOT_A_ADDR);
        TSTORE(sw_global, s_store);
      }

      // (A.6) Signal Cube: phase A workspace ready.
      pipe_barrier(PIPE_ALL);
      ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (0 << 8));

      // ====================================================================
      // PHASE C — wait QS + QKV from Cube; combine O = QS + QKV; write GM.
      // ====================================================================
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
      wait_flag_dev(1);
#else
      wait_intra_block(PIPE_MTE3, 1);
#endif
      pipe_barrier(PIPE_ALL);

      if (valid_rows > 0) {
        TileUbDataND<float, HalfC, V_DIM, HalfC, V_DIM> qs_ub;
        TASSIGN(qs_ub, SLOT_A_ADDR);
        TileUbDataND<float, HalfC, V_DIM, HalfC, V_DIM> qkv_ub;
        TASSIGN(qkv_ub, SLOT_B_ADDR);

        {
          GmShape2D qs_shape(HalfC, V_DIM);
          GmStride2D qs_stride(V_DIM);
          GmTensor2D<float> qs_global(
              workspace_handle + ws_base + WS_QS +
                  static_cast<int64_t>(vid) * HalfC * V_DIM,
              qs_shape, qs_stride);
          DynVecTile<float, HalfC, V_DIM> qs_load(HalfC, V_DIM);
          TASSIGN(qs_load, SLOT_A_ADDR);
          TLOAD(qs_load, qs_global);
        }
        {
          GmShape2D qkv_shape(HalfC, V_DIM);
          GmStride2D qkv_stride(V_DIM);
          GmTensor2D<float> qkv_global(
              workspace_handle + ws_base + WS_QKV +
                  static_cast<int64_t>(vid) * HalfC * V_DIM,
              qkv_shape, qkv_stride);
          DynVecTile<float, HalfC, V_DIM> qkv_load(HalfC, V_DIM);
          TASSIGN(qkv_load, SLOT_B_ADDR);
          TLOAD(qkv_load, qkv_global);
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        TADD(qs_ub, qs_ub, qkv_ub);
        PipeBarrierVec();

        TileUbDataND<half, HalfC, V_DIM, HalfC, V_DIM> oh_ub;
        TASSIGN(oh_ub, SLOT_D_ADDR);
        TCVT(oh_ub, qs_ub, pto::RoundMode::CAST_NONE);
        PipeBarrierVec();
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        int64_t o_offset = (chunk_start * H + head) * V_DIM +
                           static_cast<int64_t>(vid) * HalfC * BSND_STRIDE;
        GmShape2D o_shape(valid_rows, V_DIM);
        GmStride2D o_stride(BSND_STRIDE);
        GmTensor2D<half> o_global(O_handle + o_offset, o_shape, o_stride);
        DynVecTile<half, HalfC, V_DIM> o_store(valid_rows, V_DIM);
        TASSIGN(o_store, SLOT_D_ADDR);
        TSTORE(o_global, o_store);
      }
      pipe_barrier(PIPE_ALL);
    }
  }

  SYNCALL<SyncCoreType::Mix>();
#endif
}

extern "C" __global__ AICORE void kda_chunk_o(
    __gm__ uint8_t* Q, __gm__ uint8_t* K, __gm__ uint8_t* V_corr,
    __gm__ uint8_t* S, __gm__ uint8_t* G, __gm__ uint8_t* Mask,
    __gm__ uint8_t* workspace, __gm__ uint8_t* O, __gm__ uint8_t* cu_seqlens,
    int64_t batch_size, int64_t seq_len, int64_t total_tokens) {
  kda_chunk_o_kernel<GDN_H, GDN_D, GDN_C>(
      reinterpret_cast<__gm__ half*>(Q), reinterpret_cast<__gm__ half*>(K),
      reinterpret_cast<__gm__ half*>(V_corr), reinterpret_cast<__gm__ half*>(S),
      reinterpret_cast<__gm__ float*>(G), reinterpret_cast<__gm__ float*>(Mask),
      reinterpret_cast<__gm__ float*>(workspace),
      reinterpret_cast<__gm__ half*>(O),
      reinterpret_cast<__gm__ int32_t*>(cu_seqlens), batch_size, seq_len,
      total_tokens);
}

// Host-callable launch shims: the `<<<>>>` syntax is only
// understood by the kernel compiler, so the launch lives here
// rather than in the host wrappers under csrc/host/.
extern "C" void pto_launch_kda_chunk_o(uint32_t blockDim, void* stream, void* Q,
                                       void* K, void* V_corr, void* S, void* G,
                                       void* Mask, void* workspace, void* O,
                                       void* cu_seqlens, int64_t batch_size,
                                       int64_t seq_len, int64_t total_tokens) {
  kda_chunk_o<<<blockDim, nullptr, stream>>>(
      (__gm__ uint8_t*)Q, (__gm__ uint8_t*)K, (__gm__ uint8_t*)V_corr,
      (__gm__ uint8_t*)S, (__gm__ uint8_t*)G, (__gm__ uint8_t*)Mask,
      (__gm__ uint8_t*)workspace, (__gm__ uint8_t*)O,
      (__gm__ uint8_t*)cu_seqlens, batch_size, seq_len, total_tokens);
}
