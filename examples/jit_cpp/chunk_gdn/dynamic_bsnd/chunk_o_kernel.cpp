#include <pto/pto-inst.hpp>
#include <type_traits>
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

#ifdef __CCE_AICORE__

namespace {

using GmShape2D = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;
using GmStride2D = pto::Stride<1, 1, 1, pto::DYNAMIC, 1>;

template <typename T>
using GmTensor2D = pto::GlobalTensor<T, GmShape2D, GmStride2D>;

template <typename T, int32_t Rows, int32_t Cols>
using DynMatL1 = pto::Tile<pto::TileType::Mat, T, Rows, Cols,
                           pto::BLayout::ColMajor, pto::DYNAMIC,
                           pto::DYNAMIC, pto::SLayout::RowMajor, 512,
                           pto::PadValue::Zero>;

template <typename T, int32_t Rows, int32_t Cols,
          pto::PadValue PadVal = pto::PadValue::Null>
using DynVecTile = pto::Tile<pto::TileType::Vec, T, Rows, Cols,
                             pto::BLayout::RowMajor, pto::DYNAMIC,
                             pto::DYNAMIC, pto::SLayout::NoneBox, 512, PadVal>;

template <typename T, int32_t Rows, int32_t Cols>
using DynAccTile = pto::TileAcc<T, Rows, Cols, pto::DYNAMIC, pto::DYNAMIC>;

template <typename T, int Rows, int Cols, int RowValid = Rows,
          int ColValid = Cols>
using TileMatL1 = pto::Tile<pto::TileType::Mat, T, Rows, Cols,
                            pto::BLayout::ColMajor, RowValid, ColValid,
                            pto::SLayout::RowMajor, 512, pto::PadValue::Zero>;

template <typename T, int Rows, int Cols, int RowValid = Rows,
          int ColValid = Cols>
using TileMatL1ZN = pto::Tile<pto::TileType::Mat, T, Rows, Cols,
                              pto::BLayout::RowMajor, RowValid, ColValid,
                              pto::SLayout::ColMajor, 512,
                              pto::PadValue::Zero>;

template <typename T, int Rows, int Cols, int RowValid = Rows,
          int ColValid = Cols>
using TileMatL0A = pto::Tile<pto::TileType::Left, T, Rows, Cols,
                             pto::BLayout::RowMajor, RowValid, ColValid,
                             pto::SLayout::RowMajor, 512,
                             pto::PadValue::Zero>;

template <typename T, int Rows, int Cols, int RowValid = Rows,
          int ColValid = Cols>
using TileMatL0B = pto::Tile<pto::TileType::Right, T, Rows, Cols,
                             pto::BLayout::RowMajor, RowValid, ColValid,
                             pto::SLayout::ColMajor, 512,
                             pto::PadValue::Zero>;

template <typename T, int Rows, int Cols, int RowValid = Rows,
          int ColValid = Cols, pto::PadValue PadVal = pto::PadValue::Null>
using TileUbDataND =
    pto::Tile<pto::TileType::Vec, T, Rows, Cols, pto::BLayout::RowMajor,
              RowValid, ColValid, pto::SLayout::NoneBox, 512, PadVal>;

template <typename T, int Rows, int Cols, int RowValid = Rows,
          int ColValid = Cols, pto::PadValue PadVal = pto::PadValue::Null>
using TileUbDataDN =
    pto::Tile<pto::TileType::Vec, T, Rows, Cols, pto::BLayout::ColMajor,
              RowValid, ColValid, pto::SLayout::NoneBox, 512, PadVal>;

template <typename T1, typename T2, uint32_t M, uint32_t N, uint32_t K,
          uint32_t validM = M, uint32_t validN = N, uint32_t validK = K,
          uint32_t K_tail, bool transpose_A = false,
          bool transpose_B = false>
AICORE PTO_INLINE void
gemm_v0(std::conditional_t<transpose_A, TileMatL1<T1, K, M, validK, validM>,
                           TileMatL1<T1, M, K, validM, validK>> &A,
        std::conditional_t<transpose_B, TileMatL1<T1, N, K, validN, validK>,
                           TileMatL1<T1, K, N, validK, validN>> &B,
        pto::TileAcc<T2, M, N, validM, validN> &C, bool clear)
{
  constexpr uint32_t kL0Size = 128;
  const uint32_t kL0split = (K + kL0Size - 1) / kL0Size;

  auto war_event_id = (event_t)(((int)EVENT_ID0 + 1) % 8);
  set_flag(PIPE_MTE2, PIPE_MTE1, war_event_id);
  wait_flag(PIPE_MTE2, PIPE_MTE1, war_event_id);

  for (uint32_t kL0Idx = 0; kL0Idx < kL0split; ++kL0Idx) {
    const bool initflag = clear && (kL0Idx == 0);
    const bool is_tail_block = (kL0Idx == kL0split - 1);

    if (is_tail_block) {
      TileMatL0A<T1, M, K_tail, M, K_tail> l0a;
      TileMatL0B<T1, K_tail, N, K_tail, N> l0b;
      pto::TASSIGN(l0a, 0x0);
      pto::TASSIGN(l0b, 0x0);

      set_flag(PIPE_M, PIPE_MTE1, war_event_id);
      wait_flag(PIPE_M, PIPE_MTE1, war_event_id);

      if constexpr (!transpose_A) {
        pto::TEXTRACT(l0a, A, 0, kL0Idx * K_tail);
      } else {
        TileMatL1ZN<T1, M, K, validM, validK> A_t;
        pto::TRESHAPE(A_t, A);
        pto::TEXTRACT(l0a, A_t, 0, kL0Idx * K_tail);
      }

      if constexpr (!transpose_B) {
        pto::TEXTRACT(l0b, B, kL0Idx * K_tail, 0);
      } else {
        TileMatL1ZN<T1, K, N, validK, validN> B_t;
        pto::TRESHAPE(B_t, B);
        pto::TEXTRACT(l0b, B_t, kL0Idx * K_tail, 0);
      }

      set_flag(PIPE_MTE1, PIPE_M, war_event_id);
      wait_flag(PIPE_MTE1, PIPE_M, war_event_id);

      if (initflag) {
        pto::TMATMUL(C, l0a, l0b);
      } else {
        pto::TMATMUL_ACC(C, C, l0a, l0b);
      }
    } else {
      TileMatL0A<T1, M, kL0Size, M, kL0Size> l0a;
      TileMatL0B<T1, kL0Size, N, kL0Size, N> l0b;
      pto::TASSIGN(l0a, 0x0);
      pto::TASSIGN(l0b, 0x0);

      set_flag(PIPE_M, PIPE_MTE1, war_event_id);
      wait_flag(PIPE_M, PIPE_MTE1, war_event_id);

      set_flag(PIPE_FIX, PIPE_M, war_event_id);
      wait_flag(PIPE_FIX, PIPE_M, war_event_id);

      if constexpr (!transpose_A) {
        pto::TEXTRACT(l0a, A, 0, kL0Idx * kL0Size);
      } else {
        TileMatL1ZN<T1, M, K, validM, validK> A_t;
        pto::TRESHAPE(A_t, A);
        pto::TEXTRACT(l0a, A_t, 0, kL0Idx * kL0Size);
      }

      if constexpr (!transpose_B) {
        pto::TEXTRACT(l0b, B, kL0Idx * kL0Size, 0);
      } else {
        TileMatL1ZN<T1, K, N, validK, validN> B_t;
        pto::TRESHAPE(B_t, B);
        pto::TEXTRACT(l0b, B_t, kL0Idx * kL0Size, 0);
      }

      set_flag(PIPE_MTE1, PIPE_M, war_event_id);
      wait_flag(PIPE_MTE1, PIPE_M, war_event_id);

      if (initflag) {
        pto::TMATMUL(C, l0a, l0b);
      } else {
        pto::TMATMUL_ACC(C, C, l0a, l0b);
      }

      set_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);
      wait_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);
    }
  }

  set_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);
  wait_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);

  set_flag(PIPE_M, PIPE_FIX, war_event_id);
  wait_flag(PIPE_M, PIPE_FIX, war_event_id);
}

} // namespace

#endif

template <int32_t NumHeads, int32_t HiddenSize, int32_t ChunkSize>
AICORE void chunk_o_kernel(
    __gm__ half *Q_handle, __gm__ half *K_handle, __gm__ half *V_handle,
    __gm__ half *S_handle, __gm__ float *G_handle,
    __gm__ float *Msk_handle,
    __gm__ half *workspace_qk_handle,
    __gm__ half *workspace_qs_qkv_handle,
    __gm__ half *workspace_qk_gated_handle,
    __gm__ half *O_handle,
    __gm__ int32_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len, int64_t total_tokens,
    uint64_t ffts_addr)
{
  constexpr int32_t HalfChunk = ChunkSize / 2;
  constexpr uint32_t KTail =
      (HiddenSize % 128 == 0) ? 128 : (HiddenSize % 128);
  constexpr uint32_t CTail =
      (ChunkSize % 128 == 0) ? 128 : (ChunkSize % 128);

  constexpr int32_t WsQKSize = ChunkSize * ChunkSize;
  constexpr int32_t WsQSSize = ChunkSize * HiddenSize;
  constexpr int32_t WsGatedSize = ChunkSize * ChunkSize;

  constexpr int32_t GUbAddr      = 0;
  constexpr int32_t MskUbAddr    = 512;
  constexpr int32_t QKUbAddr     = 33280;
  constexpr int32_t GvUbAddr     = 66048;
  constexpr int32_t CoeffUbAddr  = 66304;
  constexpr int32_t QKHalfUbAddr = 99072;
  constexpr int32_t QSHalfUbAddr = 115456;
  constexpr int32_t QSUbAddr     = 131840;
  constexpr int32_t OHalfUbAddr  = 164608;
  constexpr int32_t GBlockUbAddr = QKUbAddr;
  constexpr int32_t OUbAddr      = QKUbAddr;

  set_ffts_base_addr(ffts_addr);
  auto cid = get_block_idx();
  auto block_num = get_block_num();
  auto vid = get_subblockid();

  int64_t num_seqs = batch_size;

  TileMatL1<half, ChunkSize, HiddenSize,
            ChunkSize, HiddenSize> q_l1;
  TASSIGN(q_l1, 0);
  TileMatL1<half, ChunkSize, HiddenSize,
            ChunkSize, HiddenSize> k_l1;
  TASSIGN(k_l1, 32768);
  TileAcc<float, ChunkSize, ChunkSize,
          ChunkSize, ChunkSize> qk_l0;
  TASSIGN(qk_l0, 0);
  TileMatL1<half, HiddenSize, HiddenSize,
            HiddenSize, HiddenSize> s_l1;
  TASSIGN(s_l1, 65536);
  TileAcc<float, ChunkSize, HiddenSize,
          ChunkSize, HiddenSize> qs_l0;
  TASSIGN(qs_l0, 65536);
  TileMatL1<half, ChunkSize, ChunkSize,
            ChunkSize, ChunkSize> qk_gated_l1;
  TASSIGN(qk_gated_l1, 98304);
  TileMatL1<half, ChunkSize, HiddenSize,
            ChunkSize, HiddenSize> v_l1;
  TASSIGN(v_l1, 131072);
  TileAcc<float, ChunkSize, HiddenSize,
          ChunkSize, HiddenSize> qkv_l0;
  TASSIGN(qkv_l0, 0);

  TileUbDataND<float, 1, ChunkSize,
               1, ChunkSize, pto::PadValue::Zero> g_ub;
  TASSIGN(g_ub, GUbAddr);
  TileUbDataND<float, HalfChunk, ChunkSize,
               HalfChunk, ChunkSize> msk_ub;
  TASSIGN(msk_ub, MskUbAddr);
  TileUbDataND<float, HalfChunk, ChunkSize,
               HalfChunk, ChunkSize> qk_ub;
  TASSIGN(qk_ub, QKUbAddr);
  TileUbDataND<float, 1, HalfChunk,
               1, HalfChunk> g_v_ub;
  TASSIGN(g_v_ub, GvUbAddr);
  TileUbDataND<float, HalfChunk, ChunkSize,
               HalfChunk, ChunkSize> coeff_ub;
  TASSIGN(coeff_ub, CoeffUbAddr);
  TileUbDataND<half, HalfChunk, ChunkSize,
               HalfChunk, ChunkSize> qk_ub_half;
  TASSIGN(qk_ub_half, QKHalfUbAddr);
  TileUbDataND<half, HalfChunk, HiddenSize,
               HalfChunk, HiddenSize> qs_ub_half;
  TASSIGN(qs_ub_half, QSHalfUbAddr);
  TileUbDataND<float, HalfChunk, HiddenSize,
               HalfChunk, HiddenSize> qs_ub;
  TASSIGN(qs_ub, QSUbAddr);
  TileUbDataND<half, HalfChunk, HiddenSize,
               HalfChunk, HiddenSize> o_ub_half;
  TASSIGN(o_ub_half, OHalfUbAddr);
  TileUbDataND<float, HalfChunk, HiddenSize,
               HalfChunk, HiddenSize> o_ub;
  TASSIGN(o_ub, OUbAddr);

  int64_t total_work = 0;
  if (cu_seqlens == nullptr) {
    int64_t chunks_per_seq = (seq_len + ChunkSize - 1) / ChunkSize;
    total_work = num_seqs * chunks_per_seq * NumHeads;
  }

#if defined(__DAV_C220_CUBE__)
  if (cu_seqlens == nullptr) {
    int64_t chunks_per_seq = (seq_len + ChunkSize - 1) / ChunkSize;
    int64_t global_chunk_base = 0;
    bool first_cube_iter = true;

    for (int64_t work_idx = static_cast<int64_t>(cid);
         work_idx < total_work;
         work_idx += static_cast<int64_t>(block_num)) {
      if (!first_cube_iter) wait_flag_dev(3);
      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

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

      int64_t qkv_offset =
          (chunk_token_start * NumHeads + head_idx) *
          static_cast<int64_t>(HiddenSize);

      int64_t chunk_global_idx = seq_idx * chunks_per_seq + ci;
      int64_t s_offset =
          (chunk_global_idx * NumHeads + head_idx) *
          static_cast<int64_t>(HiddenSize) *
          static_cast<int64_t>(HiddenSize);

      {
        GmShape2D qkv_shape(valid_rows, HiddenSize);
        GmStride2D qkv_stride(NumHeads * HiddenSize);
        GmTensor2D<half> q_global(Q_handle + qkv_offset, qkv_shape, qkv_stride);
        GmTensor2D<half> k_global(K_handle + qkv_offset, qkv_shape, qkv_stride);
        DynMatL1<half, ChunkSize, HiddenSize> q_l1_load(valid_rows, HiddenSize);
        DynMatL1<half, ChunkSize, HiddenSize> k_l1_load(valid_rows, HiddenSize);
        TASSIGN(q_l1_load, 0);
        TASSIGN(k_l1_load, 32768);
        TLOAD(q_l1_load, q_global);
        TLOAD(k_l1_load, k_global);
        if (valid_rows != ChunkSize) {
          TFILLPAD(q_l1_load, q_l1_load);
          TFILLPAD(k_l1_load, k_l1_load);
        }
      }

      // Build raw chunk-local QK scores before the vector stage applies g.
      gemm_v0<half, float,
          ChunkSize, ChunkSize, HiddenSize,
          ChunkSize, ChunkSize, HiddenSize,
          KTail, false, true>(q_l1, k_l1, qk_l0, true);

      {
        GmShape2D s_shape(HiddenSize, HiddenSize);
        GmStride2D s_stride(HiddenSize);
        GmTensor2D<half> s_global(S_handle + s_offset, s_shape, s_stride);
        DynMatL1<half, HiddenSize, HiddenSize> s_l1_load(HiddenSize,
                                                         HiddenSize);
        TASSIGN(s_l1_load, 65536);
        TLOAD(s_l1_load, s_global);
      }

      // Compute the recurrent Q*S contribution that will be added back later.
      gemm_v0<half, float,
          ChunkSize, HiddenSize, HiddenSize,
          ChunkSize, HiddenSize, HiddenSize,
          KTail, false, false>(q_l1, s_l1, qs_l0, true);

      {
        GmShape2D qk_shape(ChunkSize, ChunkSize);
        GmStride2D qk_stride(ChunkSize);
        GmTensor2D<half> qk_workspace(
            workspace_qk_handle + static_cast<int64_t>(cid) * WsQKSize,
            qk_shape, qk_stride);
        DynAccTile<float, ChunkSize, ChunkSize> qk_store(ChunkSize,
                                                         ChunkSize);
        TASSIGN(qk_store, 0);
        TSTORE(qk_workspace, qk_store);
      }

      {
        GmShape2D qs_shape(ChunkSize, HiddenSize);
        GmStride2D qs_stride(HiddenSize);
        GmTensor2D<half> qs_workspace(
            workspace_qs_qkv_handle + static_cast<int64_t>(cid) * WsQSSize,
            qs_shape, qs_stride);
        DynAccTile<float, ChunkSize, HiddenSize> qs_store(ChunkSize,
                                                          HiddenSize);
        TASSIGN(qs_store, 65536);
        TSTORE(qs_workspace, qs_store);
      }

      ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (0 << 8));

      wait_flag_dev(1);

      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

      {
        GmShape2D qk_gated_shape(ChunkSize, ChunkSize);
        GmStride2D qk_gated_stride(ChunkSize);
        GmTensor2D<half> qk_gated_global(
            workspace_qk_gated_handle +
                static_cast<int64_t>(cid) * WsGatedSize,
            qk_gated_shape, qk_gated_stride);
        DynMatL1<half, ChunkSize, ChunkSize> qk_gated_load(ChunkSize,
                                                           ChunkSize);
        TASSIGN(qk_gated_load, 98304);
        TLOAD(qk_gated_load, qk_gated_global);
      }
      {
        GmShape2D v_shape(valid_rows, HiddenSize);
        GmStride2D v_stride(NumHeads * HiddenSize);
        GmTensor2D<half> v_global(V_handle + qkv_offset, v_shape, v_stride);
        DynMatL1<half, ChunkSize, HiddenSize> v_l1_load(valid_rows,
                                                        HiddenSize);
        TASSIGN(v_l1_load, 131072);
        TLOAD(v_l1_load, v_global);
        if (valid_rows != ChunkSize) {
          TFILLPAD(v_l1_load, v_l1_load);
        }
      }

      // Turn gated QK into the QKV path that will be summed with QS.
      gemm_v0<half, float,
          ChunkSize, HiddenSize, ChunkSize,
          ChunkSize, HiddenSize, ChunkSize,
          CTail, false, false>(qk_gated_l1, v_l1, qkv_l0, true);

      {
        GmShape2D qkv_shape(ChunkSize, HiddenSize);
        GmStride2D qkv_stride(HiddenSize);
        GmTensor2D<half> qkv_workspace(
            workspace_qs_qkv_handle + static_cast<int64_t>(cid) * WsQSSize,
            qkv_shape, qkv_stride);
        DynAccTile<float, ChunkSize, HiddenSize> qkv_store(ChunkSize,
                                                           HiddenSize);
        TASSIGN(qkv_store, 0);
        TSTORE(qkv_workspace, qkv_store);
      }

      ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (2 << 8));
      first_cube_iter = false;
    }
  } else {
    int64_t gi = 0;
    int64_t chunk_global_idx = 0;
    bool first_cube_iter_v = true;
    for (int64_t si = 0; si < num_seqs; ++si) {
      int64_t bos = static_cast<int64_t>(cu_seqlens[si]);
      int64_t eos = static_cast<int64_t>(cu_seqlens[si + 1]);
      int64_t slen = eos - bos;
      int64_t nc = (slen + ChunkSize - 1) / ChunkSize;

      for (int64_t ci = 0; ci < nc; ++ci) {
        for (int32_t h = 0; h < NumHeads; ++h) {
          if (gi % static_cast<int64_t>(block_num) ==
              static_cast<int64_t>(cid)) {
            if (!first_cube_iter_v) wait_flag_dev(3);
            set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

            int64_t chunk_start = ci * ChunkSize;
            int64_t remaining = slen - chunk_start;
            int32_t valid_rows = static_cast<int32_t>(
                remaining < ChunkSize ? remaining : ChunkSize);
            int64_t chunk_token_start = bos + chunk_start;
            int32_t head_idx = h;

            int64_t qkv_offset =
                (chunk_token_start * NumHeads + head_idx) *
                static_cast<int64_t>(HiddenSize);
            int64_t s_offset =
                (chunk_global_idx * NumHeads + head_idx) *
                static_cast<int64_t>(HiddenSize) *
                static_cast<int64_t>(HiddenSize);

            {
              GmShape2D qkv_shape(valid_rows, HiddenSize);
              GmStride2D qkv_stride(NumHeads * HiddenSize);
              GmTensor2D<half> q_global(Q_handle + qkv_offset, qkv_shape,
                                        qkv_stride);
              GmTensor2D<half> k_global(K_handle + qkv_offset, qkv_shape,
                                        qkv_stride);
              DynMatL1<half, ChunkSize, HiddenSize> q_l1_load(valid_rows,
                                                              HiddenSize);
              DynMatL1<half, ChunkSize, HiddenSize> k_l1_load(valid_rows,
                                                              HiddenSize);
              TASSIGN(q_l1_load, 0);
              TASSIGN(k_l1_load, 32768);
              TLOAD(q_l1_load, q_global);
              TLOAD(k_l1_load, k_global);
              if (valid_rows != ChunkSize) {
                TFILLPAD(q_l1_load, q_l1_load);
                TFILLPAD(k_l1_load, k_l1_load);
              }
            }

            gemm_v0<half, float,
                ChunkSize, ChunkSize, HiddenSize,
                ChunkSize, ChunkSize, HiddenSize,
                KTail, false, true>(q_l1, k_l1, qk_l0, true);

            {
              GmShape2D s_shape(HiddenSize, HiddenSize);
              GmStride2D s_stride(HiddenSize);
              GmTensor2D<half> s_global(S_handle + s_offset, s_shape,
                                        s_stride);
              DynMatL1<half, HiddenSize, HiddenSize> s_l1_load(HiddenSize,
                                                               HiddenSize);
              TASSIGN(s_l1_load, 65536);
              TLOAD(s_l1_load, s_global);
            }

            gemm_v0<half, float,
                ChunkSize, HiddenSize, HiddenSize,
                ChunkSize, HiddenSize, HiddenSize,
                KTail, false, false>(q_l1, s_l1, qs_l0, true);

            {
              GmShape2D qk_shape(ChunkSize, ChunkSize);
              GmStride2D qk_stride(ChunkSize);
              GmTensor2D<half> qk_workspace(
                  workspace_qk_handle + static_cast<int64_t>(cid) * WsQKSize,
                  qk_shape, qk_stride);
              DynAccTile<float, ChunkSize, ChunkSize> qk_store(ChunkSize,
                                                               ChunkSize);
              TASSIGN(qk_store, 0);
              TSTORE(qk_workspace, qk_store);
            }

            {
              GmShape2D qs_shape(ChunkSize, HiddenSize);
              GmStride2D qs_stride(HiddenSize);
              GmTensor2D<half> qs_workspace(
                  workspace_qs_qkv_handle +
                      static_cast<int64_t>(cid) * WsQSSize,
                  qs_shape, qs_stride);
              DynAccTile<float, ChunkSize, HiddenSize> qs_store(ChunkSize,
                                                                HiddenSize);
              TASSIGN(qs_store, 65536);
              TSTORE(qs_workspace, qs_store);
            }

            ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (0 << 8));

            wait_flag_dev(1);

            set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

            {
              GmShape2D qk_gated_shape(ChunkSize, ChunkSize);
              GmStride2D qk_gated_stride(ChunkSize);
              GmTensor2D<half> qk_gated_global(
                  workspace_qk_gated_handle +
                      static_cast<int64_t>(cid) * WsGatedSize,
                  qk_gated_shape, qk_gated_stride);
              DynMatL1<half, ChunkSize, ChunkSize> qk_gated_load(ChunkSize,
                                                                 ChunkSize);
              TASSIGN(qk_gated_load, 98304);
              TLOAD(qk_gated_load, qk_gated_global);
            }
            {
              GmShape2D v_shape(valid_rows, HiddenSize);
              GmStride2D v_stride(NumHeads * HiddenSize);
              GmTensor2D<half> v_global(V_handle + qkv_offset, v_shape,
                                        v_stride);
              DynMatL1<half, ChunkSize, HiddenSize> v_l1_load(valid_rows,
                                                              HiddenSize);
              TASSIGN(v_l1_load, 131072);
              TLOAD(v_l1_load, v_global);
              if (valid_rows != ChunkSize) {
                TFILLPAD(v_l1_load, v_l1_load);
              }
            }

            gemm_v0<half, float,
                ChunkSize, HiddenSize, ChunkSize,
                ChunkSize, HiddenSize, ChunkSize,
                CTail, false, false>(qk_gated_l1, v_l1, qkv_l0, true);

            {
              GmShape2D qkv_shape(ChunkSize, HiddenSize);
              GmStride2D qkv_stride(HiddenSize);
              GmTensor2D<half> qkv_workspace(
                  workspace_qs_qkv_handle +
                      static_cast<int64_t>(cid) * WsQSSize,
                  qkv_shape, qkv_stride);
              DynAccTile<float, ChunkSize, HiddenSize> qkv_store(ChunkSize,
                                                                 HiddenSize);
              TASSIGN(qkv_store, 0);
              TSTORE(qkv_workspace, qkv_store);
            }

            ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (2 << 8));
            first_cube_iter_v = false;
          }
          gi++;
        }
        chunk_global_idx++;
      }
    }
  }
#endif

#if defined(__DAV_C220_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);

  {
    GmShape2D msk_shape(HalfChunk, ChunkSize);
    GmStride2D msk_stride(ChunkSize);
    GmTensor2D<float> msk_global(
        Msk_handle + static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
        msk_shape, msk_stride);
    TLOAD(msk_ub, msk_global);
  }
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

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

      {
        GmShape2D g_shape(1, valid_rows);
        GmStride2D g_stride(1);
        GmTensor2D<float> g_global(
            G_handle + static_cast<int64_t>(head_idx) * total_tokens +
                chunk_token_start,
            g_shape, g_stride);
        DynVecTile<float, 1, ChunkSize, pto::PadValue::Zero> g_load(
            1, valid_rows);
        TASSIGN(g_load, GUbAddr);
        TLOAD(g_load, g_global);
        if (valid_rows != ChunkSize) {
          TFILLPAD_INPLACE(g_ub, g_load);
        }
      }
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

      TileUbDataND<float, 1, HalfChunk,
                   1, HalfChunk> g_ub_temp_0;
      TASSIGN(g_ub_temp_0,
              GUbAddr + static_cast<int32_t>(vid) * HalfChunk *
                            static_cast<int32_t>(sizeof(float)));
      TMOV(g_v_ub, g_ub_temp_0);

      TileUbDataND<float, HalfChunk, ChunkSize,
                   HalfChunk, ChunkSize> g_r_2d;
      TASSIGN(g_r_2d, QSUbAddr);
      TileUbDataDN<float, HalfChunk, 1,
                   HalfChunk, 1> g_v_col;
      TASSIGN(g_v_col, GvUbAddr);
      TROWEXPAND(g_r_2d, g_v_col);
      TCOLEXPAND(coeff_ub, g_ub);
      TSUB(coeff_ub, g_r_2d, coeff_ub);
      pipe_barrier(PIPE_V);
      TMINS(coeff_ub, coeff_ub, 0.0f);
      pipe_barrier(PIPE_V);
      TEXP(coeff_ub, coeff_ub);
      pipe_barrier(PIPE_V);
      TMUL(coeff_ub, coeff_ub, msk_ub);
      TEXP(g_v_ub, g_v_ub);

      wait_flag_dev(0);

      {
        GmShape2D qk_shape(HalfChunk, ChunkSize);
        GmStride2D qk_stride(ChunkSize);
        GmTensor2D<half> qk_global(
            workspace_qk_handle +
                static_cast<int64_t>(cid) * WsQKSize +
                static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
            qk_shape, qk_stride);
        TLOAD(qk_ub_half, qk_global);
      }

      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      TCVT(qk_ub, qk_ub_half, pto::RoundMode::CAST_NONE);

      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

      {
        GmShape2D qs_shape(HalfChunk, HiddenSize);
        GmStride2D qs_stride(HiddenSize);
        GmTensor2D<half> qs_global(
            workspace_qs_qkv_handle +
                static_cast<int64_t>(cid) * WsQSSize +
                static_cast<int64_t>(vid) * HalfChunk * HiddenSize,
            qs_shape, qs_stride);
        TLOAD(qs_ub_half, qs_global);
      }

      // Gate QK by exp(min(g_j - g_i, 0)) and the causal mask.
      TMUL(qk_ub, qk_ub, coeff_ub);
      TCVT(qk_ub_half, qk_ub, pto::RoundMode::CAST_NONE);

      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      {
        GmShape2D qk_gated_shape(HalfChunk, ChunkSize);
        GmStride2D qk_gated_stride(ChunkSize);
        GmTensor2D<half> qk_gated_global(
            workspace_qk_gated_handle +
                static_cast<int64_t>(cid) * WsGatedSize +
                static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
            qk_gated_shape, qk_gated_stride);
        TSTORE(qk_gated_global, qk_ub_half);
      }
      ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (1 << 8));

      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      TCVT(qs_ub, qs_ub_half, pto::RoundMode::CAST_NONE);
      TileUbDataND<float, HalfChunk, HiddenSize,
                   HalfChunk, HiddenSize> g_exp_2d;
      TASSIGN(g_exp_2d, CoeffUbAddr);
      TileUbDataDN<float, HalfChunk, 1,
                   HalfChunk, 1> g_v_col2;
      TASSIGN(g_v_col2, GvUbAddr);
      TROWEXPAND(g_exp_2d, g_v_col2);
      pipe_barrier(PIPE_V);
      // Reweight the recurrent QS path before it joins the QKV term.
      TMUL(qs_ub, qs_ub, g_exp_2d);

      wait_flag_dev(2);

      {
        GmShape2D qkv_shape(HalfChunk, HiddenSize);
        GmStride2D qkv_stride(HiddenSize);
        GmTensor2D<half> qkv_global(
            workspace_qs_qkv_handle +
                static_cast<int64_t>(cid) * WsQSSize +
                static_cast<int64_t>(vid) * HalfChunk * HiddenSize,
            qkv_shape, qkv_stride);
        TLOAD(o_ub_half, qkv_global);
      }

      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

      TCVT(o_ub, o_ub_half, pto::RoundMode::CAST_NONE);
      // Final O rows add the recurrent QS term onto the gated QKV output.
      TADD(o_ub, qs_ub, o_ub);
      TCVT(o_ub_half, o_ub, pto::RoundMode::CAST_NONE);

      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

      int64_t o_offset =
          (chunk_token_start * NumHeads + head_idx) *
              static_cast<int64_t>(HiddenSize) +
          static_cast<int64_t>(vid) * HalfChunk * NumHeads * HiddenSize;

      {
        GmShape2D o_shape(HalfChunk, HiddenSize);
        GmStride2D o_stride(NumHeads * HiddenSize);
        GmTensor2D<half> o_global(O_handle + o_offset, o_shape, o_stride);
        TSTORE(o_global, o_ub_half);
      }

      ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (3 << 8));
    }
  } else {
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

            {
              GmShape2D g_shape(1, valid_rows);
              GmStride2D g_stride(1);
              GmTensor2D<float> g_global(
                  G_handle + static_cast<int64_t>(head_idx) * total_tokens +
                      chunk_token_start,
                  g_shape, g_stride);
              DynVecTile<float, 1, ChunkSize, pto::PadValue::Zero> g_load(
                  1, valid_rows);
              TASSIGN(g_load, GUbAddr);
              TLOAD(g_load, g_global);
              if (valid_rows != ChunkSize) {
                TFILLPAD_INPLACE(g_ub, g_load);
              }
            }
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            TileUbDataND<float, 1, HalfChunk,
                         1, HalfChunk> g_ub_temp_v;
            TASSIGN(g_ub_temp_v,
                    GUbAddr +
                        static_cast<int32_t>(vid) * HalfChunk *
                            static_cast<int32_t>(sizeof(float)));
            TMOV(g_v_ub, g_ub_temp_v);

            TileUbDataND<float, HalfChunk, ChunkSize,
                         HalfChunk, ChunkSize> g_r_2d_v;
            TASSIGN(g_r_2d_v, QSUbAddr);
            TileUbDataDN<float, HalfChunk, 1,
                         HalfChunk, 1> g_v_col_v;
            TASSIGN(g_v_col_v, GvUbAddr);
            TROWEXPAND(g_r_2d_v, g_v_col_v);
            TCOLEXPAND(coeff_ub, g_ub);
            TSUB(coeff_ub, g_r_2d_v, coeff_ub);
            pipe_barrier(PIPE_V);
            TMINS(coeff_ub, coeff_ub, 0.0f);
            pipe_barrier(PIPE_V);
            TEXP(coeff_ub, coeff_ub);
            pipe_barrier(PIPE_V);
            TMUL(coeff_ub, coeff_ub, msk_ub);
            TEXP(g_v_ub, g_v_ub);

            wait_flag_dev(0);

            {
              GmShape2D qk_shape(HalfChunk, ChunkSize);
              GmStride2D qk_stride(ChunkSize);
              GmTensor2D<half> qk_global(
                  workspace_qk_handle +
                      static_cast<int64_t>(cid) * WsQKSize +
                      static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
                  qk_shape, qk_stride);
              TLOAD(qk_ub_half, qk_global);
            }

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            TCVT(qk_ub, qk_ub_half, pto::RoundMode::CAST_NONE);

            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

            {
              GmShape2D qs_shape(HalfChunk, HiddenSize);
              GmStride2D qs_stride(HiddenSize);
              GmTensor2D<half> qs_global(
                  workspace_qs_qkv_handle +
                      static_cast<int64_t>(cid) * WsQSSize +
                      static_cast<int64_t>(vid) * HalfChunk * HiddenSize,
                  qs_shape, qs_stride);
              TLOAD(qs_ub_half, qs_global);
            }

            TMUL(qk_ub, qk_ub, coeff_ub);
            TCVT(qk_ub_half, qk_ub, pto::RoundMode::CAST_NONE);

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            {
              GmShape2D qk_gated_shape(HalfChunk, ChunkSize);
              GmStride2D qk_gated_stride(ChunkSize);
              GmTensor2D<half> qk_gated_global(
                  workspace_qk_gated_handle +
                      static_cast<int64_t>(cid) * WsGatedSize +
                      static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
                  qk_gated_shape, qk_gated_stride);
              TSTORE(qk_gated_global, qk_ub_half);
            }
            ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (1 << 8));

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            TCVT(qs_ub, qs_ub_half, pto::RoundMode::CAST_NONE);

            TileUbDataND<float, HalfChunk, HiddenSize,
                         HalfChunk, HiddenSize> g_exp_2d_v;
            TASSIGN(g_exp_2d_v, CoeffUbAddr);
            TileUbDataDN<float, HalfChunk, 1,
                         HalfChunk, 1> g_v_col2_v;
            TASSIGN(g_v_col2_v, GvUbAddr);
            TROWEXPAND(g_exp_2d_v, g_v_col2_v);
            pipe_barrier(PIPE_V);
            TMUL(qs_ub, qs_ub, g_exp_2d_v);

            wait_flag_dev(2);

            {
              GmShape2D qkv_shape(HalfChunk, HiddenSize);
              GmStride2D qkv_stride(HiddenSize);
              GmTensor2D<half> qkv_global(
                  workspace_qs_qkv_handle +
                      static_cast<int64_t>(cid) * WsQSSize +
                      static_cast<int64_t>(vid) * HalfChunk * HiddenSize,
                  qkv_shape, qkv_stride);
              TLOAD(o_ub_half, qkv_global);
            }

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            TCVT(o_ub, o_ub_half, pto::RoundMode::CAST_NONE);
            TADD(o_ub, qs_ub, o_ub);
            TCVT(o_ub_half, o_ub, pto::RoundMode::CAST_NONE);

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

            int64_t o_offset =
                (chunk_token_start * NumHeads + head_idx) *
                    static_cast<int64_t>(HiddenSize) +
                static_cast<int64_t>(vid) * HalfChunk *
                    NumHeads * HiddenSize;

            {
              GmShape2D o_shape(HalfChunk, HiddenSize);
              GmStride2D o_stride(NumHeads * HiddenSize);
              GmTensor2D<half> o_global(O_handle + o_offset, o_shape,
                                        o_stride);
              TSTORE(o_global, o_ub_half);
            }

            ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (3 << 8));
          }
          gi++;
        }
      }
    }
  }
#endif
}

extern "C" __global__ AICORE void launch_chunk_o(
    __gm__ uint8_t *Q_handle, __gm__ uint8_t *K_handle,
    __gm__ uint8_t *V_handle, __gm__ uint8_t *S_handle,
    __gm__ uint8_t *G_handle, __gm__ uint8_t *Msk_handle,
    __gm__ uint8_t *workspace_qk, __gm__ uint8_t *workspace_qs_qkv,
    __gm__ uint8_t *workspace_qk_gated,
    __gm__ uint8_t *O_handle,
    __gm__ uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len, int64_t total_tokens,
    uint64_t ffts_addr)
{
  chunk_o_kernel<GDN_H, GDN_D, GDN_C>(
      reinterpret_cast<__gm__ half *>(Q_handle),
      reinterpret_cast<__gm__ half *>(K_handle),
      reinterpret_cast<__gm__ half *>(V_handle),
      reinterpret_cast<__gm__ half *>(S_handle),
      reinterpret_cast<__gm__ float *>(G_handle),
      reinterpret_cast<__gm__ float *>(Msk_handle),
      reinterpret_cast<__gm__ half *>(workspace_qk),
      reinterpret_cast<__gm__ half *>(workspace_qs_qkv),
      reinterpret_cast<__gm__ half *>(workspace_qk_gated),
      reinterpret_cast<__gm__ half *>(O_handle),
      reinterpret_cast<__gm__ int32_t *>(cu_seqlens),
      batch_size, seq_len, total_tokens, ffts_addr);
}

extern "C" void call_kernel(
    uint32_t block_dim, void *stream,
    uint8_t *q, uint8_t *k, uint8_t *v, uint8_t *s, uint8_t *g_sum,
    uint8_t *mask,
    uint8_t *workspace_qk, uint8_t *workspace_qs_qkv,
    uint8_t *workspace_qk_gated,
    uint8_t *o,
    uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len, int64_t total_tokens)
{
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  launch_chunk_o<<<block_dim, nullptr, stream>>>(
      q, k, v, s, g_sum, mask,
      workspace_qk, workspace_qs_qkv, workspace_qk_gated,
      o,
      cu_seqlens,
      batch_size, seq_len, total_tokens, fftsAddr);
}
