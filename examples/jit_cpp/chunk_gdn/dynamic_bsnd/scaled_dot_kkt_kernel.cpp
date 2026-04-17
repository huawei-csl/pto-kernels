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
                             pto::SLayout::RowMajor, 512, pto::PadValue::Zero>;

template <typename T, int Rows, int Cols, int RowValid = Rows,
          int ColValid = Cols>
using TileMatL0B = pto::Tile<pto::TileType::Right, T, Rows, Cols,
                             pto::BLayout::RowMajor, RowValid, ColValid,
                             pto::SLayout::ColMajor, 512, pto::PadValue::Zero>;

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

template <typename T, int32_t Rows, int32_t Cols>
using DynMatL1 = TileMatL1<T, Rows, Cols, pto::DYNAMIC, pto::DYNAMIC>;

template <typename T, int32_t Rows, int32_t Cols,
          pto::PadValue PadVal = pto::PadValue::Null>
using DynVecTile =
    TileUbDataND<T, Rows, Cols, pto::DYNAMIC, pto::DYNAMIC, PadVal>;

template <typename T, int32_t Rows, int32_t Cols>
using DynAccTile = pto::TileAcc<T, Rows, Cols, pto::DYNAMIC, pto::DYNAMIC>;

// PTO cheat sheet for the bigger matrix kernels:
//   - `GlobalTensor<T>` is a GM view, similar to a tensor plus explicit strides.
//   - `Tile<..., Mat, ...>` lives in L1 and feeds Cube matmul instructions.
//   - `Tile<..., Vec, ...>` lives in UB and feeds vector instructions.
//   - `TileAcc<T, ...>` is the Cube accumulator tile written by `TMATMUL`.
//   - `TLOAD` / `TSTORE` move data between GM and on-chip memory.
//   - `TEXTRACT` slices an L1 matrix tile into an L0 tile for Cube.
//   - `TRESHAPE` changes metadata/layout view without changing the bytes.
//   - `TROWEXPAND` / `TCOLEXPAND` act like broadcast/repeat along one axis.

template <typename T1, typename T2, uint32_t M, uint32_t N, uint32_t K,
          uint32_t validM = M, uint32_t validN = N, uint32_t validK = K,
          uint32_t KTail, bool TransposeA = false, bool TransposeB = false>
AICORE PTO_INLINE void gemm_v0(
    std::conditional_t<TransposeA, TileMatL1<T1, K, M, validK, validM>,
                       TileMatL1<T1, M, K, validM, validK>> &a,
    std::conditional_t<TransposeB, TileMatL1<T1, N, K, validN, validK>,
                       TileMatL1<T1, K, N, validK, validN>> &b,
    pto::TileAcc<T2, M, N, validM, validN> &c, bool clear)
{
  // Local K-sliced matmul helper:
  //   C = A @ B  (or accumulate into C)
  // The Cube core only sees one K-slice at a time, so PTO explicitly extracts
  // L0 tiles from L1 and accumulates the partial products.
  //
  // PyTorch mental model:
  //   C = 0
  //   for k0 in range(0, K, KL0Size):
  //       C += A[:, k0:k1] @ B[k0:k1, :]
  //
  // PTO version of the same loop:
  //   L1 tile -> `TEXTRACT` -> L0A/L0B -> `TMATMUL` / `TMATMUL_ACC`
  constexpr uint32_t KL0Size = 128;
  const uint32_t k_l0_split = (K + KL0Size - 1) / KL0Size;

  auto war_event_id = (event_t)(((int)EVENT_ID0 + 1) % 8);
  set_flag(PIPE_MTE2, PIPE_MTE1, war_event_id);
  wait_flag(PIPE_MTE2, PIPE_MTE1, war_event_id);

  for (uint32_t k_l0_idx = 0; k_l0_idx < k_l0_split; ++k_l0_idx) {
    const bool init_flag = clear && (k_l0_idx == 0);
    const bool is_tail_block = (k_l0_idx == k_l0_split - 1);

    if (is_tail_block) {
      TileMatL0A<T1, M, KTail, M, KTail> l0a;
      TileMatL0B<T1, KTail, N, KTail, N> l0b;
      pto::TASSIGN(l0a, 0x0);
      pto::TASSIGN(l0b, 0x0);

      set_flag(PIPE_M, PIPE_MTE1, war_event_id);
      wait_flag(PIPE_M, PIPE_MTE1, war_event_id);

      if constexpr (!TransposeA) {
        pto::TEXTRACT(l0a, a, 0, k_l0_idx * KTail);
      } else {
        TileMatL1ZN<T1, M, K, validM, validK> a_t;
        pto::TRESHAPE(a_t, a);
        pto::TEXTRACT(l0a, a_t, 0, k_l0_idx * KTail);
      }

      if constexpr (!TransposeB) {
        pto::TEXTRACT(l0b, b, k_l0_idx * KTail, 0);
      } else {
        TileMatL1ZN<T1, K, N, validK, validN> b_t;
        pto::TRESHAPE(b_t, b);
        pto::TEXTRACT(l0b, b_t, k_l0_idx * KTail, 0);
      }

      set_flag(PIPE_MTE1, PIPE_M, war_event_id);
      wait_flag(PIPE_MTE1, PIPE_M, war_event_id);

      if (init_flag) {
        pto::TMATMUL(c, l0a, l0b);
      } else {
        pto::TMATMUL_ACC(c, c, l0a, l0b);
      }
    } else {
      TileMatL0A<T1, M, KL0Size, M, KL0Size> l0a;
      TileMatL0B<T1, KL0Size, N, KL0Size, N> l0b;
      pto::TASSIGN(l0a, 0x0);
      pto::TASSIGN(l0b, 0x0);

      set_flag(PIPE_M, PIPE_MTE1, war_event_id);
      wait_flag(PIPE_M, PIPE_MTE1, war_event_id);

      set_flag(PIPE_FIX, PIPE_M, war_event_id);
      wait_flag(PIPE_FIX, PIPE_M, war_event_id);

      if constexpr (!TransposeA) {
        pto::TEXTRACT(l0a, a, 0, k_l0_idx * KL0Size);
      } else {
        TileMatL1ZN<T1, M, K, validM, validK> a_t;
        pto::TRESHAPE(a_t, a);
        pto::TEXTRACT(l0a, a_t, 0, k_l0_idx * KL0Size);
      }

      if constexpr (!TransposeB) {
        pto::TEXTRACT(l0b, b, k_l0_idx * KL0Size, 0);
      } else {
        TileMatL1ZN<T1, K, N, validK, validN> b_t;
        pto::TRESHAPE(b_t, b);
        pto::TEXTRACT(l0b, b_t, k_l0_idx * KL0Size, 0);
      }

      set_flag(PIPE_MTE1, PIPE_M, war_event_id);
      wait_flag(PIPE_MTE1, PIPE_M, war_event_id);

      if (init_flag) {
        pto::TMATMUL(c, l0a, l0b);
      } else {
        pto::TMATMUL_ACC(c, c, l0a, l0b);
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
AICORE void kkt_kernel(
    __gm__ half *K_handle, __gm__ half *Beta_handle,
    __gm__ float *G_handle, __gm__ float *Msk_handle,
    __gm__ half *workspace_handle, __gm__ half *A_handle,
    __gm__ int32_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len, int64_t total_tokens,
    uint64_t ffts_addr)
{
  // This kernel has two phases:
  // 1. Cube builds the raw local score matrix A_raw = K @ K^T for each chunk.
  // 2. Vec applies the dynamic gate and causal mask:
  //      A[i, j] = A_raw[i, j] * mask[i, j]
  //                * exp(min(g_i + log(beta_i) - g_j, 0)).
  //
  // Shapes for one (sequence, head, chunk):
  //   K_chunk    : [valid, D]
  //   g_chunk    : [valid]
  //   beta_chunk : [valid]
  //   A_chunk    : [valid, valid]
  //
  // PyTorch / NumPy sketch:
  //   A_raw = K_chunk @ K_chunk.T
  //   row_term = g_chunk[:, None] + log(beta_chunk)[:, None]
  //   col_term = g_chunk[None, :]
  //   gate = exp(min(row_term - col_term, 0))
  //   A = tril(A_raw * gate, diagonal=-1)
  //
  // PTO schedule:
  //   Cube: GM -> L1 -> L0 -> matmul -> workspace
  //   Vec : workspace -> UB -> apply gate/mask -> final GM tensor
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
  constexpr int32_t GBlockUbAddr = AUbAddr;
  constexpr int32_t BetaBlockUbAddr = CoeffUbAddr;

  set_ffts_base_addr(ffts_addr);
  auto cid = get_block_idx();
  auto block_num = get_block_num();
  auto vid = get_subblockid();

  int64_t num_seqs = batch_size;
  int64_t total_work = num_seqs * NumHeads;

  TileMatL1<half, ChunkSize, HiddenSize, ChunkSize, HiddenSize> k_l1;
  TASSIGN(k_l1, 0);
  TileAcc<float, ChunkSize, ChunkSize, ChunkSize, ChunkSize> a_l0;
  TASSIGN(a_l0, 0);

  TileUbDataND<float, 1, ChunkSize, 1, ChunkSize, pto::PadValue::Zero> g_ub;
  TASSIGN(g_ub, GUbAddr);
  TileUbDataND<half, 1, HalfChunk, 1, HalfChunk, pto::PadValue::Zero>
      beta_ub_half;
  TASSIGN(beta_ub_half, BetaHalfUbAddr);
  TileUbDataND<float, 1, HalfChunk, 1, HalfChunk> beta_ub;
  TASSIGN(beta_ub, BetaUbAddr);
  TileUbDataND<float, 1, HalfChunk, 1, HalfChunk> g_v_ub;
  TASSIGN(g_v_ub, GvUbAddr);
  TileUbDataND<float, HalfChunk, ChunkSize, HalfChunk, ChunkSize> a_ub;
  TASSIGN(a_ub, AUbAddr);
  TileUbDataND<float, 1, HalfChunk, 1, HalfChunk> g_r_ub;
  TASSIGN(g_r_ub, GRUbAddr);
  TileUbDataND<float, 1, ChunkSize, 1, ChunkSize> g_c_ub;
  TASSIGN(g_c_ub, GCUbAddr);
  TileUbDataND<float, HalfChunk, ChunkSize, HalfChunk, ChunkSize> msk_ub;
  TASSIGN(msk_ub, MskUbAddr);
  TileUbDataND<float, HalfChunk, ChunkSize, HalfChunk, ChunkSize> g_r_2d_ub;
  TASSIGN(g_r_2d_ub, GR2dUbAddr);
  TileUbDataND<float, HalfChunk, ChunkSize, HalfChunk, ChunkSize> g_c_2d_ub;
  TASSIGN(g_c_2d_ub, GC2dUbAddr);
  TileUbDataND<float, HalfChunk, ChunkSize, HalfChunk, ChunkSize> coeff_ub;
  TASSIGN(coeff_ub, CoeffUbAddr);
  TileUbDataND<half, HalfChunk, ChunkSize, HalfChunk, ChunkSize> a_ub_half;
  TASSIGN(a_ub_half, AUbHalfAddr);

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
      // Ping-pong two per-core workspace slots so Cube can produce chunk i+1
      // while Vec is still consuming chunk i.
      int32_t slot = static_cast<int32_t>(ci & 1);
      wait_flag_dev(2 + slot);
      pipe_barrier(PIPE_ALL);

      int64_t chunk_start = ci * ChunkSize;
      int64_t remaining = slen - chunk_start;
      int32_t valid_rows = static_cast<int32_t>(
          remaining < ChunkSize ? remaining : ChunkSize);

      int64_t k_offset =
          ((bos + chunk_start) * NumHeads + head_idx) *
          static_cast<int64_t>(HiddenSize);

      {
        GmShape2D k_shape(valid_rows, HiddenSize);
        GmStride2D k_stride(NumHeads * HiddenSize);
        GmTensor2D<half> k_global(K_handle + k_offset, k_shape, k_stride);
        DynMatL1<half, ChunkSize, HiddenSize> k_l1_load(valid_rows, HiddenSize);
        TASSIGN(k_l1_load, 0);
        TLOAD(k_l1_load, k_global);
        if (valid_rows != ChunkSize) {
          TFILLPAD(k_l1_load, k_l1_load);
        }
      }

      // Compute the dense intra-chunk score matrix A = K * K^T.
      gemm_v0<half, float,
              ChunkSize, ChunkSize, HiddenSize,
              ChunkSize, ChunkSize, HiddenSize,
              KTail, false, true>(k_l1, k_l1, a_l0, true);

      {
        GmShape2D a_shape(ChunkSize, ChunkSize);
        GmStride2D a_stride(ChunkSize);
        GmTensor2D<half> workspace_global(
            workspace_handle +
                (static_cast<int64_t>(cid) * 2 + slot) * ChunkSquare,
            a_shape, a_stride);
        DynAccTile<float, ChunkSize, ChunkSize> a_store(ChunkSize, ChunkSize);
        TASSIGN(a_store, 0);
        // Save A_raw into a per-core ping-pong workspace slot.
        // Later the Vec phase reads the same slot and applies the nonlinear gate.
        TSTORE(workspace_global, a_store);
      }

      ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (slot << 8));
    }
  }
#endif

#if defined(__DAV_C220_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);

  // Vec owns the lower-triangular mask and converts the raw Cube scores into
  // gated attention coefficients one HalfChunk-row stripe at a time.
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
        // Each Vec sub-block owns HalfChunk consecutive output rows i. It loads
        // the row term g_i + log(beta_i) plus the full column term g_j.
        {
          GmShape2D g_shape(1, valid_rows);
          GmStride2D g_stride(1);
          GmTensor2D<float> g_global(
              G_handle + static_cast<int64_t>(head_idx) * total_tokens +
                  (bos + chunk_start),
              g_shape, g_stride);
          DynVecTile<float, 1, ChunkSize, pto::PadValue::Zero> g_load(
              1, valid_rows);
          TASSIGN(g_load, GUbAddr);
          TLOAD(g_load, g_global);
          if (valid_rows != ChunkSize) {
            TFILLPAD_INPLACE(g_ub, g_load);
          }
        }

        {
          GmShape2D beta_shape(1, local_valid);
          GmStride2D beta_stride(1);
          GmTensor2D<half> beta_global(
              Beta_handle + static_cast<int64_t>(head_idx) * total_tokens +
                  (bos + chunk_start + row_offset),
              beta_shape, beta_stride);
          DynVecTile<half, 1, HalfChunk, pto::PadValue::Zero> beta_load(
              1, local_valid);
          TASSIGN(beta_load, BetaHalfUbAddr);
          TLOAD(beta_load, beta_global);
          if (local_valid != HalfChunk) {
            TFILLPAD_INPLACE(beta_ub_half, beta_load);
          }
        }

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        TCVT(beta_ub, beta_ub_half, pto::RoundMode::CAST_NONE);
        TileUbDataND<float, 1, HalfChunk, 1, HalfChunk> g_ub_temp;
        TASSIGN(g_ub_temp,
                GUbAddr + row_offset *
                              static_cast<int32_t>(sizeof(float)));
        TMOV(g_v_ub, g_ub_temp);
        pipe_barrier(PIPE_V);

        // Torch-like:
        //   beta_ub = beta[row_slice].float()
        //   g_v_ub  = g[row_slice]
        //   g_v_ub += log(beta_ub)
        TLOG(beta_ub, beta_ub);
        pipe_barrier(PIPE_V);
        TADD(g_v_ub, g_v_ub, beta_ub);
        pipe_barrier(PIPE_V);
        TMOV(g_r_ub, g_v_ub);
        TMOV(g_c_ub, g_ub);
        pipe_barrier(PIPE_V);

        // Broadcast the row and column terms to a 2-D coefficient tile so PTO
        // matches the scalar formula exp(min(g_i + log(beta_i) - g_j, 0)).
        TileUbDataDN<float, HalfChunk, 1, HalfChunk, 1> g_r_ub_temp;
        TASSIGN(g_r_ub_temp, GRUbAddr);
        TROWEXPAND(g_r_2d_ub, g_r_ub_temp);
        TCOLEXPAND(g_c_2d_ub, g_c_ub);
        pipe_barrier(PIPE_V);
        TSUB(coeff_ub, g_r_2d_ub, g_c_2d_ub);
        pipe_barrier(PIPE_V);
        TMINS(coeff_ub, coeff_ub, 0.0f);
        pipe_barrier(PIPE_V);
        TEXP(coeff_ub, coeff_ub);
      }

      wait_flag_dev(slot);
      pipe_barrier(PIPE_ALL);

      if (local_valid > 0) {
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

        // Consume the raw A_raw rows produced earlier by the Cube stage.
        {
          GmShape2D a_shape(HalfChunk, ChunkSize);
          GmStride2D a_stride(ChunkSize);
          GmTensor2D<half> a_workspace_global(
              workspace_handle +
                  (static_cast<int64_t>(cid) * 2 + slot) * ChunkSquare +
                  static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
              a_shape, a_stride);
          TLOAD(a_ub_half, a_workspace_global);
        }

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        // Apply the gate exp(g_i + log(beta_j) - g_j) and causal mask to A.
        // Torch-like:
        //   a_rows = a_raw_rows.float()
        //   a_rows *= coeff_ub
        //   a_rows *= msk_ub
        TCVT(a_ub, a_ub_half, pto::RoundMode::CAST_NONE);
        TMUL(a_ub, a_ub, coeff_ub);
        TMUL(a_ub, a_ub, msk_ub);
        TCVT(a_ub_half, a_ub, pto::RoundMode::CAST_NONE);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        int64_t a_gm_offset =
            ((bos + chunk_start + row_offset) * NumHeads +
             head_idx) *
            static_cast<int64_t>(ChunkSize);

        // Write this worker's final gated A rows back to the public BSND tensor.
        {
          GmShape2D a_out_shape(local_valid, ChunkSize);
          GmStride2D a_out_stride(NumHeads * ChunkSize);
          GmTensor2D<half> a_out_global(A_handle + a_gm_offset, a_out_shape,
                                        a_out_stride);
          DynVecTile<half, HalfChunk, ChunkSize, pto::PadValue::Zero>
              a_out_tile(local_valid, ChunkSize);
          TASSIGN(a_out_tile, AUbHalfAddr);
          TSTORE(a_out_global, a_out_tile);
        }
      }

      pipe_barrier(PIPE_ALL);
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
    int64_t batch_size, int64_t seq_len, int64_t total_tokens,
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
    int64_t batch_size, int64_t seq_len, int64_t total_tokens)
{
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  launch_scaled_dot_kkt<<<block_dim, nullptr, stream>>>(
      K_handle, Beta_handle, G_handle, Msk_handle,
      workspace_handle, A_handle, cu_seqlens,
      batch_size, seq_len, total_tokens, fftsAddr);
}
