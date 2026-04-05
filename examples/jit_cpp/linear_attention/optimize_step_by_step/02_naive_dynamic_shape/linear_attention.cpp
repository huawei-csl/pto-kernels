#include <pto/pto-inst.hpp>
#include <runtime/rt_ffts.h>

using namespace pto;

#ifndef LINEAR_ATTN_H
#define LINEAR_ATTN_H 2
#endif

#ifndef LINEAR_ATTN_D
#define LINEAR_ATTN_D 128
#endif

#ifndef LINEAR_ATTN_C
#define LINEAR_ATTN_C 64
#endif

template <typename T, int Rows, int Cols>
using L1Mat = Tile<TileType::Mat, T, Rows, Cols, BLayout::ColMajor, Rows, Cols,
                   SLayout::RowMajor, 512, PadValue::Zero>;

template <typename T, int Rows, int Cols>
using L1MatTrans =
    Tile<TileType::Mat, T, Rows, Cols, BLayout::RowMajor, Rows, Cols,
         SLayout::ColMajor, 512, PadValue::Zero>;

template <typename T, int Rows, int Cols>
using UbVec = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, Rows, Cols,
                   SLayout::NoneBox, 512, PadValue::Null>;

template <pipe_t Pipe>
AICORE inline void SetCrossFlag(int32_t flag, int32_t mode) {
  const int config = 1 | (mode << 4) | (flag << 8);
  ffts_cross_core_sync(Pipe, config);
}

AICORE inline void WaitCrossFlag(int32_t flag) { wait_flag_dev(flag); }

template <int M, int N, int K, bool TransposeA = false, bool TransposeB = false>
AICORE inline void MatmulL1(
    TileAcc<float, M, N, M, N> &dst,
    std::conditional_t<TransposeA, L1Mat<half, K, M>, L1Mat<half, M, K>> &a_l1,
    std::conditional_t<TransposeB, L1Mat<half, N, K>, L1Mat<half, K, N>> &b_l1,
    bool init) {
  TileLeft<half, M, K, M, K> a_l0;
  TileRight<half, K, N, K, N> b_l0;
  TASSIGN(a_l0, 0x0);
  TASSIGN(b_l0, 0x0);

  if constexpr (TransposeA) {
    L1MatTrans<half, M, K> a_view;
    TRESHAPE(a_view, a_l1);
    TEXTRACT(a_l0, a_view, 0, 0);
  } else {
    TEXTRACT(a_l0, a_l1, 0, 0);
  }

  if constexpr (TransposeB) {
    L1MatTrans<half, K, N> b_view;
    TRESHAPE(b_view, b_l1);
    TEXTRACT(b_l0, b_view, 0, 0);
  } else {
    TEXTRACT(b_l0, b_l1, 0, 0);
  }

  pipe_barrier(PIPE_ALL);
  if (init) {
    TMATMUL(dst, a_l0, b_l0);
  } else {
    TMATMUL_ACC(dst, dst, a_l0, b_l0);
  }
  pipe_barrier(PIPE_ALL);
}

template <int NumHeads, int HiddenSize, int ChunkSize>
AICORE void main_kernel(__gm__ half *q, __gm__ half *k, __gm__ half *v,
                        __gm__ half *workspace_scores,
                        __gm__ half *workspace_state, __gm__ half *o,
                        int64_t batch_size, int64_t seq_len,
                        uint64_t ffts_addr) {
  constexpr int kVecParts = 2;
  constexpr int kHalfChunk = ChunkSize / kVecParts;
  constexpr int kHalfHidden = HiddenSize / kVecParts;
  constexpr int kChunkElems = ChunkSize * HiddenSize;
  constexpr int kScoreElems = ChunkSize * ChunkSize;
  constexpr int kStateElems = HiddenSize * HiddenSize;
  constexpr int kQueryL1Addr = 0;
  constexpr int kKeyL1Addr = kQueryL1Addr + kChunkElems * sizeof(half);
  constexpr int kValueL1Addr = kKeyL1Addr + kChunkElems * sizeof(half);
  constexpr int kStateL1Addr = kValueL1Addr + kChunkElems * sizeof(half);
  constexpr int kMaskedScoreL1Addr = kStateL1Addr + kStateElems * sizeof(half);
  constexpr int kScoreL0Addr = 0;
  constexpr int kStateL0Addr = kScoreL0Addr + kScoreElems * sizeof(float);
  constexpr int kOutputL0Addr = kStateL0Addr + kStateElems * sizeof(float);
  constexpr int kPrefixStateUbAddr = 0;
  constexpr int kScoreUbAddr = kPrefixStateUbAddr + kHalfHidden * HiddenSize * sizeof(half);
  constexpr int kStateUbAddr = kScoreUbAddr + kHalfChunk * ChunkSize * sizeof(half);
  constexpr int kZeroScoreUbAddr = kStateUbAddr + kHalfHidden * HiddenSize * sizeof(half);

  static_assert((HiddenSize % 2) == 0, "HiddenSize must be even.");
  static_assert((ChunkSize % 2) == 0, "ChunkSize must be even.");

  using ChunkGlobal =
      GlobalTensor<half, TileShape2D<half, ChunkSize, HiddenSize, Layout::ND>,
                   BaseShape2D<half, ChunkSize, HiddenSize, Layout::ND>,
                   Layout::ND>;
  using ScoreGlobal =
      GlobalTensor<half, TileShape2D<half, ChunkSize, ChunkSize, Layout::ND>,
                   BaseShape2D<half, ChunkSize, ChunkSize, Layout::ND>,
                   Layout::ND>;
  using StateGlobal =
      GlobalTensor<half, TileShape2D<half, HiddenSize, HiddenSize, Layout::ND>,
                   BaseShape2D<half, HiddenSize, HiddenSize, Layout::ND>,
                   Layout::ND>;
  using HalfScoreGlobal =
      GlobalTensor<half, TileShape2D<half, kHalfChunk, ChunkSize, Layout::ND>,
                   BaseShape2D<half, kHalfChunk, ChunkSize, Layout::ND>,
                   Layout::ND>;
  using HalfStateGlobal =
      GlobalTensor<half, TileShape2D<half, kHalfHidden, HiddenSize, Layout::ND>,
                   BaseShape2D<half, kHalfHidden, HiddenSize, Layout::ND>,
                   Layout::ND>;

  const int64_t total_work = batch_size * NumHeads;
  const int64_t chunk_count = seq_len / ChunkSize;
  const int64_t core_id = get_block_idx();
  const int64_t vector_id = get_subblockid();
  set_ffts_base_addr(ffts_addr);

  L1Mat<half, ChunkSize, HiddenSize> q_chunk_l1;
  L1Mat<half, ChunkSize, HiddenSize> k_chunk_l1;
  L1Mat<half, ChunkSize, HiddenSize> v_chunk_l1;
  L1Mat<half, HiddenSize, HiddenSize> prefix_state_l1;
  L1Mat<half, ChunkSize, ChunkSize> masked_score_l1;
  TASSIGN(q_chunk_l1, kQueryL1Addr);
  TASSIGN(k_chunk_l1, kKeyL1Addr);
  TASSIGN(v_chunk_l1, kValueL1Addr);
  TASSIGN(prefix_state_l1, kStateL1Addr);
  TASSIGN(masked_score_l1, kMaskedScoreL1Addr);

  TileAcc<float, ChunkSize, ChunkSize, ChunkSize, ChunkSize> raw_score_l0;
  TileAcc<float, HiddenSize, HiddenSize, HiddenSize, HiddenSize> state_update_l0;
  TileAcc<float, ChunkSize, HiddenSize, ChunkSize, HiddenSize> output_l0;
  TASSIGN(raw_score_l0, kScoreL0Addr);
  TASSIGN(state_update_l0, kStateL0Addr);
  TASSIGN(output_l0, kOutputL0Addr);

  UbVec<half, kHalfHidden, HiddenSize> running_state_ub;
  UbVec<half, kHalfHidden, HiddenSize> state_delta_ub;
  UbVec<half, kHalfChunk, ChunkSize> score_ub;
  UbVec<half, kHalfChunk, ChunkSize> zero_score_ub;
  TASSIGN(running_state_ub, kPrefixStateUbAddr);
  TASSIGN(score_ub, kScoreUbAddr);
  TASSIGN(state_delta_ub, kStateUbAddr);
  TASSIGN(zero_score_ub, kZeroScoreUbAddr);

#if defined(__DAV_C220_CUBE__)
  for (int64_t work_idx = 0; work_idx < (total_work + block_num - 1) / block_num;
       ++work_idx) {
    const int64_t work_id = work_idx * block_num + core_id;
    if (work_id >= total_work) {
      continue;
    }

    const int64_t head_id = work_id % NumHeads;
    const int64_t batch_id = work_id / NumHeads;
    const int64_t qkv_base = ((batch_id * NumHeads + head_id) * seq_len) * HiddenSize;
    const int64_t score_workspace_base = core_id * kScoreElems;
    const int64_t state_workspace_base = core_id * kStateElems;

    WaitCrossFlag(1);
    for (int64_t chunk_index = 0; chunk_index < chunk_count; ++chunk_index) {
      const int64_t chunk_base = qkv_base + chunk_index * kChunkElems;
      ChunkGlobal q_global(q + chunk_base);
      ChunkGlobal k_global(k + chunk_base);
      ChunkGlobal v_global(v + chunk_base);
      StateGlobal prefix_state_global(workspace_state + state_workspace_base);
      ScoreGlobal score_global(workspace_scores + score_workspace_base);
      ChunkGlobal output_global(o + chunk_base);

      // In the dynamic kernel the launch shape stays fixed; this loop maps
      // many logical (batch, head) jobs onto the available cores.
      TLOAD(q_chunk_l1, q_global);
      TLOAD(k_chunk_l1, k_global);
      TLOAD(v_chunk_l1, v_global);
      TLOAD(prefix_state_l1, prefix_state_global);
      pipe_barrier(PIPE_ALL);

      MatmulL1<ChunkSize, ChunkSize, HiddenSize, false, true>(raw_score_l0,
                                                              q_chunk_l1,
                                                              k_chunk_l1, true);
      TSTORE(score_global, raw_score_l0);
      pipe_barrier(PIPE_ALL);

      MatmulL1<HiddenSize, HiddenSize, ChunkSize, true, false>(state_update_l0,
                                                               k_chunk_l1,
                                                               v_chunk_l1, true);
      TSTORE(prefix_state_global, state_update_l0);
      pipe_barrier(PIPE_ALL);
      SetCrossFlag<PIPE_FIX>(0, 2);

      WaitCrossFlag(1);
      TLOAD(masked_score_l1, score_global);
      pipe_barrier(PIPE_ALL);

      MatmulL1<ChunkSize, HiddenSize, ChunkSize, false, false>(output_l0,
                                                               masked_score_l1,
                                                               v_chunk_l1, true);
      MatmulL1<ChunkSize, HiddenSize, HiddenSize, false, false>(output_l0,
                                                                q_chunk_l1,
                                                                prefix_state_l1,
                                                                false);
      TSTORE(output_global, output_l0);
      pipe_barrier(PIPE_ALL);
    }
  }
#endif

#if defined(__DAV_C220_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  for (int64_t work_idx = 0; work_idx < (total_work + block_num - 1) / block_num;
       ++work_idx) {
    const int64_t work_id = work_idx * block_num + core_id;
    if (work_id >= total_work) {
      continue;
    }

    const int64_t score_workspace_base =
        core_id * kScoreElems + vector_id * kHalfChunk * ChunkSize;
    const int64_t state_workspace_base =
        core_id * kStateElems + vector_id * kHalfHidden * HiddenSize;

    TEXPANDS(running_state_ub, 0.0f);
    TEXPANDS(zero_score_ub, 0.0f);
    pipe_barrier(PIPE_ALL);
    HalfStateGlobal state_slice_global(workspace_state + state_workspace_base);
    TSTORE(state_slice_global, running_state_ub);
    pipe_barrier(PIPE_ALL);
    SetCrossFlag<PIPE_MTE3>(1, 2);

    for (int64_t chunk_index = 0; chunk_index < chunk_count; ++chunk_index) {
      WaitCrossFlag(0);
      HalfScoreGlobal score_slice_global(workspace_scores + score_workspace_base);
      TLOAD(score_ub, score_slice_global);
      TLOAD(state_delta_ub, state_slice_global);
      pipe_barrier(PIPE_ALL);

      for (int row = 0; row < kHalfChunk; ++row) {
        for (int col = 0; col < ChunkSize; ++col) {
          if (vector_id * kHalfChunk + row < col) {
            score_ub.SetValue(row * ChunkSize + col,
                              zero_score_ub.GetValue(row * ChunkSize + col));
          }
        }
      }
      pipe_barrier(PIPE_ALL);

      TADD(running_state_ub, running_state_ub, state_delta_ub);
      pipe_barrier(PIPE_ALL);
      TSTORE(score_slice_global, score_ub);
      TSTORE(state_slice_global, running_state_ub);
      pipe_barrier(PIPE_ALL);
      SetCrossFlag<PIPE_MTE3>(1, 2);
    }
  }
#endif
}

extern "C" __global__ AICORE void launch_linear_attention(
    __gm__ uint8_t *q, __gm__ uint8_t *k, __gm__ uint8_t *v,
    __gm__ uint8_t *workspace_scores, __gm__ uint8_t *workspace_state,
    __gm__ uint8_t *o, int64_t batch_size, int64_t seq_len,
    uint64_t ffts_addr) {
  main_kernel<LINEAR_ATTN_H, LINEAR_ATTN_D, LINEAR_ATTN_C>(
      reinterpret_cast<__gm__ half *>(q), reinterpret_cast<__gm__ half *>(k),
      reinterpret_cast<__gm__ half *>(v),
      reinterpret_cast<__gm__ half *>(workspace_scores),
      reinterpret_cast<__gm__ half *>(workspace_state),
      reinterpret_cast<__gm__ half *>(o), batch_size, seq_len, ffts_addr);
}

extern "C" void call_kernel(uint32_t block_dim, void *stream, uint8_t *q,
                            uint8_t *k, uint8_t *v, uint8_t *workspace_scores,
                            uint8_t *workspace_state, uint8_t *o,
                            int64_t batch_size, int64_t seq_len) {
  uint32_t ffts_len = 0;
  uint64_t ffts_addr = 0;
  rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
  launch_linear_attention<<<block_dim, nullptr, stream>>>(
      q, k, v, workspace_scores, workspace_state, o, batch_size, seq_len,
      ffts_addr);
}
