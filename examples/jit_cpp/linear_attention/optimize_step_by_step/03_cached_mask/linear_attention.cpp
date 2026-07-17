#include <pto/pto-inst.hpp>
#include <pto/common/pto_tile.hpp>
#include <runtime/rt_ffts.h>
#include <type_traits>

using namespace pto;

// Step 03 keeps the naive overall schedule from step 02, but replaces the
// scalar triangular-mask loop with a precomputed mask tensor from PyTorch.
// That lets the vector core apply causality with one tile-wise multiply.

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
  // For these early steps we use a single, easy-to-follow "load to L0 then
  // matmul" helper. Later steps optimize the internals of this helper.
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

template <int32_t NumHeads, int32_t HiddenSize, int32_t ChunkSize>
AICORE void main_kernel(__gm__ half *q, __gm__ half *k, __gm__ half *v,
                        __gm__ half *workspace_1, __gm__ half *workspace_2,
                        __gm__ half *causal_mask, __gm__ half *o,
                        int64_t batch_size, int64_t seq_len,
                        uint64_t ffts_addr) {
  constexpr int32_t VecNum = 2;
  constexpr int32_t HalfChunk = ChunkSize / VecNum;
  constexpr int32_t HalfHidden = HiddenSize / VecNum;
  constexpr int32_t ChunkElems = ChunkSize * HiddenSize;
  constexpr int32_t Workspace1Elems = ChunkSize * ChunkSize;
  constexpr int32_t Workspace2Elems = HiddenSize * HiddenSize;

  constexpr int32_t QL1Addr = 0;
  constexpr int32_t KL1Addr = QL1Addr + ChunkElems * sizeof(half);
  constexpr int32_t VL1Addr = KL1Addr + ChunkElems * sizeof(half);
  constexpr int32_t HL1Addr = VL1Addr + ChunkElems * sizeof(half);
  constexpr int32_t AccL1Addr = HL1Addr + Workspace2Elems * sizeof(half);

  constexpr int32_t AccL0Addr = 0;
  constexpr int32_t HL0Addr = AccL0Addr + Workspace1Elems * sizeof(float);
  constexpr int32_t OL0Addr = HL0Addr + Workspace2Elems * sizeof(float);

  constexpr int32_t HsumUbAddr = 0;
  constexpr int32_t AccUbAddr = HsumUbAddr + HalfHidden * HiddenSize * sizeof(half);
  constexpr int32_t HUbAddr = AccUbAddr + HalfChunk * ChunkSize * sizeof(half);
  constexpr int32_t MaskUbAddr = HUbAddr + HalfHidden * HiddenSize * sizeof(half);
  constexpr int32_t MaskedAccUbAddr =
      MaskUbAddr + HalfChunk * ChunkSize * sizeof(half);

  constexpr int32_t L0CBytes =
      (Workspace1Elems + Workspace2Elems + ChunkElems) * sizeof(float);
  static_assert((HiddenSize % 2) == 0, "HiddenSize must be even.");
  static_assert((ChunkSize % 2) == 0, "ChunkSize must be even.");
  static_assert(L0CBytes <= 112 * 1024,
                "Tile sizes exceed the validated L0C budget for this minimum kernel.");

  using ChunkGlobal =
      GlobalTensor<half, TileShape2D<half, ChunkSize, HiddenSize, Layout::ND>,
                   BaseShape2D<half, ChunkSize, HiddenSize, Layout::ND>,
                   Layout::ND>;
  using AccGlobal =
      GlobalTensor<half, TileShape2D<half, ChunkSize, ChunkSize, Layout::ND>,
                   BaseShape2D<half, ChunkSize, ChunkSize, Layout::ND>,
                   Layout::ND>;
  using HiddenGlobal =
      GlobalTensor<half, TileShape2D<half, HiddenSize, HiddenSize, Layout::ND>,
                   BaseShape2D<half, HiddenSize, HiddenSize, Layout::ND>,
                   Layout::ND>;
  using HalfAccGlobal =
      GlobalTensor<half, TileShape2D<half, HalfChunk, ChunkSize, Layout::ND>,
                   BaseShape2D<half, HalfChunk, ChunkSize, Layout::ND>,
                   Layout::ND>;
  using HalfHiddenGlobal =
      GlobalTensor<half, TileShape2D<half, HalfHidden, HiddenSize, Layout::ND>,
                   BaseShape2D<half, HalfHidden, HiddenSize, Layout::ND>,
                   Layout::ND>;
  using HalfMaskGlobal =
      GlobalTensor<half, TileShape2D<half, HalfChunk, ChunkSize, Layout::ND>,
                   BaseShape2D<half, HalfChunk, ChunkSize, Layout::ND>,
                   Layout::ND>;

  const int64_t total_work = batch_size * NumHeads;
  const int64_t chunk_num = seq_len / ChunkSize;
  const int64_t cid = get_block_idx();
  const int64_t vid = get_subblockid();
  set_ffts_base_addr(ffts_addr);

  L1Mat<half, ChunkSize, HiddenSize> q_l1;
  L1Mat<half, ChunkSize, HiddenSize> k_l1;
  L1Mat<half, ChunkSize, HiddenSize> v_l1;
  L1Mat<half, HiddenSize, HiddenSize> h_l1;
  L1Mat<half, ChunkSize, ChunkSize> acc_l1;
  TASSIGN(q_l1, QL1Addr);
  TASSIGN(k_l1, KL1Addr);
  TASSIGN(v_l1, VL1Addr);
  TASSIGN(h_l1, HL1Addr);
  TASSIGN(acc_l1, AccL1Addr);

  TileAcc<float, ChunkSize, ChunkSize, ChunkSize, ChunkSize> acc_l0;
  TileAcc<float, HiddenSize, HiddenSize, HiddenSize, HiddenSize> h_l0;
  TileAcc<float, ChunkSize, HiddenSize, ChunkSize, HiddenSize> o_l0;
  TASSIGN(acc_l0, AccL0Addr);
  TASSIGN(h_l0, HL0Addr);
  TASSIGN(o_l0, OL0Addr);

  UbVec<half, HalfHidden, HiddenSize> hsum_ub;
  UbVec<half, HalfHidden, HiddenSize> h_ub;
  UbVec<half, HalfChunk, ChunkSize> acc_ub;
  UbVec<half, HalfChunk, ChunkSize> mask_ub;
  UbVec<half, HalfChunk, ChunkSize> masked_acc_ub;
  TASSIGN(hsum_ub, HsumUbAddr);
  TASSIGN(acc_ub, AccUbAddr);
  TASSIGN(h_ub, HUbAddr);
  TASSIGN(mask_ub, MaskUbAddr);
  TASSIGN(masked_acc_ub, MaskedAccUbAddr);

#if defined(__DAV_C220_CUBE__)
  for (int64_t work_idx = 0; work_idx < (total_work + block_num - 1) / block_num;
       ++work_idx) {
    const int64_t pid = work_idx * block_num + cid;
    if (pid >= total_work) {
      continue;
    }

    const int64_t by = pid % NumHeads;
    const int64_t bz = pid / NumHeads;
    const int64_t qkv_base = ((bz * NumHeads + by) * seq_len) * HiddenSize;
    const int64_t workspace1_base = cid * Workspace1Elems;
    const int64_t workspace2_base = cid * Workspace2Elems;

    WaitCrossFlag(1);

    for (int64_t i = 0; i < chunk_num; ++i) {
      const int64_t chunk_base = qkv_base + i * ChunkElems;

      ChunkGlobal q_global(q + chunk_base);
      ChunkGlobal k_global(k + chunk_base);
      ChunkGlobal v_global(v + chunk_base);
      HiddenGlobal h_global(workspace_2 + workspace2_base);

      TLOAD(q_l1, q_global);
      TLOAD(k_l1, k_global);
      TLOAD(v_l1, v_global);
      TLOAD(h_l1, h_global);
      pipe_barrier(PIPE_ALL);

      // Cube computes two intermediates for this chunk:
      // 1) chunk-local scores Q K^T
      // 2) hidden-state update K^T V
      MatmulL1<ChunkSize, ChunkSize, HiddenSize, false, true>(acc_l0, q_l1, k_l1,
                                                              true);
      AccGlobal acc_global(workspace_1 + workspace1_base);
      TSTORE(acc_global, acc_l0);
      pipe_barrier(PIPE_ALL);

      MatmulL1<HiddenSize, HiddenSize, ChunkSize, true, false>(h_l0, k_l1, v_l1,
                                                               true);
      HiddenGlobal h_out_global(workspace_2 + workspace2_base);
      TSTORE(h_out_global, h_l0);
      pipe_barrier(PIPE_ALL);
      SetCrossFlag<PIPE_FIX>(0, 2);

      // Vector core overwrites workspace_1 with the masked scores, then cube
      // finishes O = masked_scores @ V + Q @ prefix_state.
      WaitCrossFlag(1);
      AccGlobal masked_acc_global(workspace_1 + workspace1_base);
      TLOAD(acc_l1, masked_acc_global);
      pipe_barrier(PIPE_ALL);

      MatmulL1<ChunkSize, HiddenSize, ChunkSize, false, false>(o_l0, acc_l1,
                                                               v_l1, true);
      MatmulL1<ChunkSize, HiddenSize, HiddenSize, false, false>(o_l0, q_l1,
                                                                h_l1, false);

      ChunkGlobal o_global(o + chunk_base);
      TSTORE(o_global, o_l0);
      pipe_barrier(PIPE_ALL);
    }
  }
#endif

#if defined(__DAV_C220_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // This is the key change in step 03: each vector sub-core loads its own half
  // of the triangular mask once, outside the chunk loop, and reuses it.
  HalfMaskGlobal mask_global(causal_mask + vid * HalfChunk * ChunkSize);
  TLOAD(mask_ub, mask_global);
  pipe_barrier(PIPE_ALL);

  for (int64_t work_idx = 0; work_idx < (total_work + block_num - 1) / block_num;
       ++work_idx) {
    const int64_t pid = work_idx * block_num + cid;
    if (pid >= total_work) {
      continue;
    }

    const int64_t workspace1_base =
        cid * Workspace1Elems + vid * HalfChunk * ChunkSize;
    const int64_t workspace2_base =
        cid * Workspace2Elems + vid * HalfHidden * HiddenSize;

    TEXPANDS(hsum_ub, 0.0f);
    pipe_barrier(PIPE_ALL);
    HalfHiddenGlobal init_h_global(workspace_2 + workspace2_base);
    TSTORE(init_h_global, hsum_ub);
    pipe_barrier(PIPE_ALL);
    SetCrossFlag<PIPE_MTE3>(1, 2);

    for (int64_t i = 0; i < chunk_num; ++i) {
      WaitCrossFlag(0);

      HalfAccGlobal acc_global(workspace_1 + workspace1_base);
      HalfHiddenGlobal h_global(workspace_2 + workspace2_base);
      TLOAD(acc_ub, acc_global);
      TLOAD(h_ub, h_global);
      pipe_barrier(PIPE_ALL);
      // Elementwise multiply is much cheaper than the scalar if-statements from
      // step 02, but the numerical effect is identical.
      TMUL(masked_acc_ub, acc_ub, mask_ub);

      TADD(hsum_ub, hsum_ub, h_ub);
      pipe_barrier(PIPE_ALL);
      TSTORE(acc_global, masked_acc_ub);
      TSTORE(h_global, hsum_ub);
      pipe_barrier(PIPE_ALL);
      SetCrossFlag<PIPE_MTE3>(1, 2);
    }
  }
#endif
}

extern "C" __global__ AICORE void launch_linear_attention(
    __gm__ uint8_t *q, __gm__ uint8_t *k, __gm__ uint8_t *v,
    __gm__ uint8_t *workspace_1, __gm__ uint8_t *workspace_2,
    __gm__ uint8_t *causal_mask, __gm__ uint8_t *o, int64_t batch_size,
    int64_t seq_len, uint64_t ffts_addr) {
  main_kernel<LINEAR_ATTN_H, LINEAR_ATTN_D, LINEAR_ATTN_C>(
      reinterpret_cast<__gm__ half *>(q), reinterpret_cast<__gm__ half *>(k),
      reinterpret_cast<__gm__ half *>(v),
      reinterpret_cast<__gm__ half *>(workspace_1),
      reinterpret_cast<__gm__ half *>(workspace_2),
      reinterpret_cast<__gm__ half *>(causal_mask),
      reinterpret_cast<__gm__ half *>(o), batch_size, seq_len, ffts_addr);
}

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *q,
                            uint8_t *k, uint8_t *v, uint8_t *workspace_1,
                            uint8_t *workspace_2, uint8_t *causal_mask,
                            uint8_t *o,
                            int64_t batch_size, int64_t seq_len) {
  uint32_t ffts_len = 0;
  uint64_t ffts_addr = 0;
  rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
  launch_linear_attention<<<blockDim, nullptr, stream>>>(
      q, k, v, workspace_1, workspace_2, causal_mask, o, batch_size, seq_len,
      ffts_addr);
}
