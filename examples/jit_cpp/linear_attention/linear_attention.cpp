#include <pto/pto-inst.hpp>
#include <runtime/rt_ffts.h>
#include <type_traits>

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

template <pipe_t Src, pipe_t Dst>
AICORE inline void SetFlag(uint32_t id) {
  set_flag(Src, Dst, static_cast<event_t>(id));
}

template <pipe_t Src, pipe_t Dst>
AICORE inline void WaitFlag(uint32_t id) {
  wait_flag(Src, Dst, static_cast<event_t>(id));
}

template <int M, int N, int K, bool TransposeA = false, bool TransposeB = false>
AICORE inline void MatmulL1(
    TileAcc<float, M, N, M, N> &dst,
    std::conditional_t<TransposeA, L1Mat<half, K, M>, L1Mat<half, M, K>> &a_l1,
    std::conditional_t<TransposeB, L1Mat<half, N, K>, L1Mat<half, K, N>> &b_l1,
    bool init) {
  if constexpr ((K % 64 == 0) && (K > 64)) {
    constexpr int KStep = 64;
    constexpr int Parts = K / KStep;
    constexpr uintptr_t AStepBytes = M * KStep * sizeof(half);
    constexpr uintptr_t BStepBytes = KStep * N * sizeof(half);

    TileLeft<half, M, KStep, M, KStep> a_l0[2];
    TileRight<half, KStep, N, KStep, N> b_l0[2];
    TASSIGN(a_l0[0], static_cast<uintptr_t>(0));
    TASSIGN(a_l0[1], AStepBytes);
    TASSIGN(b_l0[0], static_cast<uintptr_t>(0));
    TASSIGN(b_l0[1], BStepBytes);

    SetFlag<PIPE_M, PIPE_MTE1>(0);
    SetFlag<PIPE_M, PIPE_MTE1>(1);

    for (int part = 0; part < Parts; ++part) {
      const int buf = part & 1;
      WaitFlag<PIPE_M, PIPE_MTE1>(buf);

      if constexpr (TransposeA) {
        L1MatTrans<half, M, K> a_view;
        TRESHAPE(a_view, a_l1);
        TEXTRACT(a_l0[buf], a_view, 0, part * KStep);
      } else {
        TEXTRACT(a_l0[buf], a_l1, 0, part * KStep);
      }

      if constexpr (TransposeB) {
        L1MatTrans<half, K, N> b_view;
        TRESHAPE(b_view, b_l1);
        TEXTRACT(b_l0[buf], b_view, part * KStep, 0);
      } else {
        TEXTRACT(b_l0[buf], b_l1, part * KStep, 0);
      }

      SetFlag<PIPE_MTE1, PIPE_M>(buf);
      WaitFlag<PIPE_MTE1, PIPE_M>(buf);

      if (init && part == 0) {
        TMATMUL(dst, a_l0[buf], b_l0[buf]);
      } else {
        TMATMUL_ACC(dst, dst, a_l0[buf], b_l0[buf]);
      }

      SetFlag<PIPE_M, PIPE_MTE1>(buf);
    }

    WaitFlag<PIPE_M, PIPE_MTE1>(0);
    WaitFlag<PIPE_M, PIPE_MTE1>(1);
    pipe_barrier(PIPE_ALL);
  } else {
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
}

struct LinearAttnSeqInfo {
  uint32_t bos;
  uint32_t seq_len;
  uint32_t chunk_offset;
  uint32_t token_base_offset;
  uint32_t row_stride;
};

AICORE inline uint32_t DivCeilU32(uint32_t x, uint32_t y) {
  return (x + y - 1) / y;
}

AICORE inline LinearAttnSeqInfo GetLinearAttnSeqInfo(
    uint32_t seq_idx, uint32_t head_idx, uint32_t num_heads,
    uint32_t hidden_size, uint32_t chunk_size, uint32_t fixed_seq_len,
    bool seq_first, __gm__ int32_t *cu_seqlens) {
  if (!seq_first) {
    const uint32_t chunk_num = DivCeilU32(fixed_seq_len, chunk_size);
    return {
        seq_idx * fixed_seq_len,
        fixed_seq_len,
        seq_idx * chunk_num,
        ((seq_idx * num_heads + head_idx) * fixed_seq_len) * hidden_size,
        hidden_size,
    };
  }

  if (cu_seqlens == nullptr) {
    const uint32_t bos = seq_idx * fixed_seq_len;
    const uint32_t chunk_num = DivCeilU32(fixed_seq_len, chunk_size);
    return {
        bos,
        fixed_seq_len,
        seq_idx * chunk_num,
        bos * num_heads * hidden_size + head_idx * hidden_size,
        num_heads * hidden_size,
    };
  }

  uint32_t bos = 0;
  uint32_t chunk_offset = 0;
  for (uint32_t i = 0; i < seq_idx; ++i) {
    const uint32_t seq_start = static_cast<uint32_t>(cu_seqlens[i]);
    const uint32_t seq_end = static_cast<uint32_t>(cu_seqlens[i + 1]);
    chunk_offset += DivCeilU32(seq_end - seq_start, chunk_size);
  }
  bos = static_cast<uint32_t>(cu_seqlens[seq_idx]);
  const uint32_t eos = static_cast<uint32_t>(cu_seqlens[seq_idx + 1]);
  return {
      bos,
      eos - bos,
      chunk_offset,
      bos * num_heads * hidden_size + head_idx * hidden_size,
      num_heads * hidden_size,
  };
}

template <int32_t NumHeads, int32_t HiddenSize, int32_t ChunkSize>
AICORE void main_kernel_precomputed(__gm__ half *q, __gm__ half *k,
                                    __gm__ half *v, __gm__ half *workspace_1,
                                    __gm__ half *h, __gm__ half *causal_mask,
                                    __gm__ half *o, __gm__ int32_t *cu_seqlens,
                                    int64_t batch_size, int64_t seq_len,
                                    bool seq_first, uint64_t ffts_addr) {
  constexpr int32_t HalfChunk = ChunkSize / 2;
  constexpr int32_t HalfHidden = HiddenSize / 2;
  constexpr int32_t ChunkElems = ChunkSize * HiddenSize;
  constexpr int32_t Workspace1Elems = ChunkSize * ChunkSize;
  constexpr int32_t HiddenElems = HiddenSize * HiddenSize;

  constexpr int32_t QL1Addr = 0;
  constexpr int32_t KL1Addr = QL1Addr + ChunkElems * sizeof(half);
  constexpr int32_t VL1Addr = KL1Addr + ChunkElems * sizeof(half);
  constexpr int32_t HL1Addr = VL1Addr + ChunkElems * sizeof(half);
  constexpr int32_t AccL1Addr = HL1Addr + HiddenElems * sizeof(half);
  constexpr int32_t SharedL0Addr = 0;
  constexpr int32_t AccUbAddr = 0;
  constexpr int32_t MaskUbAddr = AccUbAddr + HalfChunk * ChunkSize * sizeof(half);

  using ChunkGlobal =
      GlobalTensor<half, TileShape2D<half, ChunkSize, HiddenSize, Layout::ND>,
                   BaseShape2D<half, ChunkSize, HiddenSize, Layout::ND>,
                   Layout::ND>;
  using ChunkGlobalDynShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using ChunkGlobalDynStride = Stride<1, 1, 1, DYNAMIC, 1>;
  using ChunkGlobalDyn =
      GlobalTensor<half, ChunkGlobalDynShape, ChunkGlobalDynStride, Layout::ND>;
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
  using HalfMaskGlobal =
      GlobalTensor<half, TileShape2D<half, HalfChunk, ChunkSize, Layout::ND>,
                   BaseShape2D<half, HalfChunk, ChunkSize, Layout::ND>,
                   Layout::ND>;
  using OutGlobalDyn =
      GlobalTensor<half, ChunkGlobalDynShape, ChunkGlobalDynStride, Layout::ND>;

  using ChunkL1Dyn = Tile<TileType::Mat, half, ChunkSize, HiddenSize,
                          BLayout::ColMajor, DYNAMIC, DYNAMIC,
                          SLayout::RowMajor, 512, PadValue::Zero>;
  using OutL0Dyn =
      TileAcc<float, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC>;

  const int64_t total_work = batch_size * NumHeads;
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
  TileAcc<float, ChunkSize, HiddenSize, ChunkSize, HiddenSize> o_l0;
  TASSIGN(acc_l0, SharedL0Addr);
  TASSIGN(o_l0, SharedL0Addr);

  UbVec<half, HalfChunk, ChunkSize> acc_ub;
  UbVec<half, HalfChunk, ChunkSize> mask_ub;
  TASSIGN(acc_ub, AccUbAddr);
  TASSIGN(mask_ub, MaskUbAddr);

#if defined(__DAV_C220_CUBE__)
  for (int64_t work_idx = 0; work_idx < (total_work + block_num - 1) / block_num;
       ++work_idx) {
    const int64_t pid = work_idx * block_num + cid;
    if (pid >= total_work) {
      continue;
    }

    const uint32_t by = static_cast<uint32_t>(pid % NumHeads);
    const uint32_t bz = static_cast<uint32_t>(pid / NumHeads);
    const LinearAttnSeqInfo seq_info =
        GetLinearAttnSeqInfo(bz, by, NumHeads, HiddenSize, ChunkSize,
                             static_cast<uint32_t>(seq_len), seq_first,
                             cu_seqlens);
    const uint32_t chunk_num = DivCeilU32(seq_info.seq_len, ChunkSize);
    const int64_t workspace1_base = cid * Workspace1Elems;

    for (uint32_t i = 0; i < chunk_num; ++i) {
      const uint32_t row_start = i * ChunkSize;
      const uint32_t valid_rows =
          min(static_cast<uint32_t>(seq_info.seq_len - row_start),
              static_cast<uint32_t>(ChunkSize));
      const uint32_t token_offset = seq_info.token_base_offset +
                                    row_start * seq_info.row_stride;

      if (valid_rows == ChunkSize && seq_info.row_stride == HiddenSize) {
        ChunkGlobal q_global(q + token_offset);
        ChunkGlobal k_global(k + token_offset);
        ChunkGlobal v_global(v + token_offset);
        TLOAD(q_l1, q_global);
        TLOAD(k_l1, k_global);
        TLOAD(v_l1, v_global);
      } else {
        // Tail chunks load only the valid rows. The later causal masking zeros
        // the invalid score columns, so the untouched rows in these full-size
        // L1 tiles never contribute to the stored valid output rows.
        ChunkL1Dyn q_dyn(valid_rows, HiddenSize);
        ChunkL1Dyn k_dyn(valid_rows, HiddenSize);
        ChunkL1Dyn v_dyn(valid_rows, HiddenSize);
        TASSIGN(q_dyn, QL1Addr);
        TASSIGN(k_dyn, KL1Addr);
        TASSIGN(v_dyn, VL1Addr);
        ChunkGlobalDyn q_global_dyn(q + token_offset,
                                    {1, 1, 1, static_cast<int>(valid_rows),
                                     HiddenSize},
                                    {1, 1, 1,
                                     static_cast<int>(seq_info.row_stride), 1});
        ChunkGlobalDyn k_global_dyn(k + token_offset,
                                    {1, 1, 1, static_cast<int>(valid_rows),
                                     HiddenSize},
                                    {1, 1, 1,
                                     static_cast<int>(seq_info.row_stride), 1});
        ChunkGlobalDyn v_global_dyn(v + token_offset,
                                    {1, 1, 1, static_cast<int>(valid_rows),
                                     HiddenSize},
                                    {1, 1, 1,
                                     static_cast<int>(seq_info.row_stride), 1});
        TLOAD(q_dyn, q_global_dyn);
        TLOAD(k_dyn, k_global_dyn);
        TLOAD(v_dyn, v_global_dyn);
      }

      HiddenGlobal h_global(
          h + ((seq_info.chunk_offset + i) * NumHeads + by) * HiddenElems);
      TLOAD(h_l1, h_global);
      pipe_barrier(PIPE_ALL);

      MatmulL1<ChunkSize, ChunkSize, HiddenSize, false, true>(acc_l0, q_l1, k_l1,
                                                              true);
      AccGlobal acc_global(workspace_1 + workspace1_base);
      TSTORE(acc_global, acc_l0);
      pipe_barrier(PIPE_ALL);
      SetCrossFlag<PIPE_FIX>(0, 2);

      WaitCrossFlag(1);
      TLOAD(acc_l1, acc_global);
      pipe_barrier(PIPE_ALL);

      MatmulL1<ChunkSize, HiddenSize, ChunkSize, false, false>(o_l0, acc_l1, v_l1,
                                                               true);
      MatmulL1<ChunkSize, HiddenSize, HiddenSize, false, false>(o_l0, q_l1, h_l1,
                                                                false);

      if (valid_rows == ChunkSize && seq_info.row_stride == HiddenSize) {
        ChunkGlobal o_global(o + token_offset);
        TSTORE(o_global, o_l0);
      } else {
        OutL0Dyn o_tail(valid_rows, HiddenSize);
        TASSIGN(o_tail, SharedL0Addr);
        OutGlobalDyn o_global_dyn(o + token_offset,
                                  {1, 1, 1, static_cast<int>(valid_rows),
                                   HiddenSize},
                                  {1, 1, 1,
                                   static_cast<int>(seq_info.row_stride), 1});
        TSTORE(o_global_dyn, o_tail);
      }
      pipe_barrier(PIPE_ALL);
    }
  }
#endif

#if defined(__DAV_C220_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  HalfMaskGlobal mask_global(causal_mask + vid * HalfChunk * ChunkSize);
  TLOAD(mask_ub, mask_global);
  pipe_barrier(PIPE_ALL);

  for (int64_t work_idx = 0; work_idx < (total_work + block_num - 1) / block_num;
       ++work_idx) {
    const int64_t pid = work_idx * block_num + cid;
    if (pid >= total_work) {
      continue;
    }

    const uint32_t by = static_cast<uint32_t>(pid % NumHeads);
    const uint32_t bz = static_cast<uint32_t>(pid / NumHeads);
    const LinearAttnSeqInfo seq_info =
        GetLinearAttnSeqInfo(bz, by, NumHeads, HiddenSize, ChunkSize,
                             static_cast<uint32_t>(seq_len), seq_first,
                             cu_seqlens);
    const uint32_t chunk_num = DivCeilU32(seq_info.seq_len, ChunkSize);
    const int64_t workspace1_base = cid * Workspace1Elems;

    for (uint32_t i = 0; i < chunk_num; ++i) {
      WaitCrossFlag(0);
      HalfAccGlobal acc_global(workspace_1 + workspace1_base +
                               vid * HalfChunk * ChunkSize);
      TLOAD(acc_ub, acc_global);
      pipe_barrier(PIPE_ALL);
      TMUL(acc_ub, acc_ub, mask_ub);
      pipe_barrier(PIPE_ALL);
      TSTORE(acc_global, acc_ub);
      pipe_barrier(PIPE_ALL);
      SetCrossFlag<PIPE_MTE3>(1, 2);
    }
  }
#endif
}

template <int32_t NumHeads, int32_t HiddenSize, int32_t ChunkSize>
AICORE void main_kernel(__gm__ half *q, __gm__ half *k, __gm__ half *v,
                        __gm__ half *workspace_1, __gm__ half *workspace_2,
                        __gm__ half *causal_mask, __gm__ half *o,
                        int64_t batch_size, int64_t seq_len,
                        uint64_t ffts_addr) {
  constexpr int32_t StageCount = 2;
  constexpr bool UseTwoStagePipeline = (ChunkSize >= 128);
  constexpr bool InplaceMaskApply = (ChunkSize >= 128);
  constexpr int32_t VecNum = 2;
  constexpr int32_t HalfChunk = ChunkSize / VecNum;
  constexpr int32_t HalfHidden = HiddenSize / VecNum;
  constexpr int32_t ChunkElems = ChunkSize * HiddenSize;
  constexpr int32_t Workspace1SlotElems = ChunkSize * ChunkSize;
  constexpr int32_t Workspace2SlotElems = HiddenSize * HiddenSize;
  constexpr int32_t Workspace1Elems = StageCount * Workspace1SlotElems;
  constexpr int32_t Workspace2Elems = StageCount * Workspace2SlotElems;

  constexpr int32_t QL1Addr = 0;
  constexpr int32_t KL1Addr = QL1Addr + ChunkElems * sizeof(half);
  constexpr int32_t VL1Addr = KL1Addr + ChunkElems * sizeof(half);
  constexpr int32_t HL1Addr = VL1Addr + ChunkElems * sizeof(half);
  constexpr int32_t AccL1Addr = HL1Addr + Workspace2SlotElems * sizeof(half);
  constexpr int32_t HNextL1Addr = AccL1Addr + Workspace1SlotElems * sizeof(half);

  constexpr int32_t SharedL0Addr = 0;

  constexpr int32_t HsumUbAddr = 0;
  constexpr int32_t AccUbAddr =
      HsumUbAddr + HalfHidden * HiddenSize * sizeof(half);
  constexpr int32_t HUbAddr = AccUbAddr + HalfChunk * ChunkSize * sizeof(half);
  constexpr int32_t RawUBBytes =
      (HalfHidden * HiddenSize + HalfChunk * ChunkSize + HalfHidden * HiddenSize +
       HalfChunk * ChunkSize +
       (InplaceMaskApply ? 0 : HalfChunk * ChunkSize)) *
      sizeof(half);
  constexpr bool PreloadMask = RawUBBytes <= 72 * 1024;
  constexpr bool AliasMaskIntoH =
      !PreloadMask && (HalfHidden * HiddenSize >= HalfChunk * ChunkSize);
  constexpr int32_t MaskUbAddr =
      AliasMaskIntoH ? HUbAddr : HUbAddr + HalfHidden * HiddenSize * sizeof(half);
  constexpr int32_t MaskedAccUbAddr =
      InplaceMaskApply ? AccUbAddr : MaskUbAddr + HalfChunk * ChunkSize * sizeof(half);

  constexpr int32_t L0CBytes =
      (Workspace2SlotElems > Workspace1SlotElems
           ? (Workspace2SlotElems > ChunkElems ? Workspace2SlotElems : ChunkElems)
           : (Workspace1SlotElems > ChunkElems ? Workspace1SlotElems : ChunkElems)) *
      sizeof(float);
  constexpr int32_t UBBytes =
      (HalfHidden * HiddenSize + HalfChunk * ChunkSize +
       (AliasMaskIntoH ? HalfHidden * HiddenSize
                       : HalfHidden * HiddenSize + HalfChunk * ChunkSize) +
       (InplaceMaskApply ? 0 : HalfChunk * ChunkSize)) *
      sizeof(half);
  constexpr int32_t L1Bytes =
      UseTwoStagePipeline ? (HNextL1Addr + Workspace2SlotElems * sizeof(half))
                          : (AccL1Addr + Workspace1SlotElems * sizeof(half));
  static_assert((HiddenSize % 2) == 0, "HiddenSize must be even.");
  static_assert((ChunkSize % 2) == 0, "ChunkSize must be even.");
  static_assert(L0CBytes <= 112 * 1024,
                "Tile sizes exceed the validated L0C budget for this minimum kernel.");
  static_assert(L1Bytes <= 192 * 1024,
                "Tile sizes exceed the validated L1 budget for this minimum kernel.");
  static_assert(PreloadMask || AliasMaskIntoH,
                "Current minimum kernel requires either a preloaded mask or H UB large enough to alias the mask.");
  static_assert(UBBytes <= 72 * 1024,
                "Tile sizes exceed the validated UB budget for this minimum kernel.");

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
  L1Mat<half, HiddenSize, HiddenSize> h_next_l1;
  L1Mat<half, ChunkSize, ChunkSize> acc_l1;
  TASSIGN(q_l1, QL1Addr);
  TASSIGN(k_l1, KL1Addr);
  TASSIGN(v_l1, VL1Addr);
  TASSIGN(h_l1, HL1Addr);
  TASSIGN(h_next_l1, HNextL1Addr);
  TASSIGN(acc_l1, AccL1Addr);

  TileAcc<float, ChunkSize, ChunkSize, ChunkSize, ChunkSize> acc_l0;
  TileAcc<float, HiddenSize, HiddenSize, HiddenSize, HiddenSize> h_l0;
  TileAcc<float, ChunkSize, HiddenSize, ChunkSize, HiddenSize> o_l0;
  TASSIGN(acc_l0, SharedL0Addr);
  TASSIGN(h_l0, SharedL0Addr);
  TASSIGN(o_l0, SharedL0Addr);

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

    if constexpr (UseTwoStagePipeline) {
      const int32_t flag_base = static_cast<int32_t>((work_idx & 3) * 6);
      int32_t h_buf = 0;
      WaitCrossFlag(flag_base + 4);
      HiddenGlobal zero_h_global(workspace_2 + workspace2_base + Workspace2SlotElems);
      TLOAD(h_l1, zero_h_global);
      pipe_barrier(PIPE_ALL);

      {
        const int64_t chunk_base = qkv_base;
        ChunkGlobal q_global(q + chunk_base);
        ChunkGlobal k_global(k + chunk_base);
        ChunkGlobal v_global(v + chunk_base);
        TLOAD(q_l1, q_global);
        TLOAD(k_l1, k_global);
        TLOAD(v_l1, v_global);
        pipe_barrier(PIPE_ALL);

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
        SetCrossFlag<PIPE_FIX>(flag_base, 2);
      }

      for (int64_t i = 0; i < chunk_num; ++i) {
        const int32_t slot = static_cast<int32_t>(i & 1);
        const int32_t next_slot = slot ^ 1;
        const int64_t chunk_base = qkv_base + i * ChunkElems;

        if (i + 1 < chunk_num) {
          const int64_t next_chunk_base = qkv_base + (i + 1) * ChunkElems;
          const int64_t next_workspace1_base =
              workspace1_base + next_slot * Workspace1SlotElems;
          const int64_t next_workspace2_base =
              workspace2_base + next_slot * Workspace2SlotElems;

          ChunkGlobal q_global(q + next_chunk_base);
          ChunkGlobal k_global(k + next_chunk_base);
          ChunkGlobal v_global(v + next_chunk_base);
          TLOAD(q_l1, q_global);
          TLOAD(k_l1, k_global);
          TLOAD(v_l1, v_global);
          pipe_barrier(PIPE_ALL);

          MatmulL1<ChunkSize, ChunkSize, HiddenSize, false, true>(
              acc_l0, q_l1, k_l1, true);
          AccGlobal acc_global(workspace_1 + next_workspace1_base);
          TSTORE(acc_global, acc_l0);
          pipe_barrier(PIPE_ALL);

          MatmulL1<HiddenSize, HiddenSize, ChunkSize, true, false>(h_l0, k_l1,
                                                                   v_l1, true);
          HiddenGlobal h_out_global(workspace_2 + next_workspace2_base);
          TSTORE(h_out_global, h_l0);
          pipe_barrier(PIPE_ALL);
          SetCrossFlag<PIPE_FIX>(flag_base + next_slot, 2);
        }

        WaitCrossFlag(flag_base + 2 + slot);
        AccGlobal masked_acc_global(workspace_1 + workspace1_base +
                                    slot * Workspace1SlotElems);
        TLOAD(acc_l1, masked_acc_global);
        ChunkGlobal q_global(q + chunk_base);
        ChunkGlobal v_global(v + chunk_base);
        TLOAD(q_l1, q_global);
        TLOAD(v_l1, v_global);
        if (i + 1 < chunk_num) {
          HiddenGlobal next_h_global(workspace_2 + workspace2_base +
                                     slot * Workspace2SlotElems);
          if (h_buf == 0) {
            TLOAD(h_next_l1, next_h_global);
          } else {
            TLOAD(h_l1, next_h_global);
          }
        }
        pipe_barrier(PIPE_ALL);

        MatmulL1<ChunkSize, HiddenSize, ChunkSize, false, false>(o_l0, acc_l1,
                                                                 v_l1, true);
        if (h_buf == 0) {
          MatmulL1<ChunkSize, HiddenSize, HiddenSize, false, false>(o_l0, q_l1,
                                                                    h_l1, false);
        } else {
          MatmulL1<ChunkSize, HiddenSize, HiddenSize, false, false>(o_l0, q_l1,
                                                                    h_next_l1,
                                                                    false);
        }

        ChunkGlobal o_global(o + chunk_base);
        TSTORE(o_global, o_l0);
        pipe_barrier(PIPE_ALL);

        if (i + 1 < chunk_num) {
          h_buf ^= 1;
        }
      }
      SetCrossFlag<PIPE_FIX>(flag_base + 5, 2);
    } else {
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

        WaitCrossFlag(1);
        AccGlobal masked_acc_global(workspace_1 + workspace1_base);
        TLOAD(acc_l1, masked_acc_global);
        pipe_barrier(PIPE_ALL);

        MatmulL1<ChunkSize, HiddenSize, ChunkSize, false, false>(o_l0, acc_l1,
                                                                 v_l1, true);
        MatmulL1<ChunkSize, HiddenSize, HiddenSize, false, false>(o_l0, q_l1, h_l1,
                                                                  false);

        ChunkGlobal o_global(o + chunk_base);
        TSTORE(o_global, o_l0);
        pipe_barrier(PIPE_ALL);
      }
    }
  }
#endif

#if defined(__DAV_C220_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  HalfMaskGlobal mask_global(causal_mask + vid * HalfChunk * ChunkSize);
  if constexpr (PreloadMask) {
    TLOAD(mask_ub, mask_global);
    pipe_barrier(PIPE_ALL);
  }

  for (int64_t work_idx = 0; work_idx < (total_work + block_num - 1) / block_num;
       ++work_idx) {
    const int64_t pid = work_idx * block_num + cid;
    if (pid >= total_work) {
      continue;
    }

    const int64_t workspace1_base = cid * Workspace1Elems;
    const int64_t workspace2_base = cid * Workspace2Elems;

    TEXPANDS(hsum_ub, 0.0f);
    pipe_barrier(PIPE_ALL);
    if constexpr (UseTwoStagePipeline) {
      const int32_t flag_base = static_cast<int32_t>((work_idx & 3) * 6);
      HalfHiddenGlobal init_h_global_0(workspace_2 + workspace2_base +
                                       vid * HalfHidden * HiddenSize);
      HalfHiddenGlobal init_h_global_1(workspace_2 + workspace2_base +
                                       Workspace2SlotElems +
                                       vid * HalfHidden * HiddenSize);
      TSTORE(init_h_global_0, hsum_ub);
      TSTORE(init_h_global_1, hsum_ub);
      pipe_barrier(PIPE_ALL);
      SetCrossFlag<PIPE_MTE3>(flag_base + 4, 2);

      for (int64_t i = 0; i < chunk_num; ++i) {
        const int32_t slot = static_cast<int32_t>(i & 1);
        WaitCrossFlag(flag_base + slot);

        const int64_t slot_workspace1_base =
            workspace1_base + slot * Workspace1SlotElems;
        const int64_t slot_workspace2_base =
            workspace2_base + slot * Workspace2SlotElems;
        HalfAccGlobal acc_global(workspace_1 + slot_workspace1_base +
                                 vid * HalfChunk * ChunkSize);
        HalfHiddenGlobal h_global(workspace_2 + slot_workspace2_base +
                                  vid * HalfHidden * HiddenSize);
        TLOAD(acc_ub, acc_global);
        TLOAD(h_ub, h_global);
        pipe_barrier(PIPE_ALL);

        // Precompute the chunk carry state H_t = sum_{j<=t}(K_j^T V_j) on the
        // vector core, then write it back for the cube core output stage.
        TADD(hsum_ub, hsum_ub, h_ub);
        pipe_barrier(PIPE_ALL);
        if constexpr (!PreloadMask) {
          TLOAD(mask_ub, mask_global);
          pipe_barrier(PIPE_ALL);
        }
        if constexpr (InplaceMaskApply) {
          TMUL(acc_ub, acc_ub, mask_ub);
        } else {
          TMUL(masked_acc_ub, acc_ub, mask_ub);
        }
        pipe_barrier(PIPE_ALL);
        if constexpr (InplaceMaskApply) {
          TSTORE(acc_global, acc_ub);
        } else {
          TSTORE(acc_global, masked_acc_ub);
        }
        TSTORE(h_global, hsum_ub);
        pipe_barrier(PIPE_ALL);
        SetCrossFlag<PIPE_MTE3>(flag_base + 2 + slot, 2);
      }
      WaitCrossFlag(flag_base + 5);
    } else {
      HalfHiddenGlobal init_h_global(workspace_2 + workspace2_base +
                                     vid * HalfHidden * HiddenSize);
      TSTORE(init_h_global, hsum_ub);
      pipe_barrier(PIPE_ALL);
      SetCrossFlag<PIPE_MTE3>(1, 2);

      for (int64_t i = 0; i < chunk_num; ++i) {
        WaitCrossFlag(0);

        HalfAccGlobal acc_global(workspace_1 + workspace1_base +
                                 vid * HalfChunk * ChunkSize);
        HalfHiddenGlobal h_global(workspace_2 + workspace2_base +
                                  vid * HalfHidden * HiddenSize);
        TLOAD(acc_ub, acc_global);
        TLOAD(h_ub, h_global);
        pipe_barrier(PIPE_ALL);

        // Precompute the chunk carry state H_t = sum_{j<=t}(K_j^T V_j) on the
        // vector core, then write it back for the cube core output stage.
        TADD(hsum_ub, hsum_ub, h_ub);
        pipe_barrier(PIPE_ALL);
        if constexpr (!PreloadMask) {
          TLOAD(mask_ub, mask_global);
          pipe_barrier(PIPE_ALL);
        }
        if constexpr (InplaceMaskApply) {
          TMUL(acc_ub, acc_ub, mask_ub);
        } else {
          TMUL(masked_acc_ub, acc_ub, mask_ub);
        }
        pipe_barrier(PIPE_ALL);
        if constexpr (InplaceMaskApply) {
          TSTORE(acc_global, acc_ub);
        } else {
          TSTORE(acc_global, masked_acc_ub);
        }
        TSTORE(h_global, hsum_ub);
        pipe_barrier(PIPE_ALL);
        SetCrossFlag<PIPE_MTE3>(1, 2);
      }
    }
  }
#endif
}

extern "C" __global__ AICORE void launch_linear_attention(
    __gm__ uint8_t *q, __gm__ uint8_t *k, __gm__ uint8_t *v,
    __gm__ uint8_t *workspace_1, __gm__ uint8_t *workspace_2,
    __gm__ uint8_t *causal_mask, __gm__ uint8_t *o,
    __gm__ int32_t *cu_seqlens, int64_t batch_size, int64_t seq_len,
    uint32_t seq_first, uint32_t use_precomputed_h, uint64_t ffts_addr) {
  if (use_precomputed_h != 0) {
    main_kernel_precomputed<LINEAR_ATTN_H, LINEAR_ATTN_D, LINEAR_ATTN_C>(
        reinterpret_cast<__gm__ half *>(q), reinterpret_cast<__gm__ half *>(k),
        reinterpret_cast<__gm__ half *>(v),
        reinterpret_cast<__gm__ half *>(workspace_1),
        reinterpret_cast<__gm__ half *>(workspace_2),
        reinterpret_cast<__gm__ half *>(causal_mask),
        reinterpret_cast<__gm__ half *>(o), cu_seqlens, batch_size, seq_len,
        seq_first != 0, ffts_addr);
    return;
  }

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
                            uint8_t *o, int32_t *cu_seqlens,
                            int64_t batch_size, int64_t seq_len,
                            uint32_t seq_first, uint32_t use_precomputed_h) {
  uint32_t ffts_len = 0;
  uint64_t ffts_addr = 0;
  rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
  launch_linear_attention<<<blockDim, nullptr, stream>>>(
      q, k, v, workspace_1, workspace_2, causal_mask, o, cu_seqlens,
      batch_size, seq_len, seq_first, use_precomputed_h, ffts_addr);
}
