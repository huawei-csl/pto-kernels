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

namespace linear_attention_pto {

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

template <typename T, uint32_t M, uint32_t N, uint32_t M_L1, uint32_t N_L1,
          bool transpose = false>
AICORE PTO_INLINE void copy_l1_to_l0a(
    TileMatL0A<T, M, N, M, N> &l0a,
    std::conditional_t<transpose, TileMatL1ZN<T, M_L1, N_L1, M_L1, N_L1>,
                       TileMatL1<T, M_L1, N_L1, M_L1, N_L1>> &src,
    uint32_t index_row, uint32_t index_col) {
  pto::TEXTRACT(l0a, src, index_row, index_col);
}

template <typename T, uint32_t M, uint32_t N, uint32_t M_L1, uint32_t N_L1,
          bool transpose = false>
AICORE PTO_INLINE void copy_l1_to_l0b(
    TileMatL0B<T, M, N, M, N> &l0b,
    std::conditional_t<transpose, TileMatL1ZN<T, M_L1, N_L1, M_L1, N_L1>,
                       TileMatL1<T, M_L1, N_L1, M_L1, N_L1>> &src,
    uint32_t index_row, uint32_t index_col) {
  pto::TEXTRACT(l0b, src, index_row, index_col);
}

template <typename T1, typename T2, uint32_t M, uint32_t N, uint32_t K,
          uint32_t validM = M, uint32_t validN = N, uint32_t validK = K,
          uint32_t K_tail, bool transpose_A = false, bool transpose_B = false>
AICORE PTO_INLINE void gemm_v0(
    std::conditional_t<transpose_A, TileMatL1<T1, K, M, validK, validM>,
                       TileMatL1<T1, M, K, validM, validK>> &A,
    std::conditional_t<transpose_B, TileMatL1<T1, N, K, validN, validK>,
                       TileMatL1<T1, K, N, validK, validN>> &B,
    pto::TileAcc<T2, M, N, validM, validN> &C, bool clear) {
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
        copy_l1_to_l0a<T1, M, K_tail, M, K, false>(l0a, A, 0,
                                                   kL0Idx * K_tail);
      } else {
        TileMatL1ZN<T1, M, K, validM, validK> A_t;
        pto::TRESHAPE(A_t, A);
        copy_l1_to_l0a<T1, M, K_tail, M, K, true>(l0a, A_t, 0,
                                                  kL0Idx * K_tail);
      }

      if constexpr (!transpose_B) {
        copy_l1_to_l0b<T1, K_tail, N, K, N, false>(l0b, B,
                                                   kL0Idx * K_tail, 0);
      } else {
        TileMatL1ZN<T1, K, N, validK, validN> B_t;
        pto::TRESHAPE(B_t, B);
        copy_l1_to_l0b<T1, K_tail, N, K, N, true>(l0b, B_t,
                                                  kL0Idx * K_tail, 0);
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
        copy_l1_to_l0a<T1, M, kL0Size, M, K, false>(l0a, A, 0,
                                                    kL0Idx * kL0Size);
      } else {
        TileMatL1ZN<T1, M, K, validM, validK> A_t;
        pto::TRESHAPE(A_t, A);
        copy_l1_to_l0a<T1, M, kL0Size, M, K, true>(l0a, A_t, 0,
                                                   kL0Idx * kL0Size);
      }

      if constexpr (!transpose_B) {
        copy_l1_to_l0b<T1, kL0Size, N, K, N, false>(l0b, B,
                                                    kL0Idx * kL0Size, 0);
      } else {
        TileMatL1ZN<T1, K, N, validK, validN> B_t;
        pto::TRESHAPE(B_t, B);
        copy_l1_to_l0b<T1, kL0Size, N, K, N, true>(l0b, B_t,
                                                   kL0Idx * kL0Size, 0);
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

template <typename T1, typename T2, int32_t shape1, int32_t shape2,
          int32_t shape3, int32_t shape4, int32_t shape5, int32_t stride1,
          int32_t stride2, int32_t stride3, int32_t stride4, int32_t stride5,
          uint32_t valid1, uint32_t valid2>
AICORE PTO_INLINE void copy_gm_to_l1(__gm__ T1 *handle, int32_t buffer_addr,
                                     int32_t offset, int32_t actualTailM = 0,
                                     int32_t actualTailN = 0) {
  constexpr uint8_t len = sizeof(T2);
  const bool useTail = shape4 == valid1 && shape5 == valid2;
  const int tailM = (useTail && actualTailM != 0) ? actualTailM : valid1;
  const int tailN = (useTail && actualTailN != 0) ? actualTailN : valid2;

  TileMatL1<T2, shape4, shape5, pto::DYNAMIC, pto::DYNAMIC> tile(tailM, tailN);
  pto::TASSIGN(tile, buffer_addr + offset * len);

  pto::Shape<shape1, shape2, shape3, pto::DYNAMIC, pto::DYNAMIC> dynamic_shape;
  dynamic_shape.shape[3] = useTail ? tailM : shape4;
  dynamic_shape.shape[4] = useTail ? tailN : shape5;

  pto::GlobalTensor<
      T1, pto::Shape<shape1, shape2, shape3, pto::DYNAMIC, pto::DYNAMIC>,
      pto::Stride<stride1, stride2, stride3, stride4, stride5>>
      global_tensor(handle, dynamic_shape);
  pto::TLOAD(tile, global_tensor);
}

template <typename T1, typename T2, int32_t shape1, int32_t shape2,
          int32_t shape3, int32_t shape4, int32_t shape5, int32_t stride1,
          int32_t stride2, int32_t stride3, int32_t stride4, int32_t stride5,
          uint32_t valid1, uint32_t valid2>
AICORE PTO_INLINE void copy_l0c_to_gm(__gm__ T1 *handle, int32_t buffer_addr,
                                      int32_t offset, int32_t actualTailM = 0,
                                      int32_t actualTailN = 0) {
  constexpr uint8_t len = sizeof(T2);
  const bool useTail = shape4 == valid1 && shape5 == valid2;
  const int tailM = (useTail && actualTailM != 0) ? actualTailM : valid1;
  const int tailN = (useTail && actualTailN != 0) ? actualTailN : valid2;

  pto::TileAcc<T2, shape4, shape5, pto::DYNAMIC, pto::DYNAMIC> tile(tailM,
                                                                     tailN);
  pto::TASSIGN(tile, buffer_addr + offset * len);

  pto::Shape<shape1, shape2, shape3, pto::DYNAMIC, pto::DYNAMIC> dynamic_shape;
  dynamic_shape.shape[3] = useTail ? tailM : shape4;
  dynamic_shape.shape[4] = useTail ? tailN : shape5;

  pto::GlobalTensor<
      T1, pto::Shape<shape1, shape2, shape3, pto::DYNAMIC, pto::DYNAMIC>,
      pto::Stride<stride1, stride2, stride3, stride4, stride5>>
      global_tensor(handle, dynamic_shape);
  pto::TSTORE(global_tensor, tile);
}

template <typename T1, typename T2, int32_t shape1, int32_t shape2,
          int32_t shape3, int32_t shape4, int32_t shape5, int32_t stride1,
          int32_t stride2, int32_t stride3, int32_t stride4, int32_t stride5,
          uint32_t ub_shape1, uint32_t ub_shape2,
          pto::PadValue PadVal = pto::PadValue::Null>
AICORE PTO_INLINE void copy_gm_to_ub(__gm__ T1 *handle, int32_t ub_shape_addr,
                                     int32_t ub_offset, int32_t valid_row,
                                     int32_t valid_col) {
  constexpr uint8_t len = sizeof(T2);
  pto::Shape<shape1, shape2, shape3, pto::DYNAMIC, pto::DYNAMIC> dynamic_shape;
  dynamic_shape.shape[3] = valid_row;
  dynamic_shape.shape[4] = valid_col;

  pto::GlobalTensor<
      T1, pto::Shape<shape1, shape2, shape3, pto::DYNAMIC, pto::DYNAMIC>,
      pto::Stride<stride1, stride2, stride3, stride4, stride5>>
      global_tensor(handle, dynamic_shape);

  if constexpr (std::is_same_v<T1, T2>) {
    using SrcTile = TileUbDataND<T2, ub_shape1, ub_shape2, pto::DYNAMIC,
                                 pto::DYNAMIC, PadVal>;
    SrcTile src_tile(valid_row, valid_col);
    pto::TASSIGN(src_tile, ub_shape_addr + ub_offset * len);
    pto::TLOAD(src_tile, global_tensor);

    if constexpr (PadVal != pto::PadValue::Null) {
      if (valid_row != static_cast<int32_t>(ub_shape1) ||
          valid_col != static_cast<int32_t>(ub_shape2)) {
        using DstTile = pto::Tile<pto::TileType::Vec, T2, ub_shape1,
                                  ub_shape2, pto::BLayout::RowMajor, ub_shape1,
                                  ub_shape2, pto::SLayout::NoneBox, 512,
                                  PadVal>;
        DstTile dst_tile;
        pto::TASSIGN(dst_tile, ub_shape_addr + ub_offset * len);
        pto::TFILLPAD_INPLACE(dst_tile, src_tile);
      }
    }
  } else {
    TileUbDataND<T1, ub_shape1, ub_shape2, pto::DYNAMIC, pto::DYNAMIC>
        temp_src_ub(valid_row, valid_col);
    pto::TASSIGN(temp_src_ub, ub_shape_addr + ub_offset * len);
    pto::TLOAD(temp_src_ub, global_tensor);

    TileUbDataND<T2, ub_shape1, ub_shape2, pto::DYNAMIC, pto::DYNAMIC>
        temp_dst_ub(valid_row, valid_col);
    pto::TASSIGN(temp_dst_ub, ub_shape_addr + ub_offset * len);
    pto::TCVT(temp_dst_ub, temp_src_ub, pto::RoundMode::CAST_NONE);
  }
}

template <typename T1, typename T2, int32_t shape1, int32_t shape2,
          int32_t shape3, int32_t shape4, int32_t shape5, int32_t stride1,
          int32_t stride2, int32_t stride3, int32_t stride4, int32_t stride5,
          uint32_t ub_shape1, uint32_t ub_shape2>
AICORE PTO_INLINE void copy_ub_to_gm(__gm__ T1 *handle, int32_t ub_shape_addr,
                                     int32_t ub_offset, int32_t valid_row,
                                     int32_t valid_col) {
  pto::Shape<shape1, shape2, shape3, pto::DYNAMIC, pto::DYNAMIC> dynamic_shape;
  dynamic_shape.shape[3] = valid_row;
  dynamic_shape.shape[4] = valid_col;

  pto::GlobalTensor<
      T1, pto::Shape<shape1, shape2, shape3, pto::DYNAMIC, pto::DYNAMIC>,
      pto::Stride<stride1, stride2, stride3, stride4, stride5>>
      global_tensor(handle, dynamic_shape);

  constexpr uint8_t len = sizeof(T2);
  constexpr bool use_nd = (static_cast<uint64_t>(ub_shape2) * len) >= 32;

  if constexpr (std::is_same_v<T1, T2>) {
    if constexpr (use_nd) {
      TileUbDataND<T2, ub_shape1, ub_shape2, pto::DYNAMIC, pto::DYNAMIC> temp_ub(
          valid_row, valid_col);
      pto::TASSIGN(temp_ub, ub_shape_addr + ub_offset * len);
      pto::TSTORE(global_tensor, temp_ub);
    } else {
      TileUbDataDN<T2, ub_shape1, ub_shape2, pto::DYNAMIC, pto::DYNAMIC> temp_ub(
          valid_row, valid_col);
      pto::TASSIGN(temp_ub, ub_shape_addr + ub_offset * len);
      pto::TSTORE(global_tensor, temp_ub);
    }
  } else {
    if constexpr (use_nd) {
      TileUbDataND<T2, ub_shape1, ub_shape2, pto::DYNAMIC, pto::DYNAMIC>
          temp_src_ub(valid_row, valid_col);
      pto::TASSIGN(temp_src_ub, ub_shape_addr + ub_offset * len);
      TileUbDataND<T1, ub_shape1, ub_shape2, pto::DYNAMIC, pto::DYNAMIC>
          temp_dst_ub(valid_row, valid_col);
      pto::TASSIGN(temp_dst_ub, ub_shape_addr + ub_offset * sizeof(T1));
      pto::TCVT(temp_dst_ub, temp_src_ub, pto::RoundMode::CAST_NONE);
      pto::TSTORE(global_tensor, temp_dst_ub);
    } else {
      TileUbDataDN<T2, ub_shape1, ub_shape2, pto::DYNAMIC, pto::DYNAMIC>
          temp_src_ub(valid_row, valid_col);
      pto::TASSIGN(temp_src_ub, ub_shape_addr + ub_offset * len);
      TileUbDataDN<T1, ub_shape1, ub_shape2, pto::DYNAMIC, pto::DYNAMIC>
          temp_dst_ub(valid_row, valid_col);
      pto::TASSIGN(temp_dst_ub, ub_shape_addr + ub_offset * sizeof(T1));
      pto::TCVT(temp_dst_ub, temp_src_ub, pto::RoundMode::CAST_NONE);
      pto::TSTORE(global_tensor, temp_dst_ub);
    }
  }
}

template <pipe_t pipe>
AICORE PTO_INLINE void set_cross_flag(int32_t flag, int32_t mode) {
  const int config = 1 | (mode << 4) | (flag << 8);
  ffts_cross_core_sync(pipe, config);
}

AICORE PTO_INLINE void wait_cross_flag(int32_t flag) { wait_flag_dev(flag); }

}  // namespace linear_attention_pto

template <int32_t NumHeads, int32_t HiddenSize, int32_t ChunkSize>
AICORE void main_kernel(__gm__ half *Q_handle, __gm__ half *K_handle,
                        __gm__ half *V_handle, __gm__ half *workspace_1_handle,
                        __gm__ half *workspace_2_handle, __gm__ half *O_handle,
                        int64_t batch_size, int64_t seq_len,
                        uint64_t ffts_addr) {
  constexpr int32_t VecNum = 2;
  constexpr int32_t Workspace1Elems = ChunkSize * ChunkSize;
  constexpr int32_t Workspace2Elems = HiddenSize * HiddenSize;
  constexpr int32_t ChunkElems = ChunkSize * HiddenSize;
  constexpr int32_t HalfHidden = HiddenSize / VecNum;
  constexpr int32_t HalfChunk = ChunkSize / VecNum;
  constexpr int32_t QL1Addr = 0;
  constexpr int32_t KL1Addr = ChunkElems * sizeof(half);
  constexpr int32_t VL1Addr = 2 * ChunkElems * sizeof(half);
  constexpr int32_t HL1Addr = 3 * ChunkElems * sizeof(half);
  constexpr int32_t AccL0Addr = 0;
  constexpr int32_t HL0Addr = Workspace1Elems * sizeof(float);
  constexpr int32_t AccL1Addr =
      (3 * ChunkElems + Workspace2Elems) * sizeof(half);
  constexpr int32_t OL0Addr =
      Workspace1Elems * sizeof(float) + Workspace2Elems * sizeof(float);
  constexpr int32_t HsumUbAddr = 0;
  constexpr int32_t ZeroUbAddr = HalfHidden * HiddenSize * sizeof(half);
  constexpr int32_t AccUbAddr =
      HsumUbAddr + HalfHidden * HiddenSize * sizeof(half) +
      HalfChunk * ChunkSize * sizeof(half);
  constexpr int32_t HUbAddr =
      HsumUbAddr + HalfHidden * HiddenSize * sizeof(half) +
      2 * HalfChunk * ChunkSize * sizeof(half);

  const int64_t total_work = batch_size * NumHeads;
  const int64_t chunk_num = (seq_len + ChunkSize - 1) / ChunkSize;
  const int64_t cid = get_block_idx();
  const int64_t vid = get_subblockid();
  set_ffts_base_addr(ffts_addr);

  linear_attention_pto::TileMatL1<half, ChunkSize, HiddenSize, ChunkSize,
                                  HiddenSize>
      q_l1;
  TASSIGN(q_l1, QL1Addr);
  linear_attention_pto::TileMatL1<half, ChunkSize, HiddenSize, ChunkSize,
                                  HiddenSize>
      k_l1;
  TASSIGN(k_l1, KL1Addr);
  linear_attention_pto::TileMatL1<half, ChunkSize, HiddenSize, ChunkSize,
                                  HiddenSize>
      v_l1;
  TASSIGN(v_l1, VL1Addr);
  linear_attention_pto::TileMatL1<half, HiddenSize, HiddenSize, HiddenSize,
                                  HiddenSize>
      h_l1;
  TASSIGN(h_l1, HL1Addr);
  TileAcc<float, ChunkSize, ChunkSize, ChunkSize, ChunkSize> acc_l0;
  TASSIGN(acc_l0, AccL0Addr);
  TileAcc<float, HiddenSize, HiddenSize, HiddenSize, HiddenSize> h_l0;
  TASSIGN(h_l0, HL0Addr);
  linear_attention_pto::TileMatL1<half, ChunkSize, ChunkSize, ChunkSize,
                                  ChunkSize>
      acc_l1;
  TASSIGN(acc_l1, AccL1Addr);
  TileAcc<float, ChunkSize, HiddenSize, ChunkSize, HiddenSize> o_l0;
  TASSIGN(o_l0, OL0Addr);
  linear_attention_pto::TileUbDataND<half, HalfHidden, HiddenSize, HalfHidden,
                                     HiddenSize>
      hsum_ub;
  TASSIGN(hsum_ub, HsumUbAddr);
  linear_attention_pto::TileUbDataND<half, HalfChunk, ChunkSize, HalfChunk,
                                     ChunkSize>
      zero_ub;
  TASSIGN(zero_ub, ZeroUbAddr);
  linear_attention_pto::TileUbDataND<half, HalfChunk, ChunkSize, HalfChunk,
                                     ChunkSize>
      acc_ub;
  TASSIGN(acc_ub, AccUbAddr);
  linear_attention_pto::TileUbDataND<half, HalfHidden, HiddenSize, HalfHidden,
                                     HiddenSize>
      h_ub;
  TASSIGN(h_ub, HUbAddr);

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
    const int64_t out_base = qkv_base;
    const int64_t workspace1_base = cid * Workspace1Elems;
    const int64_t workspace2_base = cid * Workspace2Elems;

    linear_attention_pto::wait_cross_flag(1);

    for (int64_t i = 0; i < chunk_num; ++i) {
      const int64_t chunk_base = qkv_base + i * ChunkElems;

      set_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID1);
      wait_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID1);
      linear_attention_pto::copy_gm_to_l1<half, half, 1, 1, 1, ChunkSize,
                                          HiddenSize, ChunkElems, ChunkElems,
                                          ChunkElems, HiddenSize, 1, ChunkSize,
                                          HiddenSize>(Q_handle + chunk_base, 0,
                                                      0, ChunkSize, HiddenSize);
      linear_attention_pto::copy_gm_to_l1<half, half, 1, 1, 1, ChunkSize,
                                          HiddenSize, ChunkElems, ChunkElems,
                                          ChunkElems, HiddenSize, 1, ChunkSize,
                                          HiddenSize>(K_handle + chunk_base,
                                                      KL1Addr, 0, ChunkSize,
                                                      HiddenSize);
      linear_attention_pto::copy_gm_to_l1<half, half, 1, 1, 1, ChunkSize,
                                          HiddenSize, ChunkElems, ChunkElems,
                                          ChunkElems, HiddenSize, 1, ChunkSize,
                                          HiddenSize>(V_handle + chunk_base,
                                                      VL1Addr, 0,
                                                      ChunkSize, HiddenSize);
      linear_attention_pto::copy_gm_to_l1<half, half, 1, 1, 1, HiddenSize,
                                          HiddenSize, Workspace2Elems,
                                          Workspace2Elems, Workspace2Elems,
                                          HiddenSize, 1, HiddenSize,
                                          HiddenSize>(workspace_2_handle +
                                                          workspace2_base,
                                                      HL1Addr, 0,
                                                      HiddenSize, HiddenSize);
      set_flag(PIPE_MTE2, PIPE_M, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_M, EVENT_ID1);
      linear_attention_pto::gemm_v0<half, float, ChunkSize, ChunkSize,
                                    HiddenSize, ChunkSize, ChunkSize,
                                    HiddenSize, HiddenSize, false, true>(
          q_l1, k_l1, acc_l0, true);
      set_flag(PIPE_M, PIPE_FIX, EVENT_ID2);
      wait_flag(PIPE_M, PIPE_FIX, EVENT_ID2);
      linear_attention_pto::copy_l0c_to_gm<half, float, 1, 1, 1, ChunkSize,
                                           ChunkSize, Workspace1Elems,
                                           Workspace1Elems, Workspace1Elems,
                                           ChunkSize, 1, ChunkSize, ChunkSize>(
          workspace_1_handle + workspace1_base, 0, 0, ChunkSize, ChunkSize);
      set_flag(PIPE_MTE2, PIPE_M, EVENT_ID3);
      wait_flag(PIPE_MTE2, PIPE_M, EVENT_ID3);
      pipe_barrier(PIPE_M);
      linear_attention_pto::gemm_v0<half, float, HiddenSize, HiddenSize,
                                    ChunkSize, HiddenSize, HiddenSize,
                                    ChunkSize, ChunkSize, true, false>(
          k_l1, v_l1, h_l0, true);
      set_flag(PIPE_MTE2, PIPE_FIX, EVENT_ID4);
      wait_flag(PIPE_MTE2, PIPE_FIX, EVENT_ID4);
      set_flag(PIPE_M, PIPE_FIX, EVENT_ID5);
      wait_flag(PIPE_M, PIPE_FIX, EVENT_ID5);
      linear_attention_pto::copy_l0c_to_gm<half, float, 1, 1, 1, HiddenSize,
                                           HiddenSize, Workspace2Elems,
                                           Workspace2Elems, Workspace2Elems,
                                           HiddenSize, 1, HiddenSize,
                                           HiddenSize>(workspace_2_handle +
                                                           workspace2_base,
                                                       HL0Addr, 0,
                                                       HiddenSize, HiddenSize);
      linear_attention_pto::set_cross_flag<PIPE_FIX>(0, 2);

      linear_attention_pto::wait_cross_flag(1);
      set_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID6);
      wait_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID6);
      linear_attention_pto::copy_gm_to_l1<half, half, 1, 1, 1, ChunkSize,
                                          ChunkSize, Workspace1Elems,
                                          Workspace1Elems, Workspace1Elems,
                                          ChunkSize, 1, ChunkSize, ChunkSize>(
          workspace_1_handle + workspace1_base,
          AccL1Addr, 0, ChunkSize, ChunkSize);
      set_flag(PIPE_MTE2, PIPE_M, EVENT_ID7);
      wait_flag(PIPE_MTE2, PIPE_M, EVENT_ID7);
      linear_attention_pto::gemm_v0<half, float, ChunkSize, HiddenSize,
                                    ChunkSize, ChunkSize, HiddenSize,
                                    ChunkSize, ChunkSize, false, false>(
          acc_l1, v_l1, o_l0, true);
      pipe_barrier(PIPE_M);
      linear_attention_pto::gemm_v0<half, float, ChunkSize, HiddenSize,
                                    HiddenSize, ChunkSize, HiddenSize,
                                    HiddenSize, HiddenSize, false, false>(
          q_l1, h_l1, o_l0, false);
      set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
      pipe_barrier(PIPE_FIX);
      linear_attention_pto::copy_l0c_to_gm<half, float, 1, 1, 1, ChunkSize,
                                           HiddenSize, ChunkElems, ChunkElems,
                                           ChunkElems, HiddenSize, 1, ChunkSize,
                                           HiddenSize>(O_handle + out_base +
                                                           i * ChunkElems,
                                                       OL0Addr,
                                                       0, ChunkSize,
                                                       HiddenSize);
    }
  }
#endif

#if defined(__DAV_C220_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  TEXPANDS(zero_ub, 0.0f);

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

    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    TEXPANDS(hsum_ub, 0.0f);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    linear_attention_pto::copy_ub_to_gm<half, half, 1, 1, 1, HalfHidden,
                                        HiddenSize, HalfHidden * HiddenSize,
                                        HalfHidden * HiddenSize,
                                        HalfHidden * HiddenSize, HiddenSize, 1,
                                        HalfHidden, HiddenSize>(
        workspace_2_handle + workspace2_base, HsumUbAddr, 0, HalfHidden,
        HiddenSize);
    linear_attention_pto::set_cross_flag<PIPE_MTE3>(1, 2);

    for (int64_t i = 0; i < chunk_num; ++i) {
      linear_attention_pto::wait_cross_flag(0);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      linear_attention_pto::copy_gm_to_ub<half, half, 1, 1, 1, HalfChunk,
                                          ChunkSize, HalfChunk * ChunkSize,
                                          HalfChunk * ChunkSize,
                                          HalfChunk * ChunkSize, ChunkSize, 1,
                                          HalfChunk, ChunkSize,
                                          pto::PadValue::Zero>(
          workspace_1_handle + workspace1_base, AccUbAddr, 0, HalfChunk,
          ChunkSize);
      linear_attention_pto::copy_gm_to_ub<half, half, 1, 1, 1, HalfHidden,
                                          HiddenSize, HalfHidden * HiddenSize,
                                          HalfHidden * HiddenSize,
                                          HalfHidden * HiddenSize, HiddenSize, 1,
                                          HalfHidden, HiddenSize,
                                          pto::PadValue::Zero>(
          workspace_2_handle + workspace2_base, HUbAddr, 0, HalfHidden,
          HiddenSize);

      for (int32_t j = 0; j < HalfChunk; ++j) {
        for (int32_t k = 0; k < ChunkSize; ++k) {
          pipe_barrier(PIPE_ALL);
          if ((vid * HalfChunk + j) < k) {
            acc_ub.SetValue(j * ChunkSize + k,
                            zero_ub.GetValue(j * ChunkSize + k));
          }
          pipe_barrier(PIPE_ALL);
          pipe_barrier(PIPE_ALL);
        }
      }

      TADD(hsum_ub, hsum_ub, h_ub);
      linear_attention_pto::copy_ub_to_gm<half, half, 1, 1, 1, HalfChunk,
                                          ChunkSize, HalfChunk * ChunkSize,
                                          HalfChunk * ChunkSize,
                                          HalfChunk * ChunkSize, ChunkSize, 1,
                                          HalfChunk, ChunkSize>(
          workspace_1_handle + workspace1_base, AccUbAddr, 0, HalfChunk,
          ChunkSize);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      linear_attention_pto::copy_ub_to_gm<half, half, 1, 1, 1, HalfHidden,
                                          HiddenSize, HalfHidden * HiddenSize,
                                          HalfHidden * HiddenSize,
                                          HalfHidden * HiddenSize, HiddenSize, 1,
                                          HalfHidden, HiddenSize>(
          workspace_2_handle + workspace2_base, HsumUbAddr, 0, HalfHidden,
          HiddenSize);
      linear_attention_pto::set_cross_flag<PIPE_MTE3>(1, 2);
    }
  }
#endif
}

extern "C" __global__ AICORE void launch_linear_attention(
    __gm__ uint8_t *Q_handle, __gm__ uint8_t *K_handle, __gm__ uint8_t *V_handle,
    __gm__ uint8_t *workspace_1_handle, __gm__ uint8_t *workspace_2_handle,
    __gm__ uint8_t *O_handle, int64_t batch_size, int64_t seq_len,
    uint64_t ffts_addr) {
  main_kernel<LINEAR_ATTN_H, LINEAR_ATTN_D, LINEAR_ATTN_C>(
      reinterpret_cast<__gm__ half *>(Q_handle),
      reinterpret_cast<__gm__ half *>(K_handle),
      reinterpret_cast<__gm__ half *>(V_handle),
      reinterpret_cast<__gm__ half *>(workspace_1_handle),
      reinterpret_cast<__gm__ half *>(workspace_2_handle),
      reinterpret_cast<__gm__ half *>(O_handle), batch_size, seq_len, ffts_addr);
}

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *Q_handle,
                            uint8_t *K_handle, uint8_t *V_handle,
                            uint8_t *workspace_1_handle,
                            uint8_t *workspace_2_handle, uint8_t *O_handle,
                            int64_t batch_size, int64_t seq_len) {
  uint32_t ffts_len = 0;
  uint64_t ffts_addr = 0;
  rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
  launch_linear_attention<<<blockDim, nullptr, stream>>>(
      Q_handle, K_handle, V_handle, workspace_1_handle, workspace_2_handle,
      O_handle, batch_size, seq_len, ffts_addr);
}
