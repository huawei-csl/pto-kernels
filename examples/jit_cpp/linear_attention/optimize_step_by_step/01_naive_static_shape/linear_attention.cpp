#include <pto/pto-inst.hpp>
#include <runtime/rt_ffts.h>
#include <type_traits>

using namespace pto;

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

AICORE void main_kernel(__gm__ half *Q_handle, __gm__ half *K_handle,
                        __gm__ half *V_handle, __gm__ half *workspace_1_handle,
                        __gm__ half *workspace_2_handle, __gm__ half *O_handle,
                        uint64_t ffts_addr) {
  auto cid = get_block_idx();
  set_ffts_base_addr(ffts_addr);

  linear_attention_pto::TileMatL1<half, 64, 128, 64, 128> q_l1;
  TASSIGN(q_l1, 0);
  linear_attention_pto::TileMatL1<half, 64, 128, 64, 128> k_l1;
  TASSIGN(k_l1, 16384);
  linear_attention_pto::TileMatL1<half, 64, 128, 64, 128> v_l1;
  TASSIGN(v_l1, 32768);
  linear_attention_pto::TileMatL1<half, 128, 128, 128, 128> h_l1;
  TASSIGN(h_l1, 49152);
  TileAcc<float, 64, 64, 64, 64> acc_l0;
  TASSIGN(acc_l0, 0);
  TileAcc<float, 128, 128, 128, 128> h_l0;
  TASSIGN(h_l0, 16384);
  linear_attention_pto::TileMatL1<half, 64, 64, 64, 64> acc_l1;
  TASSIGN(acc_l1, 81920);
  TileAcc<float, 64, 128, 64, 128> o_l0;
  TASSIGN(o_l0, 81920);
  linear_attention_pto::TileUbDataND<half, 64, 128, 64, 128> hsum_ub;
  TASSIGN(hsum_ub, 0);
  linear_attention_pto::TileUbDataND<half, 32, 64, 32, 64> zero_ub;
  TASSIGN(zero_ub, 16384);
  linear_attention_pto::TileUbDataND<half, 32, 64, 32, 64> acc_ub;
  TASSIGN(acc_ub, 20480);
  linear_attention_pto::TileUbDataND<half, 64, 128, 64, 128> h_ub;
  TASSIGN(h_ub, 24576);
  auto vid = get_subblockid();

#if defined(__DAV_C220_CUBE__)
  for (int32_t i = 0; i < 8; ++i) {
    set_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID1);
    linear_attention_pto::copy_gm_to_l1<half, half, 1, 1, 1, 64, 128, 262144,
                                        131072, 65536, 128, 1, 64, 128>(
        Q_handle + ((cid * 65536) + (i * 8192)), 0, 0, 64, 128);
    linear_attention_pto::copy_gm_to_l1<half, half, 1, 1, 1, 64, 128, 262144,
                                        131072, 65536, 128, 1, 64, 128>(
        K_handle + ((cid * 65536) + (i * 8192)), 16384, 0, 64, 128);
    linear_attention_pto::copy_gm_to_l1<half, half, 1, 1, 1, 64, 128, 262144,
                                        131072, 65536, 128, 1, 64, 128>(
        V_handle + ((cid * 65536) + (i * 8192)), 32768, 0, 64, 128);
    linear_attention_pto::copy_gm_to_l1<half, half, 1, 1, 1, 128, 128, 65536,
                                        32768, 16384, 128, 1, 128, 128>(
        workspace_2_handle + (cid * 16384), 49152, 0, 128, 128);
    set_flag(PIPE_MTE2, PIPE_M, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_M, EVENT_ID1);
    linear_attention_pto::gemm_v0<half, float, 64, 64, 128, 64, 64, 128, 128,
                                  false, true>(q_l1, k_l1, acc_l0, true);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID2);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID2);
    linear_attention_pto::copy_l0c_to_gm<half, float, 1, 1, 1, 64, 64, 16384,
                                         8192, 4096, 64, 1, 64, 64>(
        workspace_1_handle + (cid * 4096), 0, 0, 64, 64);
    set_flag(PIPE_MTE2, PIPE_M, EVENT_ID3);
    wait_flag(PIPE_MTE2, PIPE_M, EVENT_ID3);
    pipe_barrier(PIPE_M);
    linear_attention_pto::gemm_v0<half, float, 128, 128, 64, 128, 128, 64, 64,
                                  true, false>(k_l1, v_l1, h_l0, true);
    set_flag(PIPE_MTE2, PIPE_FIX, EVENT_ID4);
    wait_flag(PIPE_MTE2, PIPE_FIX, EVENT_ID4);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID5);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID5);
    linear_attention_pto::copy_l0c_to_gm<half, float, 1, 1, 1, 128, 128, 65536,
                                         32768, 16384, 128, 1, 128, 128>(
        workspace_2_handle + (cid * 16384), 16384, 0, 128, 128);
    linear_attention_pto::set_cross_flag<PIPE_FIX>(0, 2);
    linear_attention_pto::wait_cross_flag(1);
    set_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID6);
    wait_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID6);
    linear_attention_pto::copy_gm_to_l1<half, half, 1, 1, 1, 64, 64, 16384,
                                        8192, 4096, 64, 1, 64, 64>(
        workspace_1_handle + (cid * 4096), 81920, 0, 64, 64);
    set_flag(PIPE_MTE2, PIPE_M, EVENT_ID7);
    wait_flag(PIPE_MTE2, PIPE_M, EVENT_ID7);
    linear_attention_pto::gemm_v0<half, float, 64, 128, 64, 64, 128, 64, 64,
                                  false, false>(acc_l1, v_l1, o_l0, true);
    pipe_barrier(PIPE_M);
    linear_attention_pto::gemm_v0<half, float, 64, 128, 128, 64, 128, 128, 128,
                                  false, false>(q_l1, h_l1, o_l0, false);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    linear_attention_pto::copy_l0c_to_gm<half, float, 1, 1, 1, 64, 128, 262144,
                                         131072, 65536, 128, 1, 64, 128>(
        O_handle + ((cid * 65536) + (i * 8192)), 81920, 0, 64, 128);
  }
#endif

#if defined(__DAV_C220_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  TEXPANDS(hsum_ub, 0.0f);
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  TEXPANDS(zero_ub, 0.0f);

  for (int32_t i = 0; i < 8; ++i) {
    linear_attention_pto::wait_cross_flag(0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    linear_attention_pto::copy_gm_to_ub<half, half, 1, 1, 1, 32, 64, 16384,
                                        8192, 4096, 64, 1, 32, 64,
                                        pto::PadValue::Zero>(
        workspace_1_handle + ((cid * 4096) + (vid * 2048)), 20480, 0, 32, 64);
    linear_attention_pto::copy_gm_to_ub<half, half, 1, 1, 1, 64, 128, 65536,
                                        32768, 16384, 128, 1, 64, 128,
                                        pto::PadValue::Zero>(
        workspace_2_handle + ((cid * 16384) + (vid * 8192)), 24576, 0, 64,
        128);

    for (int32_t j = 0; j < 32; ++j) {
      for (int32_t k = 0; k < 64; ++k) {
        pipe_barrier(PIPE_ALL);
        if (((vid * 32) + j) < k) {
          acc_ub.SetValue((j * 64) + k, zero_ub.GetValue((j * 64) + k));
        }
        pipe_barrier(PIPE_ALL);
        pipe_barrier(PIPE_ALL);
      }
    }

    TADD(hsum_ub, hsum_ub, h_ub);
    linear_attention_pto::copy_ub_to_gm<half, half, 1, 1, 1, 32, 64, 16384,
                                        8192, 4096, 64, 1, 32, 64>(
        workspace_1_handle + ((cid * 4096) + (vid * 2048)), 20480, 0, 32, 64);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    linear_attention_pto::copy_ub_to_gm<half, half, 1, 1, 1, 64, 128, 65536,
                                        32768, 16384, 128, 1, 64, 128>(
        workspace_2_handle + ((cid * 16384) + (vid * 8192)), 0, 0, 64, 128);
    linear_attention_pto::set_cross_flag<PIPE_MTE3>(1, 2);
  }
#endif
}

extern "C" __global__ AICORE void launch_linear_attention(
    __gm__ uint8_t *Q_handle, __gm__ uint8_t *K_handle, __gm__ uint8_t *V_handle,
    __gm__ uint8_t *workspace_1_handle, __gm__ uint8_t *workspace_2_handle,
    __gm__ uint8_t *O_handle, uint64_t ffts_addr) {
  main_kernel(reinterpret_cast<__gm__ half *>(Q_handle),
              reinterpret_cast<__gm__ half *>(K_handle),
              reinterpret_cast<__gm__ half *>(V_handle),
              reinterpret_cast<__gm__ half *>(workspace_1_handle),
              reinterpret_cast<__gm__ half *>(workspace_2_handle),
              reinterpret_cast<__gm__ half *>(O_handle), ffts_addr);
}

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *Q_handle,
                            uint8_t *K_handle, uint8_t *V_handle,
                            uint8_t *workspace_1_handle,
                            uint8_t *workspace_2_handle, uint8_t *O_handle) {
  uint32_t ffts_len = 0;
  uint64_t ffts_addr = 0;
  rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
  launch_linear_attention<<<blockDim, nullptr, stream>>>(
      Q_handle, K_handle, V_handle, workspace_1_handle, workspace_2_handle,
      O_handle, ffts_addr);
}
