#pragma once

#include <pto/pto-inst.hpp>
#include <pto/common/pto_tile.hpp>
#include <runtime/rt_ffts.h>

#include <type_traits>

using namespace pto;

template <typename T, int Rows, int Cols>
using GdnL1Mat = Tile<TileType::Mat, T, Rows, Cols, BLayout::ColMajor, Rows, Cols,
                      SLayout::RowMajor, 512, PadValue::Zero>;

template <typename T, int Rows, int Cols>
using GdnL1MatTrans =
    Tile<TileType::Mat, T, Rows, Cols, BLayout::RowMajor, Rows, Cols,
         SLayout::ColMajor, 512, PadValue::Zero>;

template <typename T, int Rows, int Cols, pto::PadValue PadVal = pto::PadValue::Null>
using GdnUbND = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, Rows, Cols,
                     SLayout::NoneBox, 512, PadVal>;

template <typename T, int Rows, int Cols, pto::PadValue PadVal = pto::PadValue::Null>
using GdnUbDN = Tile<TileType::Vec, T, Rows, Cols, BLayout::ColMajor, Rows, Cols,
                     SLayout::NoneBox, 512, PadVal>;

template <pipe_t Pipe>
AICORE inline void GdnSetCrossFlag(int32_t flag, int32_t mode) {
  const int config = 1 | (mode << 4) | (flag << 8);
  ffts_cross_core_sync(Pipe, config);
}

AICORE inline void GdnWaitCrossFlag(int32_t flag) { wait_flag_dev(flag); }

template <pipe_t Src, pipe_t Dst>
AICORE inline void GdnSetFlag(uint32_t id) {
  set_flag(Src, Dst, static_cast<event_t>(id));
}

template <pipe_t Src, pipe_t Dst>
AICORE inline void GdnWaitFlag(uint32_t id) {
  wait_flag(Src, Dst, static_cast<event_t>(id));
}

template <typename TileData>
AICORE inline void GdnBuildLowerTriMask(TileData &mask_tile, int64_t vector_id,
                                        bool inclusive) {
  constexpr int32_t rows = TileData::Rows;
  constexpr int32_t cols = TileData::Cols;
  const int32_t row_offset = static_cast<int32_t>(vector_id) * rows;
  for (int32_t r = 0; r < rows; ++r) {
    const int32_t global_r = row_offset + r;
    for (int32_t c = 0; c < cols; ++c) {
      const bool keep = inclusive ? (global_r >= c) : (global_r > c);
      mask_tile.SetValue(r * cols + c, keep ? static_cast<half>(1.0f)
                                            : static_cast<half>(0.0f));
    }
  }
}

template <int M, int N, int K, bool TransposeA = false, bool TransposeB = false>
AICORE inline void GdnMatmulL1(
    TileAcc<float, M, N, M, N> &dst,
    std::conditional_t<TransposeA, GdnL1Mat<half, K, M>, GdnL1Mat<half, M, K>> &a_l1,
    std::conditional_t<TransposeB, GdnL1Mat<half, N, K>, GdnL1Mat<half, K, N>> &b_l1,
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

    GdnSetFlag<PIPE_M, PIPE_MTE1>(0);
    GdnSetFlag<PIPE_M, PIPE_MTE1>(1);

    for (int part = 0; part < Parts; ++part) {
      const int buf = part & 1;
      GdnWaitFlag<PIPE_M, PIPE_MTE1>(buf);

      if constexpr (TransposeA) {
        GdnL1MatTrans<half, M, K> a_view;
        TRESHAPE(a_view, a_l1);
        TEXTRACT(a_l0[buf], a_view, 0, part * KStep);
      } else {
        TEXTRACT(a_l0[buf], a_l1, 0, part * KStep);
      }

      if constexpr (TransposeB) {
        GdnL1MatTrans<half, K, N> b_view;
        TRESHAPE(b_view, b_l1);
        TEXTRACT(b_l0[buf], b_view, part * KStep, 0);
      } else {
        TEXTRACT(b_l0[buf], b_l1, part * KStep, 0);
      }

      GdnSetFlag<PIPE_MTE1, PIPE_M>(buf);
      GdnWaitFlag<PIPE_MTE1, PIPE_M>(buf);

      if (init && part == 0) {
        TMATMUL(dst, a_l0[buf], b_l0[buf]);
      } else {
        TMATMUL_ACC(dst, dst, a_l0[buf], b_l0[buf]);
      }

      GdnSetFlag<PIPE_M, PIPE_MTE1>(buf);
    }

    GdnWaitFlag<PIPE_M, PIPE_MTE1>(0);
    GdnWaitFlag<PIPE_M, PIPE_MTE1>(1);
    pipe_barrier(PIPE_ALL);
  } else {
    TileLeft<half, M, K, M, K> a_l0;
    TileRight<half, K, N, K, N> b_l0;
    TASSIGN(a_l0, 0x0);
    TASSIGN(b_l0, 0x0);

    if constexpr (TransposeA) {
      GdnL1MatTrans<half, M, K> a_view;
      TRESHAPE(a_view, a_l1);
      TEXTRACT(a_l0, a_view, 0, 0);
    } else {
      TEXTRACT(a_l0, a_l1, 0, 0);
    }

    if constexpr (TransposeB) {
      GdnL1MatTrans<half, K, N> b_view;
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
