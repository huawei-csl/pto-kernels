// Copyright (c) 2026 Huawei Technologies Co., Ltd. See repository LICENSE.
#pragma once
#include "primitives.h"

// Seven 64-KiB L1 slots retain native FP32 operands across one chunk's
// adjoints. Products finish at the caller's cross-core phase boundary. Loads to
// disjoint slots can follow cube_product() without a full barrier and overlap
// M/FIX work.
template <int Slot>
AICORE inline void cube_load(__gm__ float* input) {
#if defined(__DAV_CUBE__)
  Tile<TileType::Mat, float, 128, 128, BLayout::ColMajor, 128, 128,
       SLayout::RowMajor, 512>
      value;
  TASSIGN(value, Slot * 65536);
  GlobalTensor<float, Shape<1, 1, 1, 128, 128>, pto::Stride<1, 1, 1, 128, 1>>
      gm(input);
  TLOAD(value, gm);
#endif
}
template <int A, int B, bool TA = false, bool TB = false,
          bool Symmetric = false>
AICORE inline void cube_product(__gm__ float* output) {
#if defined(__DAV_CUBE__)
  using Mat = Tile<TileType::Mat, float, 128, 128, BLayout::ColMajor, 128, 128,
                   SLayout::RowMajor, 512>;
  using Trans = Tile<TileType::Mat, float, 128, 128, BLayout::RowMajor, 128,
                     128, SLayout::ColMajor, 512>;
  Mat a, b;
  Trans at, bt;
  TileLeft<float, 128, 128> left;
  TileRight<float, 128, 128> right;
  TileAcc<float, 128, 128> acc;
  TASSIGN(a, A * 65536);
  TASSIGN(b, B * 65536);
  TASSIGN(left, 0);
  TASSIGN(right, 0);
  TASSIGN(acc, 0);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  if constexpr (TA) {
    TRESHAPE(at, a);
    TEXTRACT(left, at, 0, 0);
  } else
    TEXTRACT(left, a, 0, 0);
  if constexpr (TB) {
    TRESHAPE(bt, b);
    TEXTRACT(right, bt, 0, 0);
  } else
    TEXTRACT(right, b, 0, 0);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
  TMATMUL(acc, left, right);
  if constexpr (Symmetric) {
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    TRESHAPE(at, a);
    TEXTRACT(left, at, 0, 0);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    TMATMUL_ACC(acc, acc, left, right);
  }
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  GlobalTensor<float, Shape<1, 1, 1, 128, 128>, pto::Stride<1, 1, 1, 128, 1>>
      gm(output);
  TSTORE(gm, acc);
#endif
}

// Q/K already have 16-bit storage: read them directly into L1 for shared Gram
// products. No quantization or extra GM packing is introduced by this path.
template <class T>
AICORE inline void input_grams(const Input& q, const Input& k, int first,
                               int head, __gm__ float* kk, __gm__ float* qk) {
#if defined(__DAV_CUBE__)
  using Mat = Tile<TileType::Mat, T, 128, 128, BLayout::ColMajor, 128, 128,
                   SLayout::RowMajor, 512>;
  using Trans = Tile<TileType::Mat, T, 128, 128, BLayout::RowMajor, 128, 128,
                     SLayout::ColMajor, 512>;
  Mat qm, km;
  Trans kt;
  TileLeft<T, 128, 128> left;
  TileRight<T, 128, 128> right;
  TileAcc<float, 128, 128> acc;
  TASSIGN(qm, 0);
  TASSIGN(km, 32768);
  TASSIGN(left, 0);
  TASSIGN(right, 0);
  TASSIGN(acc, 0);
  GlobalTensor<T, Shape<1, 1, 1, 128, 128>, pto::Stride<1, 1, 1, DYNAMIC, 1>>
      gq((__gm__ T*)q.data + first * q.stride[1] + head * q.stride[2],
         {1, 1, 1, 128, 128}, {1, 1, 1, q.stride[1], 1});
  GlobalTensor<T, Shape<1, 1, 1, 128, 128>, pto::Stride<1, 1, 1, DYNAMIC, 1>>
      gk((__gm__ T*)k.data + first * k.stride[1] + head * k.stride[2],
         {1, 1, 1, 128, 128}, {1, 1, 1, k.stride[1], 1});
  TLOAD(qm, gq);
  TLOAD(km, gk);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TRESHAPE(kt, km);
  TEXTRACT(left, km, 0, 0);
  TEXTRACT(right, kt, 0, 0);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  TMATMUL(acc, left, right);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  GlobalTensor<float, Shape<1, 1, 1, 128, 128>, pto::Stride<1, 1, 1, 128, 1>>
      out_kk(kk), out_qk(qk);
  TSTORE(out_kk, acc);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  TEXTRACT(left, qm, 0, 0);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
  TMATMUL(acc, left, right);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TSTORE(out_qk, acc);
#endif
}
