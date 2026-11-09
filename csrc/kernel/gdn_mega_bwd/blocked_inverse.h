// Copyright (c) 2026 Huawei Technologies Co., Ltd. See repository LICENSE.
#pragma once
#include "../gdn_mega/inverse.h"
#include "../gdn_mega/recurrence_pipeline.h"

#ifdef __CCE_AICORE__
// This gather table survives all stage-0 operations; initialize once per block.
AICORE inline void inverse_indices() {
#if defined(__DAV_VEC__)
  Tile<TileType::Vec, uint32_t, 1, 1024> indices;
  TASSIGN(indices, 176 * 1024);
  // Match forward: A2 vgather uses bytes; A5 TGATHER uses element indices.
#if __CCE_AICORE__ == 220
  constexpr int gather_scale = sizeof(float);
#else
  constexpr int gather_scale = 1;
#endif
  for (int j = 0; j < 16; ++j)
    for (int col = 0; col < 64; ++col)
      indices.SetValue(j * 64 + col, (((col / 16) * 16 + j) * 128 +
                                      get_subblockid() * 64 + (col / 16) * 16) *
                                         gather_scale);
  sync();
#endif
}

// Invert eight 16x16 diagonal blocks in parallel, then join 32/64/128 blocks
// with Cube products. Reuse forward's stable substitution, not a power series.
AICORE inline void blocked_inverse(__gm__ float* factor, __gm__ float* scratch,
                                   __gm__ float* at) {
#if defined(__DAV_VEC__)
  Half a, identity, lower;
  TASSIGN(a, 96 * 1024);
  TASSIGN(identity, 0);
  TASSIGN(lower, 32768);
  int stripe = get_subblockid();
  load<64, 128>(a, factor + stripe * 8192);
  TMULS(a, a, -1.0f);
  sync();
  causal_mask(identity, stripe, false, 66048);
  causal_mask(lower, stripe, true, 66048);
  TSUB(identity, identity, lower);
  sync();
  mega_diagonal_inverses(a, identity, stripe * 64);
  store<64, 128>(a, factor + stripe * 8192);
#endif
  join();
#if defined(__DAV_CUBE__)
  MegaFloat<128, 128> a;
  TASSIGN(a, 0);
  load<128, 128>(a, factor);
  mega_float_join_level<16, 128>(a, scratch, {(__gm__ uint8_t*)factor, 2});
  mega_float_join_level<32, 128>(a, scratch, {(__gm__ uint8_t*)factor, 2});
  mega_float_join_level<64, 128>(a, scratch, {(__gm__ uint8_t*)factor, 2});
#endif
  join();
  transpose_inverse(factor, at);
  join();
}
#endif
