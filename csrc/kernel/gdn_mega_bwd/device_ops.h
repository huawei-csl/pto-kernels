// Copyright (c) 2026 Huawei Technologies Co., Ltd. See repository LICENSE.
#pragma once
#include "layout.h"
#include "primitives.h"

#ifdef __CCE_AICORE__
// Both vector subblocks participate in the handshake, and both
// participate in arithmetic where rows are independent. Each block owns its
// scratch, with no cross-block waits.
AICORE inline void join() {
  sync();
#if defined(__DAV_CUBE__)
  // Release both vectors only after both have completed the previous work.
  // Signalling first would allow one vector to read the other's unfinished
  // rows.
  kernel_utils::WaitCubeVecFlag(1);
  kernel_utils::SetCubeVecFlag<PIPE_FIX>(0);
#elif defined(__DAV_VEC__)
  kernel_utils::SetCubeVecFlag<PIPE_MTE3>(1);
  kernel_utils::WaitCubeVecFlag(0);
#endif
  sync();
}
// Explicit Cube/Vec dependency boundary; no operation dispatch or allocation.
AICORE inline void phase_end(int& epoch, bool independent_stripes = false) {
  if (independent_stripes) {
    sync();
#if defined(__DAV_CUBE__)
    kernel_utils::SetCubeVecFlag<PIPE_FIX>(2 + epoch);
    kernel_utils::WaitCubeVecFlag(4 + epoch);
#elif defined(__DAV_VEC__)
    kernel_utils::SetCubeVecFlag<PIPE_MTE3>(4 + epoch);
    kernel_utils::WaitCubeVecFlag(2 + epoch);
#endif
    sync();
    epoch ^= 1;  // A fast subcore cannot signal twice into one epoch.
  } else
    join();
}
AICORE inline __gm__ uint8_t* inverse_data(const Plan& p, int c, int h) {
  return (__gm__ uint8_t*)p.inverse_cache +
         (int64_t(c) * p.heads + h) * Matrix * (p.inverse_dtype == 2 ? 4 : 2);
}
struct Buffer {
  __gm__ float* data = nullptr;
  bool vector = false;
};
AICORE inline bool shared_grams(const Plan& p) {
  return p.chunks * p.key_heads >= BlockCount && p.input[0].dtype != 2 &&
         p.input[1].dtype != 2 && p.input[2].dtype != 2;
}
struct Chunk {
  __gm__ float* base;
  __gm__ float* keys;
  bool shared_q = false;
  AICORE Buffer m(int slot) const {
    return {
        (slot == K || slot == KK || (slot == Q_DQ && shared_q) ? keys : base) +
        slot * Matrix};
  }
  AICORE Buffer v(int slot) const {
    return {base + MatrixSlots * Matrix + slot * 128, true};
  }
};
AICORE inline Chunk chunk(const Plan& p, int c, int h) {
  int kh = shared_grams(p)
               ? h / (p.heads / p.key_heads) * (p.heads / p.key_heads)
               : h;
  auto start = (__gm__ float*)p.workspace + int64_t(c) * p.heads * RecordSize;
  return {start + h * RecordSize, start + kh * RecordSize, fused_output(p)};
}
AICORE inline int sequence(const Plan& p, int c) {
  int s = 0;
  while (c >= p.offset[s + 1]) ++s;
  return s;
}
AICORE inline __gm__ float* scratch(const Plan& p) {
  return (__gm__ float*)p.workspace + int64_t(p.chunks) * p.heads * RecordSize +
         get_block_idx() * ScratchSize;
}
#endif
