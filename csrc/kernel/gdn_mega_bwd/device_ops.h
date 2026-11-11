// Copyright (c) 2026 Huawei Technologies Co., Ltd. See repository LICENSE.
#pragma once
#include "layout.h"
#include "primitives.h"

#ifdef __CCE_AICORE__
struct Batch {
  __gm__ const SequenceMetadata* metadata;
  int sequence_base, sequences, chunks, chunk_base, tokens;
  bool shared_grams;
  AICORE int start(int s) const { return metadata ? metadata->start[s] : 0; }
  AICORE int length(int s) const {
    return metadata ? metadata->length[s] : tokens;
  }
  AICORE int offset(int s) const {
    return metadata ? metadata->offset[s] : s * chunks;
  }
};
AICORE inline Batch sequence_batch(const KernelArgs& p, int first) {
  auto meta =
      p.metadata
          ? (__gm__ const SequenceMetadata*)(p.metadata +
                                             int64_t(first / MaxSequences) *
                                                 MetadataWords *
                                                 sizeof(int32_t))
          : nullptr;
  int count = min(MaxSequences, p.total_sequences - first);
  int chunks = meta ? meta->offset[count] : int((p.tokens + 127LL) / 128);
  return {meta,
          first,
          count,
          chunks,
          meta ? meta->chunk_base : 0,
          p.tokens,
          chunks * p.key_heads >= BlockCount && !p.accurate};
}
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
AICORE inline __gm__ uint8_t* inverse_data(const KernelArgs& p,
                                           const Batch& batch, int c, int h) {
  return (__gm__ uint8_t*)p.inverse_cache +
         ((int64_t(batch.chunk_base) + c) * p.heads + h) * Matrix *
             (p.inverse_dtype == 2 ? 4 : 2);
}
struct Buffer {
  __gm__ float* data = nullptr;
  bool vector = false;
};
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
AICORE inline Chunk chunk(const KernelArgs& p, const Batch& batch, int c,
                          int h) {
  int kh = batch.shared_grams ? h / p.head_group * p.head_group : h;
  auto start = (__gm__ float*)p.workspace + int64_t(c) * p.heads * RecordSize;
  return {start + h * RecordSize, start + kh * RecordSize, p.fused_output};
}
struct TokenRange {
  int first, length;
};
// Grid-stride tasks visit chunks in order. Each core scans each boundary at
// most once per stage, including empty sequences, and reuses bounds across
// heads.
struct SequenceCursor {
  int sequence = -1, begin = 0, end = 0, first = 0, limit = 0;
  AICORE TokenRange locate(const Batch& batch, int c) {
    if (c >= end) {
      do {
        ++sequence;
        begin = end;
        end = batch.offset(sequence + 1);
      } while (c >= end);
      first = batch.start(sequence);
      limit = first + batch.length(sequence);
    }
    int token = first + (c - begin) * 128;
    return {token, min(128, limit - token)};
  }
};
AICORE inline __gm__ float* scratch(const KernelArgs& p) {
  return (__gm__ float*)p.workspace +
         int64_t(p.capacity) * p.heads * RecordSize +
         get_block_idx() * ScratchSize;
}
#endif
