#pragma once

#include <cstdint>

struct GdnSeqInfo {
  uint32_t bos;
  uint32_t seq_len;
  uint32_t chunk_offset;
};

struct GdnBsndSeqInfo {
  uint32_t bos;
  uint32_t seq_len;
  uint32_t chunk_offset;
  uint32_t token_base_offset;
  uint32_t row_stride;
};

AICORE inline uint32_t GdnDivCeilU32(uint32_t x, uint32_t y) {
  return (x + y - 1) / y;
}

AICORE inline GdnSeqInfo GetGdnSeqInfo(uint32_t seq_idx, uint32_t chunk_size,
                                       uint32_t fixed_seq_len,
                                       __gm__ int32_t *cu_seqlens) {
  if (cu_seqlens == nullptr) {
    const uint32_t bos = seq_idx * fixed_seq_len;
    const uint32_t chunk_offset = seq_idx * GdnDivCeilU32(fixed_seq_len, chunk_size);
    return {bos, fixed_seq_len, chunk_offset};
  }

  uint32_t chunk_offset = 0;
  for (uint32_t i = 0; i < seq_idx; ++i) {
    const uint32_t seq_start = static_cast<uint32_t>(cu_seqlens[i]);
    const uint32_t seq_end = static_cast<uint32_t>(cu_seqlens[i + 1]);
    chunk_offset += GdnDivCeilU32(seq_end - seq_start, chunk_size);
  }
  const uint32_t bos = static_cast<uint32_t>(cu_seqlens[seq_idx]);
  const uint32_t eos = static_cast<uint32_t>(cu_seqlens[seq_idx + 1]);
  return {bos, eos - bos, chunk_offset};
}

AICORE inline GdnBsndSeqInfo GetGdnBsndSeqInfo(uint32_t seq_idx,
                                               uint32_t head_idx,
                                               uint32_t num_heads,
                                               uint32_t hidden_size,
                                               uint32_t chunk_size,
                                               uint32_t fixed_seq_len,
                                               __gm__ int32_t *cu_seqlens) {
  if (cu_seqlens == nullptr) {
    const uint32_t bos = seq_idx * fixed_seq_len;
    const uint32_t chunk_num = GdnDivCeilU32(fixed_seq_len, chunk_size);
    return {
        bos,
        fixed_seq_len,
        seq_idx * chunk_num,
        bos * num_heads * hidden_size + head_idx * hidden_size,
        num_heads * hidden_size,
    };
  }

  uint32_t chunk_offset = 0;
  for (uint32_t i = 0; i < seq_idx; ++i) {
    const uint32_t seq_start = static_cast<uint32_t>(cu_seqlens[i]);
    const uint32_t seq_end = static_cast<uint32_t>(cu_seqlens[i + 1]);
    chunk_offset += GdnDivCeilU32(seq_end - seq_start, chunk_size);
  }
  const uint32_t bos = static_cast<uint32_t>(cu_seqlens[seq_idx]);
  const uint32_t eos = static_cast<uint32_t>(cu_seqlens[seq_idx + 1]);
  return {
      bos,
      eos - bos,
      chunk_offset,
      bos * num_heads * hidden_size + head_idx * hidden_size,
      num_heads * hidden_size,
  };
}
