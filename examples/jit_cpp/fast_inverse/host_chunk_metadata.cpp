/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/

#include <algorithm>
#include <cstdint>

extern "C" uint32_t count_varlen_chunks_host_cpp(const int32_t* cu_seqlens,
                                                  uint32_t num_sequences,
                                                  uint32_t chunk_size) {
  uint32_t total_chunks = 0;
  for (uint32_t seq_idx = 0; seq_idx < num_sequences; ++seq_idx) {
    const uint32_t seq_start = static_cast<uint32_t>(cu_seqlens[seq_idx]);
    const uint32_t seq_end = static_cast<uint32_t>(cu_seqlens[seq_idx + 1]);
    const uint32_t seq_len = seq_end - seq_start;
    total_chunks += (seq_len + chunk_size - 1) / chunk_size;
  }
  return total_chunks;
}

extern "C" void build_varlen_chunk_metadata_host_cpp(
    const int32_t* cu_seqlens, uint32_t num_sequences, uint32_t chunk_size,
    int32_t* chunk_indices, int32_t* chunk_valid_sizes) {
  uint32_t chunk_idx = 0;
  for (uint32_t seq_idx = 0; seq_idx < num_sequences; ++seq_idx) {
    const uint32_t seq_start = static_cast<uint32_t>(cu_seqlens[seq_idx]);
    const uint32_t seq_end = static_cast<uint32_t>(cu_seqlens[seq_idx + 1]);
    for (uint32_t row_start = seq_start; row_start < seq_end;
         row_start += chunk_size) {
      const uint32_t valid_size =
          std::min(chunk_size, static_cast<uint32_t>(seq_end - row_start));
      chunk_indices[chunk_idx] = static_cast<int32_t>(row_start);
      chunk_valid_sizes[chunk_idx] = static_cast<int32_t>(valid_size);
      ++chunk_idx;
    }
  }
}

extern "C" void build_chunk_sequence_prefix_host_cpp(
    const int32_t* cu_seqlens, uint32_t num_sequences, uint32_t chunk_size,
    int32_t* chunk_sequence_prefix) {
  chunk_sequence_prefix[0] = static_cast<int32_t>(num_sequences);
  chunk_sequence_prefix[1] = 0;

  uint32_t total_chunks = 0;
  for (uint32_t seq_idx = 0; seq_idx < num_sequences; ++seq_idx) {
    const uint32_t seq_start = static_cast<uint32_t>(cu_seqlens[seq_idx]);
    const uint32_t seq_end = static_cast<uint32_t>(cu_seqlens[seq_idx + 1]);
    const uint32_t seq_len = seq_end - seq_start;
    total_chunks += (seq_len + chunk_size - 1) / chunk_size;
    chunk_sequence_prefix[seq_idx + 2] = static_cast<int32_t>(total_chunks);
  }
}
