/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/

#pragma once

// Synchronization, packed tile I/O, and variable-length chunk indexing.
#ifdef __CCE_AICORE__

constexpr uint16_t GDN_SYNC_AIV_FLAG = 12;
constexpr uint16_t GDN_SYNC_AIC_FLAG = 11;
constexpr uint16_t GDN_SYNC_AIC_AIV_FLAG = 13;
constexpr uint16_t SYNC_MODE_SHIFT_VALUE = 4;
constexpr uint16_t SYNC_FLAG_SHIFT_VALUE = 8;

// A2/A3 vs A5 compatibility helpers. Mirrors csrc/kernel/kernel_utils.h,
// duplicated here because that header would clash with the local GetffstMsg
// and SYNC_* definitions below.
constexpr uint16_t GDN_VEC_FLAG_OFFSET = 16;

/// The vector pipe barrier is required on A2/A3; A5's vector pipe handles
/// the dependency itself, so this compiles to nothing there.
AICORE inline void PipeBarrierVec() {
#if __CCE_AICORE__ == 220
  pipe_barrier(PIPE_V);
#endif
}

/// A5 has two Vec sub-blocks whose flag ids differ by 16, so one logical
/// event signalled by Cube has two waiters.
template <pipe_t Pipe>
AICORE inline void GdnSignalBothVec(uint16_t flag) {
  set_intra_block(Pipe, flag);
  set_intra_block(Pipe, flag + GDN_VEC_FLAG_OFFSET);
}

template <pipe_t Pipe>
AICORE inline void GdnWaitBothVec(uint16_t flag) {
  wait_intra_block(Pipe, flag);
  wait_intra_block(Pipe, flag + GDN_VEC_FLAG_OFFSET);
}

AICORE inline uint16_t GetffstMsg(uint16_t mode, uint16_t flagId) {
  return (0x1 + ((mode & 0x3) << SYNC_MODE_SHIFT_VALUE) +
          ((flagId & 0xf) << SYNC_FLAG_SHIFT_VALUE));
}

// Preserve the producer pipe on A2/A3. A5 intra-block events require
// explicit completion ordering and a flag for each Vec sub-block.
template <pipe_t Pipe>
AICORE inline void mega_signal(uint16_t flag) {
#if __CCE_AICORE__ == 220
  ffts_cross_core_sync(Pipe, GetffstMsg(0x2, flag));
#else
  pipe_barrier(PIPE_ALL);
#if defined(__DAV_CUBE__)
  GdnSignalBothVec<PIPE_FIX>(flag);
#elif defined(__DAV_VEC__)
  set_intra_block(Pipe, flag);
#endif
#endif
}

AICORE inline void mega_wait(uint16_t flag) {
#if __CCE_AICORE__ == 220
  wait_flag_dev(flag);
#else
#if defined(__DAV_CUBE__)
  GdnWaitBothVec<PIPE_MTE2>(flag);
#elif defined(__DAV_VEC__)
  wait_intra_block(PIPE_MTE3, flag);
#endif
  pipe_barrier(PIPE_ALL);
#endif
}

AICORE inline void mega_sync_all() {
  pipe_barrier(PIPE_ALL);
#if __CCE_AICORE__ == 220
#if defined(__DAV_CUBE__)
  wait_flag_dev(GDN_SYNC_AIV_FLAG);
  ffts_cross_core_sync(PIPE_FIX, GetffstMsg(0x0, GDN_SYNC_AIC_FLAG));
  wait_flag_dev(GDN_SYNC_AIC_FLAG);
  ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, GDN_SYNC_AIC_AIV_FLAG));
#elif defined(__DAV_VEC__)
  ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, GDN_SYNC_AIV_FLAG));
  wait_flag_dev(GDN_SYNC_AIC_AIV_FLAG);
#endif
#else
  // Join both Vec sub-blocks before the device-wide Cube barrier, then
  // release both only after every block has finished the shared products.
#if defined(__DAV_CUBE__)
  GdnWaitBothVec<PIPE_S>(GDN_SYNC_AIV_FLAG);
  ffts_cross_core_sync(PIPE_FIX, GetffstMsg(0x0, GDN_SYNC_AIC_FLAG));
  wait_flag_dev(PIPE_S, GDN_SYNC_AIC_FLAG);
  GdnSignalBothVec<PIPE_S>(GDN_SYNC_AIC_AIV_FLAG);
#elif defined(__DAV_VEC__)
  set_intra_block(PIPE_MTE3, GDN_SYNC_AIV_FLAG);
  wait_intra_block(PIPE_S, GDN_SYNC_AIC_AIV_FLAG);
#endif
  pipe_barrier(PIPE_ALL);
#endif
}

#endif

/**
 * @brief Struct containing starting address and size of a single tile
 */
struct BSNDVarlenTileInfo {
  uint32_t bsnd_offset; /**< Contains the starting index in the global tensor */
  uint32_t valid_size;  /**< This is the size (num_rows/cols) of the tile */
};

/*
 * For cu_seqlens-based varlen BSND, tile_id still enumerates chunk-major then
 * head-major. We recover the owning sequence by scanning cu_seqlens and
 * counting chunks per sequence.
 */
AICORE inline BSNDVarlenTileInfo mega_chunk_info(uint32_t tile_id,
                                                 uint32_t num_bsnd_heads,
                                                 uint32_t matrix_size,
                                                 __gm__ int32_t* cu_seqlens) {
  const uint32_t head_idx = tile_id % num_bsnd_heads;
  const uint32_t chunk_idx = tile_id / num_bsnd_heads;

  uint32_t seq_start = static_cast<uint32_t>(cu_seqlens[0]);
  uint32_t accumulated_chunks = 0;
  for (uint32_t seq_idx = 0;; ++seq_idx) {
    const uint32_t seq_end = static_cast<uint32_t>(cu_seqlens[seq_idx + 1]);
    const uint32_t seq_len = seq_end - seq_start;
    const uint32_t seq_num_chunks = ((seq_len + matrix_size - 1) / matrix_size);
    if (chunk_idx < accumulated_chunks + seq_num_chunks) {
      const uint32_t local_chunk_idx = chunk_idx - accumulated_chunks;
      const uint32_t row_start = seq_start + local_chunk_idx * matrix_size;
      const uint32_t valid_size =
          min(static_cast<uint32_t>(seq_end - row_start), matrix_size);
      return {row_start * num_bsnd_heads * matrix_size + head_idx * matrix_size,
              valid_size};
    }
    accumulated_chunks += seq_num_chunks;
    seq_start = seq_end;
  }
}

// Static full-tile descriptors avoid dynamic DMA setup on the common path.
template <int R, int C, int Stride, typename Full, typename Tail>
AICORE inline void mega_load_packed(Full& full, Tail& tail,
                                    __gm__ half* pointer, int rows, int cols) {
  if (rows == R && cols == C) {
    GlobalTensor<half, Shape<1, 1, 1, R, C>, pto::Stride<1, 1, 1, Stride, 1>>
        gm(pointer);
    TLOAD(full, gm);
  } else {
    GlobalTensor<half, Shape<1, 1, 1, DYNAMIC, DYNAMIC>,
                 pto::Stride<1, 1, 1, Stride, 1>>
        gm(pointer, {1, 1, 1, rows, cols});
    TLOAD(tail, gm);
  }
}

template <int R, int C, int Stride, typename Full, typename Tail>
AICORE inline void mega_store_packed(Full& full, Tail& tail,
                                     __gm__ half* pointer, int rows, int cols) {
  if (rows == R && cols == C) {
    GlobalTensor<half, Shape<1, 1, 1, R, C>, pto::Stride<1, 1, 1, Stride, 1>>
        gm(pointer);
    TSTORE(gm, full);
  } else {
    GlobalTensor<half, Shape<1, 1, 1, DYNAMIC, DYNAMIC>,
                 pto::Stride<1, 1, 1, Stride, 1>>
        gm(pointer, {1, 1, 1, rows, cols});
    TSTORE(gm, tail);
  }
}
