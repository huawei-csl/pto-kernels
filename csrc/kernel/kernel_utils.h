/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/
#pragma once

#include <pto/pto-inst.hpp>
#include <type_traits>

// clang-format off: so it does not get wrongfully flagged by linter
#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t*  // To avoid #include "kernel_operator.h"
#endif
// clang-format on

namespace kernel_utils {

/// True when this translation unit is compiled for a Vector (AIV) core.
#if defined(__DAV_VEC__)
constexpr bool IS_AIV = true;
constexpr bool IS_AIC = false;
#elif defined(__DAV_CUBE__)
constexpr bool IS_AIV = false;
constexpr bool IS_AIC = true;
#else
constexpr bool IS_AIV = false;
constexpr bool IS_AIC = false;
#endif
/**
 * @brief Do a sync step (set-wait flag) between two pipes.
 *
 * @tparam SrcPipe The pipe that sets the flag.
 * @tparam DstPipe The pipe that waits for the flag.
 * @param [in] id The event id to sync for.
 */
template <pipe_t SrcPipe, pipe_t DstPipe>
AICORE inline void SetWaitFlag(uint32_t id) {
  set_flag(SrcPipe, DstPipe, static_cast<event_t>(id));
  wait_flag(SrcPipe, DstPipe, static_cast<event_t>(id));
}

/**
 * @brief Performs a division on two integral numbers and rounds the result up
 * to the nearest integer.
 *
 * @tparam T1 Data type of dividend.
 * @tparam T2 Data type of divisor.
 * @param [in] value Dividend.
 * @param [in] divisor Divisor.
 * @return Result of division.
 */
template <typename T1, typename T2,
          typename std::enable_if<std::is_integral<T1>::value &&
                                      std::is_integral<T2>::value,
                                  int>::type = 0>
AICORE inline T1 CeilDiv(T1 value, T2 divisor) {
  return (value + divisor - 1) / divisor;
}

/**
 * @brief Performs a division on two integral numbers and rounds the result down
 * to the nearest integer.
 *
 * @tparam T1 Data type of dividend.
 * @tparam T2 Data type of divisor.
 * @param [in] value Dividend.
 * @param [in] divisor Divisor.
 * @return Result of division.
 */
template <typename T1, typename T2,
          typename std::enable_if<std::is_integral<T1>::value &&
                                      std::is_integral<T2>::value,
                                  int>::type = 0>
__aicore__ inline T1 FloorDiv(T1 value, T2 divisor) {
  return value / divisor;
}

#define BSND_OFFSET(tile_id, N, S, D) \
  (((tile_id) / (N)) * (S) * (N) * (D) + ((tile_id) % (N)) * (D))

/*
 * For aligned BSND, tile_id enumerates chunk-major then head-major and maps to
 * a fixed-stride address inside the dense BSND tensor.
 */
AICORE inline uint32_t GetBSNDFixedTileOffset(uint32_t tile_id,
                                              uint32_t num_bsnd_heads,
                                              uint32_t matrix_size) {
  return BSND_OFFSET(tile_id, num_bsnd_heads, matrix_size, matrix_size);
}

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
AICORE inline BSNDVarlenTileInfo GetBSNDVarlenTileInfoFromCuSeqlens(
    uint32_t tile_id, uint32_t num_bsnd_heads, uint32_t matrix_size,
    __gm__ int32_t* cu_seqlens) {
  const uint32_t head_idx = tile_id % num_bsnd_heads;
  const uint32_t chunk_idx = tile_id / num_bsnd_heads;

  uint32_t seq_start = static_cast<uint32_t>(cu_seqlens[0]);
  uint32_t accumulated_chunks = 0;
  for (uint32_t seq_idx = 0;; ++seq_idx) {
    const uint32_t seq_end = static_cast<uint32_t>(cu_seqlens[seq_idx + 1]);
    const uint32_t seq_len = seq_end - seq_start;
    const uint32_t seq_num_chunks = CeilDiv(seq_len, matrix_size);
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

// ─── SyncAll: full cross-core barrier ────────────────────────
constexpr uint16_t SYNC_AIV_FLAG = 12;
constexpr uint16_t SYNC_AIC_FLAG = 11;
constexpr uint16_t SYNC_AIC_AIV_FLAG = 13;
constexpr uint16_t SYNC_AIV_ONLY_ALL = 14;
constexpr uint16_t SYNC_MODE_SHIFT_VALUE = 4;
constexpr uint16_t SYNC_FLAG_SHIFT_VALUE = 8;

/**
 * @brief Gets the FFTS message for cross-core synchronization.
 *
 * @param mode The synchronization mode.
 * @param flagId The event id to sync for.
 * @return The FFTS message.
 */
AICORE inline uint16_t GetffstMsg(uint16_t mode, uint16_t flagId) {
  return (0x1 + ((mode & 0x3) << SYNC_MODE_SHIFT_VALUE) +
          ((flagId & 0xf) << SYNC_FLAG_SHIFT_VALUE));
}

/**
 * @brief Synchronizes all cores.
 *
 * @tparam isAIVOnly Whether to synchronize only AIV cores.
 */
template <bool isAIVOnly = true>
AICORE inline void SyncAll() {
  pipe_barrier(PIPE_ALL);
  if constexpr (isAIVOnly) {
    ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x0, SYNC_AIV_ONLY_ALL));
    wait_flag_dev(SYNC_AIV_ONLY_ALL);
    return;
  }
#if defined(__DAV_CUBE__)
  wait_flag_dev(SYNC_AIV_FLAG);
  ffts_cross_core_sync(PIPE_FIX, GetffstMsg(0x0, SYNC_AIC_FLAG));
  wait_flag_dev(SYNC_AIC_FLAG);
  ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, SYNC_AIC_AIV_FLAG));
#elif defined(__DAV_VEC__)
  ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, SYNC_AIV_FLAG));
  wait_flag_dev(SYNC_AIC_AIV_FLAG);
#endif
}

/**
 * @brief It is true if `DataType` is supported as input for the Cube unit.
 *
 * @tparam DataType Data type to check.
 */
template <typename DataType>
constexpr bool IsCubeSupported =
    std::is_same_v<DataType, half> || std::is_same_v<DataType, float> ||
    std::is_same_v<DataType, int8_t> || std::is_same_v<DataType, uint8_t>;

/**
 * @brief Get the size of the fractal used internally by hardware along the
 * matrix K dimension.
 *
 * @tparam DataType Data type used for matrix multiplication.
 * @return The size of K dimension of the fractal.
 */
template <typename T>
constexpr __aicore__ inline uint16_t GetFractalK() {
  return 32 / sizeof(T);
}

/**
 * @brief Get the size of the fractal used internally by hardware along the
 * matrix M/N dimensions.
 *
 * @tparam DataType Data type used for matrix multiplication.
 * @return The size of both M and N dimensions of the fractal.
 */
template <typename T>
constexpr __aicore__ inline uint16_t GetFractalMN() {
  return 16;
}

/**
 * @brief A type metafunction for Cube's input / output types. The following
 * type pairs are supported:(half, float), (float, float), (int8_t, int32_t),
 * (uint8_t, uint32_t)
 *
 * @tparam InputT Input cube type. Must be int8_t, uint8_t, half, or float.
 */
template <typename InputT>
struct CubeOutType {
  /// @brief Type
  using type = InputT;
};

/**
 * @brief Cube data type map int8_t -> int32_t.
 */
template <>
struct CubeOutType<int8_t> {
  /// @brief Type
  using type = int32_t;
};

/**
 * @brief Cube data type map uint8_t -> uint32_t.
 */
template <>
struct CubeOutType<uint8_t> {
  /// @brief Type
  using type = uint32_t;
};

/**
 * @brief Cube data type map half -> float.
 */
template <>
struct CubeOutType<half> {
  /// @brief Type
  using type = float;
};

/**
 * @brief Cube data type map float -> float.
 */
template <>
struct CubeOutType<float> {
  /// @brief Type
  using type = float;
};

/**
 * @brief Syntactic sugar for `CubeOutType`
 * @tparam T Input type
 */
template <typename T>
using CubeOutType_t = typename CubeOutType<T>::type;

/**
 * @brief Defines how the workload should be distributed among cores.
 *
 * The function returns the number of tiles to be processed by each block so
 * that the depth of execution is minimized. If the workload is not balanced it
 * will greedily assign as many tiles as possible starting from the first block,
 * but keeping the maximum depth optimal. If `vec_len` is not divisible by
 * `tile_size` the last tile will be smaller.
 *
 * @param [in] vec_len Size of the input vector.
 * @param [in] tile_size Tile size.
 * @param [in] block_n Number of blocks.
 *
 * @return Number of tiles assigned to the block calling the function.
 */
__aicore__ inline uint32_t GetWorkDistribution(uint32_t vec_len,
                                               uint32_t tile_size,
                                               uint32_t block_n) {
  const uint32_t num_tiles = CeilDiv(vec_len, tile_size);
  const uint32_t max_num_tiles_per_block = CeilDiv(num_tiles, block_n);
  uint32_t num_tiles_to_process = max_num_tiles_per_block;
  const int tiles_left =
      (int)num_tiles - (int)(get_block_idx() * max_num_tiles_per_block);

  if (tiles_left < 0) {
    num_tiles_to_process = 0;
  } else if (tiles_left < static_cast<int>(max_num_tiles_per_block)) {
    num_tiles_to_process = tiles_left;
  }
  return num_tiles_to_process;
}

/**
 * @brief Returns a synchronization config.
 *
 * @param [in] mode Synchronization mode.
 * @param [in] flag_id Flag to use for synchronization.
 * @return Synchronization config.
 */
__aicore__ inline int GetSyncConf(int mode, int flag_id) {
  return 1 | (mode << 4) | (flag_id << 8);
}

/**
 * @brief Used to specifies the direction of synchronization when synchronizing
 * cube and vectors within a single group.
 *
 * A single group consists of one cube core and two vector cores.
 *
 * Can be used to specify either the symmetric or asymetric synchronization.
 */
enum class GroupSyncDirection {
  /// Asymetric synchronization - cube continues execution only after vectors
  /// reach the synchronization point. Can be used when cube consumes the data
  /// produced by vectors from the same group.
  CUBE_WAIT_FOR_VEC,
  /// Asymetric synchronization - vectors continue execution only after cube
  /// reaches the synchronization point. Can be used when vectors consume the
  /// data produced by cube from the same group.
  VEC_WAIT_FOR_CUBE,
  /// Symmetric synchronization - execution continues after cubes and vectors
  /// synchronize at the same time
  FULL
};

/**
 * @brief Synchronize cube and vector cores within a single group.
 *
 * @tparam Dir Direction of the synchronization.
 */
template <GroupSyncDirection Dir = GroupSyncDirection::FULL>
__aicore__ inline void SyncGroup() {
  const int mode = 2;

  if constexpr (Dir == GroupSyncDirection::CUBE_WAIT_FOR_VEC) {
    const int AIV_SET_FLAG_ID = 11;
    if constexpr (IS_AIV) {
      ffts_cross_core_sync(PIPE_MTE3, GetSyncConf(mode, AIV_SET_FLAG_ID));
    }
    if constexpr (IS_AIC) {
      wait_flag_dev(AIV_SET_FLAG_ID);
    }
    return;
  }
  if constexpr (Dir == GroupSyncDirection::VEC_WAIT_FOR_CUBE) {
    const int AIC_SET_FLAG_ID = 12;
    if constexpr (IS_AIC) {
      ffts_cross_core_sync(PIPE_FIX, GetSyncConf(mode, AIC_SET_FLAG_ID));
    }
    if constexpr (IS_AIV) {
      wait_flag_dev(AIC_SET_FLAG_ID);
    }
    return;
  }
  if constexpr (Dir == GroupSyncDirection::FULL) {
    const int AIV_SET_FLAG_ID = 11;
    const int AIC_SET_FLAG_ID = 12;
    if constexpr (IS_AIV) {
      ffts_cross_core_sync(PIPE_MTE3, GetSyncConf(mode, AIV_SET_FLAG_ID));
      wait_flag_dev(AIC_SET_FLAG_ID);
    }
    if constexpr (IS_AIC) {
      ffts_cross_core_sync(PIPE_FIX, GetSyncConf(mode, AIC_SET_FLAG_ID));
      wait_flag_dev(AIV_SET_FLAG_ID);
    }
    return;
  }
}

}  // namespace kernel_utils
