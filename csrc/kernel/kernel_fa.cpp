/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/

// TSyncCVID currently defines FFTS helper macros that collide with names used
// while pto-inst.hpp is parsed, so this include order is intentional.
// clang-format off
#include <pto/pto-inst.hpp>
#include <pto/npu/a2a3/custom/TSyncCVID.hpp>
// clang-format on

#include "fa_config.h"
#include "pto_macro_fa_gu.hpp"
#include "pto_macro_fa_softmax.hpp"
#include "pto_macro_matmul.hpp"

#define UF_ENABLE 1

using namespace std;
using namespace pto;

#ifndef FFTS_BUFFER_FLAG_ENUM
#define FFTS_BUFFER_FLAG_ENUM
// Buffer flag values for FFTS pipeline coordination
enum FftsBufferFlag : uint32_t {
  BUF0_QK_READY = 0,     // Buffer 0: QK data ready
  BUF0_SM_CONSUMED = 1,  // Buffer 0: Softmax consumed
  BUF1_SM_READY = 2,     // Buffer 1: Softmax output ready
  BUF1_SV_CONSUMED = 3,  // Buffer 1: SV consumed
  UPDATE_READY = 4,      // Update stage ready
  UPDATE_CONSUMED = 5,   // Update stage consumed
  CV_BLOCK_END =
      7,  // CV comm slot block end (CV_COMM_CTRL reserved in TSyncCVID)
};
#endif

enum CoreEvtID : uint32_t {
  QK_EVENT_ID0,
  QK_EVENT_ID1,
  PV_EVENT_ID0,
  PV_EVENT_ID1,
};

// -----------------------------------------------------------------------------
// Performance tuning knobs (high-level)
//
// The kernel is a cross-core pipeline (Cube + Vec) with explicit FIFOs:
//   QK (Cube):  compute_qk   -> qk_tile_fifo (fp32)
//   P  (Vec):   compute_p    -> p_tile_fifo  (fp16 x_exp) + l1_exp_max_ififo
//   PV (Cube):  compute_pv   -> pv_tile_fifo (fp32)
//   GU (Vec):   compute_gu   -> o_out (fp32) with running rescale/update
//
// Key knobs that impact throughput (see runTFA<> below):
// - CUBE_S0 / CUBE_S1: tile sizes for QK/PV cube matmuls (compute intensity vs.
// buffer pressure)
// - qkPreloadNum: pipeline warmup depth (more overlap vs. more L1 FIFO
// footprint)
// - *_TNBuffers: ping/pong depth for Mat tiles (overlap) and Vec tiles (latency
// hiding)
// - QKV_CV_FIFO / PV_CV_FIFO: FIFO depth between stages (avoid backpressure)
// -----------------------------------------------------------------------------

// Inline macro used for small, performance-sensitive functions
#ifndef PTO_INLINE
#define PTO_INLINE __attribute__((always_inline)) inline
#endif

// Detect build-time macros and expose as constexpr flags for clearer
// conditionals
#ifdef __DAV_CUBE__
constexpr bool DAV_CUBE = true;
#else
constexpr bool DAV_CUBE = false;
#endif

#ifdef __DAV_VEC__
constexpr bool DAV_VEC = true;
#else
constexpr bool DAV_VEC = false;
#endif

constexpr std::size_t MAX_TILE_L1_BYTES = 512U * 1024U;
constexpr std::size_t MAX_VEC_UB_BYTES = 192U * 1024U;

template <typename TileType>
constexpr AICORE std::size_t tile_storage_bytes() {
  using ElementType = typename TileType::DType;
  return static_cast<std::size_t>(TileType::Rows * TileType::Cols) *
         sizeof(ElementType);
}

template <typename TileType, std::size_t NumBuffers>
constexpr AICORE std::size_t tile_buffer_total_bytes() {
  return tile_storage_bytes<TileType>() * NumBuffers;
}

template <typename TileType, std::size_t NumBuffers>
AICORE inline uint32_t assign_tile_buffers(TileType (&tiles)[NumBuffers],
                                           uint32_t base_offset) {
  if constexpr (NumBuffers == 0) {
    return base_offset;
  }

  constexpr std::size_t total_storage_bytes =
      tile_buffer_total_bytes<TileType, NumBuffers>();
  static_assert(total_storage_bytes <= MAX_TILE_L1_BYTES,
                "Tile buffer L1 allocation exceeds 512KB");

  for (std::size_t idx = 0; idx < NumBuffers; ++idx) {
    const uint32_t tile_offset =
        base_offset +
        static_cast<uint32_t>(idx * tile_storage_bytes<TileType>());
    TASSIGN(tiles[idx], tile_offset);
  }

  return base_offset + static_cast<uint32_t>(total_storage_bytes);
}

template <typename TileA, std::size_t NumA, typename TileB, std::size_t NumB>
AICORE inline uint32_t assign_tile_buffers_union(TileA (&tilesA)[NumA],
                                                 TileB (&tilesB)[NumB],
                                                 uint32_t base_offset) {
  static_assert(NumA == NumB,
                "Union assignment expects matching buffer counts");
  if constexpr (NumA == 0) {
    return base_offset;
  }

  constexpr std::size_t stride_bytes =
      (tile_storage_bytes<TileA>() > tile_storage_bytes<TileB>())
          ? tile_storage_bytes<TileA>()
          : tile_storage_bytes<TileB>();
  constexpr std::size_t total_storage_bytes = stride_bytes * NumA;
  static_assert(total_storage_bytes <= MAX_VEC_UB_BYTES,
                "Union tile UB allocation exceeds 192KB");

  for (std::size_t idx = 0; idx < NumA; ++idx) {
    const uint32_t tile_offset =
        base_offset + static_cast<uint32_t>(idx * stride_bytes);
    TASSIGN(tilesA[idx], tile_offset);
    TASSIGN(tilesB[idx], tile_offset);
  }

  return base_offset + static_cast<uint32_t>(total_storage_bytes);
}

template <typename TileQType, std::size_t NumQ, typename TileKType,
          std::size_t NumK, typename TilePType, std::size_t NumP,
          typename TileVType, std::size_t NumV>
AICORE inline void allocate_cube_tile_buffers(TileQType (&qTiles)[NumQ],
                                              TileKType (&kTiles)[NumK],
                                              TilePType (&pTiles)[NumP],
                                              TileVType (&vTiles)[NumV]) {
  constexpr std::size_t total_bytes =
      tile_buffer_total_bytes<TileQType, NumQ>() +
      tile_buffer_total_bytes<TileKType, NumK>() +
      tile_buffer_total_bytes<TilePType, NumP>() +
      tile_buffer_total_bytes<TileVType, NumV>();
  static_assert(total_bytes <= MAX_TILE_L1_BYTES,
                "Total cube L1 allocation exceeds 512KB");

  uint32_t l1_offset = 0;
  l1_offset = assign_tile_buffers(qTiles, l1_offset);
  l1_offset = assign_tile_buffers(kTiles, l1_offset);
  l1_offset = assign_tile_buffers(pTiles, l1_offset);
  l1_offset = assign_tile_buffers(vTiles, l1_offset);
  (void)l1_offset;
}

template <typename TileDataF_T, typename ReduceTileF_T, typename TileDataH_T,
          typename TileOutT, std::size_t SrcBuffers, std::size_t XexpBuffers,
          std::size_t pvVecBuffers, std::size_t ExpMaxBuffers>
AICORE inline void allocate_vec_tile_buffers(
    TileDataF_T (&srcTiles)[SrcBuffers], ReduceTileF_T &m1_local_max,
    TileDataF_T &input_reduce_tmp, ReduceTileF_T &l1_local_sum,
    ReduceTileF_T &m2_global_max, ReduceTileF_T &l2_global_sum,
    ReduceTileF_T (&l1_exp_max)[ExpMaxBuffers],
    TileDataH_T (&x_expT)[XexpBuffers], TileOutT (&pvTile)[pvVecBuffers],
    TileOutT &runningOTile, TileDataF_T &triu, TileDataH_T &causal_e) {
  constexpr std::size_t float_tile_bytes = tile_storage_bytes<TileDataF_T>();
  constexpr std::size_t reduce_tile_bytes = tile_storage_bytes<ReduceTileF_T>();
  constexpr std::size_t xexp_bytes =
      tile_buffer_total_bytes<TileDataH_T, XexpBuffers>();
  constexpr std::size_t half_tile_bytes =
      tile_storage_bytes<TileDataH_T>();  // causal_e (fp16 i-j mask base)
  constexpr std::size_t out_tile_bytes = tile_storage_bytes<TileOutT>();
  constexpr std::size_t union_stride =
      (tile_storage_bytes<TileDataF_T>() > tile_storage_bytes<TileOutT>())
          ? tile_storage_bytes<TileDataF_T>()
          : tile_storage_bytes<TileOutT>();
  static_assert(
      SrcBuffers == pvVecBuffers,
      "src/pv ping-pong buffer counts must match for union allocation");
  constexpr std::size_t union_bytes = union_stride * SrcBuffers;
  constexpr std::size_t total_bytes =
      union_bytes + xexp_bytes + (reduce_tile_bytes * (3U + ExpMaxBuffers)) +
      (float_tile_bytes / 8 * 1U) + (float_tile_bytes * 1U) + out_tile_bytes +
      half_tile_bytes;
  static_assert(total_bytes <= MAX_VEC_UB_BYTES,
                "Vec tile UB allocation exceeds 192KB");

  uint32_t offset = 0;
  TASSIGN(runningOTile, offset);
  offset += out_tile_bytes;
  offset = assign_tile_buffers_union(srcTiles, pvTile, offset);

  TASSIGN(m1_local_max, offset);
  offset += static_cast<uint32_t>(reduce_tile_bytes);

  TASSIGN(m2_global_max, offset);
  offset += static_cast<uint32_t>(reduce_tile_bytes);

  uint32_t tmp_float_offset = offset;
  TASSIGN(input_reduce_tmp, tmp_float_offset);
  offset += static_cast<uint32_t>(float_tile_bytes) / 8;

  TASSIGN(triu, offset);
  offset += static_cast<uint32_t>(float_tile_bytes);

  TASSIGN(l1_local_sum, offset);
  offset += static_cast<uint32_t>(reduce_tile_bytes);

  TASSIGN(l2_global_sum, offset);
  offset += static_cast<uint32_t>(reduce_tile_bytes);

  offset = assign_tile_buffers(l1_exp_max, offset);

  offset = assign_tile_buffers(x_expT, offset);

  TASSIGN(causal_e, offset);
  offset += static_cast<uint32_t>(half_tile_bytes);

  (void)offset;
}

// Helper to assign an accumulator tile to one of two ping-pong UB addresses
// (0x0 / 0x10000). Keeps a per-type static running index that toggles on every
// call. Caller may pass `initial_id` (0 or 1) to set the starting buffer index
// on the first call for that tile type.
template <typename AccTileT>
AICORE inline int assign_running_acc_tile(AccTileT &accTile,
                                          int initial_id = -1) {
  static int running_tile_buffer_idx =
      0;  // per-instantiation running buffer index: 0 -> base0, 1 -> base1
  if (initial_id == 0 || initial_id == 1) {
    running_tile_buffer_idx = initial_id;
  }
  const int id = running_tile_buffer_idx;
  const uint32_t base_addr = (id == 0) ? 0x0u : 0x10000u;
  TASSIGN(accTile, base_addr);
  running_tile_buffer_idx ^= 1;  // toggle for next call
  return id;
}

template <typename QKPipe, int HEAD_SIZE, int CUBE_S0, int CUBE_S1, int TILE_S1,
          int CV_FIFO_CONS_SYNC_PERIOD, bool INTERMEDIATE_CHECK,
          bool CAUSAL_MASK, typename TileMatQData, typename TileMatKData,
          typename TileQKData, typename QKSlotGlobal>
AICORE inline void compute_qk(QKPipe &qkPipe, int tile_id, int sub_tile_id,
                              __gm__ half *q, __gm__ half *k,
                              __gm__ float *qk_tile_fifo,
                              TileMatQData &qMatTile, TileMatKData &kMatTile,
                              TileQKData &qkAccTile, QKSlotGlobal &qkSlotGlobal,
                              uint64_t qkMatTileEventId, int accTileEvtID,
                              int blk_idx, int64_t q_seq_stride,
                              int64_t kv_seq_stride) {
  if constexpr (DAV_CUBE) {
    constexpr uint32_t Cube_S0 = CUBE_S0;
    constexpr uint32_t Cube_S1 = CUBE_S1;
    constexpr uint32_t Tile_S1 = TILE_S1;
    constexpr uint32_t kTileFactor = Tile_S1 / Cube_S1;
    constexpr uint32_t Cube_HEAD = HEAD_SIZE;

    constexpr int QKP_CV_FIFO = QKPipe::RingFiFo::SLOT_NUM;
    static_assert(QKP_CV_FIFO >= 1, "QKP_CV_FIFO must be >= 1");
    static_assert(Tile_S1 % Cube_S1 == 0,
                  "TILE_S1 must be divisible by CUBE_S1");

    const int s0_index = blk_idx * CUBE_S0;
    const int s1_index = tile_id * static_cast<int>(Tile_S1) +
                         sub_tile_id * static_cast<int>(Cube_S1);
    if (sub_tile_id == 0) {
      TALLOC<QKPipe, QKSlotGlobal, TileSplitAxis::TILE_UP_DOWN>(qkPipe,
                                                                qkSlotGlobal);
    }
    if constexpr (CAUSAL_MASK) {
      if (s1_index > s0_index) {
        if (sub_tile_id == static_cast<int>(kTileFactor) - 1) {
          TPUSH<QKPipe, QKSlotGlobal, TileSplitAxis::TILE_UP_DOWN>(
              qkPipe, qkSlotGlobal);
        }
        return;
      }
    }
    // Seq (row/col) strides are runtime so the same tile loads serve BNSD
    // (seq_stride == HEAD_SIZE) and, in future, BSND (seq_stride == num_heads *
    // HEAD_SIZE). The head dim stays contiguous (stride 1); only the
    // sequence-axis stride is DYNAMIC.
    using GlobalDataQ =
        GlobalTensor<half, pto::Shape<1, 1, 1, Cube_S0, HEAD_SIZE>,
                     pto::Stride<1, 1, 1, pto::DYNAMIC, 1>>;
    using GlobalDataK =
        GlobalTensor<half, pto::Shape<1, 1, 1, HEAD_SIZE, Cube_S1>,
                     pto::Stride<1, 1, 1, 1, pto::DYNAMIC>,
                     Layout::DN>;  // BNSD - (N, K) layout

    GlobalDataQ qGlobal(q, typename GlobalDataQ::Shape{},
                        typename GlobalDataQ::Stride(q_seq_stride));
    GlobalDataK kGlobal(k + s1_index * kv_seq_stride,
                        typename GlobalDataK::Shape{},
                        typename GlobalDataK::Stride(kv_seq_stride));

    wait_flag(PIPE_MTE1, PIPE_MTE2, qkMatTileEventId);

    if (tile_id == 0 && sub_tile_id == 0) {
      TLOAD(qMatTile, qGlobal);
    }

    TLOAD(kMatTile, kGlobal);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

#if UF_ENABLE
    pto_macro_matmul<Cube_S0, Cube_HEAD, Cube_S1>(qMatTile, kMatTile, qkAccTile,
                                                  AccMode::InitFinalSum);
#else
    wait_flag(PIPE_FIX, PIPE_M, accTileEvtID);
    pto_macro_matmul<Cube_S0, Cube_HEAD, Cube_S1>(qMatTile, kMatTile, qkAccTile,
                                                  AccMode::Init);
#endif

    set_flag(PIPE_MTE1, PIPE_MTE2, qkMatTileEventId);
#if !UF_ENABLE
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

    using QKStoreGlobal =
        GlobalTensor<float, pto::Shape<1, 1, 1, Cube_S0, Cube_S1>,
                     pto::Stride<1, 1, 1, Cube_S1, 1>>;
    const uint32_t buf_idx = static_cast<uint32_t>(tile_id % QKP_CV_FIFO);
    const size_t base_elems =
        static_cast<size_t>(buf_idx) * static_cast<size_t>(kTileFactor) *
            static_cast<size_t>(Cube_S0) * static_cast<size_t>(Cube_S1) +
        static_cast<size_t>(sub_tile_id) * static_cast<size_t>(Cube_S0) *
            static_cast<size_t>(Cube_S1);
    QKStoreGlobal qkStoreGlobal(qk_tile_fifo + base_elems);
#if UF_ENABLE
    TSTORE<STPhase::Final>(qkStoreGlobal, qkAccTile);
#else
    TSTORE(qkStoreGlobal, qkAccTile);
#endif

    if (sub_tile_id == static_cast<int>(kTileFactor) - 1) {
      TPUSH<QKPipe, QKSlotGlobal, TileSplitAxis::TILE_UP_DOWN>(qkPipe,
                                                               qkSlotGlobal);
    }

#if !UF_ENABLE
    set_flag(PIPE_FIX, PIPE_M, accTileEvtID);
#endif
  }
}

template <typename PPipe, typename PVPipe, int HEAD_SIZE, int CUBE_S0,
          int CUBE_S1, int TILE_S1, int PV_CV_FIFO,
          int CV_FIFO_CONS_SYNC_PERIOD, bool INTERMEDIATE_CHECK,
          bool CAUSAL_MASK, typename TileMatPData, typename TileMatVData,
          typename TilePVData, typename PSlotGlobal, typename PVSlotGlobal>
AICORE inline void compute_pv(PPipe &pPipe, PVPipe &pvPipe, int tile_id,
                              int sub_tile_id, __gm__ half *v,
                              __gm__ half *p_tile_fifo, TileMatPData &pMatTile,
                              TileMatVData &vMatTile, TilePVData &pvAccTile,
                              PSlotGlobal &pSlotGlobal,
                              PVSlotGlobal &pvSlotGlobal,
                              uint64_t svMatTileEventId, int accTileEvtID,
                              int blk_idx, int64_t kv_seq_stride) {
  constexpr uint32_t Cube_S0 = CUBE_S0;
  constexpr uint32_t Cube_S1 = CUBE_S1;
  constexpr uint32_t Tile_S1 = TILE_S1;
  constexpr uint32_t kTileFactor = Tile_S1 / Cube_S1;
  constexpr uint32_t Cube_HEAD = HEAD_SIZE;
  constexpr uint32_t TileElems = Cube_S0 * Tile_S1;
  constexpr int QKP_CV_FIFO = PVPipe::RingFiFo::SLOT_NUM;
  static_assert(QKP_CV_FIFO >= 1, "PV_CV_FIFO must be >= 1");
  static_assert(Tile_S1 % Cube_S1 == 0, "TILE_S1 must be divisible by CUBE_S1");

  const int s0_index = blk_idx * Cube_S0;
  const int s1_index = tile_id * static_cast<int>(Tile_S1) +
                       sub_tile_id * static_cast<int>(Cube_S1);
  const bool is_last_subtile =
      (sub_tile_id + 1 == static_cast<int>(kTileFactor));
  const bool next_will_be_skipped =
      (s1_index + static_cast<int>(Cube_S1)) > s0_index && CAUSAL_MASK;

  if constexpr (DAV_CUBE) {
    if (sub_tile_id == 0) {
      TPOP<PPipe, PSlotGlobal, TileSplitAxis::TILE_UP_DOWN>(pPipe, pSlotGlobal);
    }
    if constexpr (CAUSAL_MASK) {
      if (s1_index > s0_index) {
        if (is_last_subtile) {
          TFREE<PPipe, PSlotGlobal, TileSplitAxis::TILE_UP_DOWN>(pPipe,
                                                                 pSlotGlobal);
        }
        return;
      }
    }

    // Runtime seq stride (see compute_qk): BNSD -> HEAD_SIZE, BSND ->
    // num_kv_heads * HEAD_SIZE.
    using GlobalVT = GlobalTensor<half, pto::Shape<1, 1, 1, Cube_S1, HEAD_SIZE>,
                                  pto::Stride<1, 1, 1, pto::DYNAMIC, 1>>;

    wait_flag(PIPE_MTE1, PIPE_MTE2, svMatTileEventId);

    GlobalVT vLoad((__gm__ half *)(v + s1_index * kv_seq_stride),
                   typename GlobalVT::Shape{},
                   typename GlobalVT::Stride(kv_seq_stride));
    TLOAD(vMatTile, vLoad);

    using PLoadGlobal =
        GlobalTensor<half, pto::Shape<1, 1, 1, Cube_S0, Cube_S1>,
                     pto::Stride<1, 1, 1, Cube_S1, 1>>;
    const uint32_t buf_idx =
        static_cast<uint32_t>(tile_id % PPipe::RingFiFo::SLOT_NUM);
    const size_t base_elems =
        static_cast<size_t>(buf_idx) * static_cast<size_t>(Cube_S0) *
            static_cast<size_t>(Tile_S1) +
        static_cast<size_t>(sub_tile_id) * static_cast<size_t>(Cube_S0) *
            static_cast<size_t>(Cube_S1);
    PLoadGlobal pLoadGlobal(p_tile_fifo + base_elems);
    TLOAD(pMatTile, pLoadGlobal);
    if (is_last_subtile) {
      TFREE<PPipe, PSlotGlobal, TileSplitAxis::TILE_UP_DOWN>(pPipe,
                                                             pSlotGlobal);
    }

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

#if !UF_ENABLE
    if (sub_tile_id == 0) {
      wait_flag(PIPE_FIX, PIPE_M, accTileEvtID);
    }
#endif

#if UF_ENABLE
    const AccMode accMode = (sub_tile_id == 0)
                                ? (is_last_subtile || next_will_be_skipped
                                       ? AccMode::InitFinalSum
                                       : AccMode::InitPartialSum)
                                : (is_last_subtile || next_will_be_skipped
                                       ? AccMode::AccFinalSum
                                       : AccMode::AccPartialSum);
    pto_macro_matmul<Cube_S0, Cube_S1, Cube_HEAD>(pMatTile, vMatTile, pvAccTile,
                                                  accMode);
#else
    const AccMode accMode = (sub_tile_id == 0) ? AccMode::Init : AccMode::Acc;
    pto_macro_matmul<Cube_S0, Cube_S1, Cube_HEAD>(pMatTile, vMatTile, pvAccTile,
                                                  accMode);
#endif

    set_flag(PIPE_MTE1, PIPE_MTE2, svMatTileEventId);

    if (sub_tile_id == static_cast<int>(kTileFactor) - 1 ||
        next_will_be_skipped) {
#if !UF_ENABLE
      set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

      using PVStoreGlobal =
          GlobalTensor<float, pto::Shape<1, 1, 1, Cube_S0, HEAD_SIZE>,
                       pto::Stride<1, 1, 1, HEAD_SIZE, 1>>;
      TALLOC<PVPipe, PVSlotGlobal, TileSplitAxis::TILE_UP_DOWN>(pvPipe,
                                                                pvSlotGlobal);
      PVStoreGlobal pvStoreGlobal(pvSlotGlobal.data());
#if UF_ENABLE
      TSTORE<STPhase::Final>(pvStoreGlobal, pvAccTile);
#else
      TSTORE(pvStoreGlobal, pvAccTile);
#endif
      TPUSH<PVPipe, PVSlotGlobal, TileSplitAxis::TILE_UP_DOWN>(pvPipe,
                                                               pvSlotGlobal);

#if !UF_ENABLE
      set_flag(PIPE_FIX, PIPE_M, accTileEvtID);
#endif
    }  // end loop
  }  // end if DAV_CUBE
}

template <typename QKPipe, typename PPipe, int HEAD_SIZE, int CUBE_S0,
          int CUBE_S1, int TILE_S1, int CV_FIFO_CONS_SYNC_PERIOD,
          bool INTERMEDIATE_CHECK, bool CAUSAL_MASK, typename TileDataF_T,
          typename TileDataH_T, typename ReduceTileF_T,
          typename QKVecSlotGlobal, typename PVecSlotGlobal>
AICORE inline void compute_p(
    QKPipe &qkPipe, PPipe &pPipe, int tile_id, int row_slice,
    __gm__ float *exp_max_ififo, __gm__ float *qk_tile_fifo,
    __gm__ half *p_tile_fifo, __gm__ float *global_sum_out,
    __gm__ float *exp_max_out, TileDataF_T &qkVecTile, TileDataH_T &x_expT,
    TileDataF_T &input_reduce_tmp, ReduceTileF_T &m1_local_max,
    ReduceTileF_T &l1_local_sum, ReduceTileF_T &m2_global_max,
    ReduceTileF_T &l2_global_sum, ReduceTileF_T &l1_exp_max_ififo,
    TileDataF_T triu, TileDataH_T causal_e, QKVecSlotGlobal &qkVecSlotGlobal,
    PVecSlotGlobal &pVecSlotGlobal, uint64_t pTileEventId, int blk_idx) {
  constexpr uint32_t Cube_S0 = CUBE_S0;
  constexpr uint32_t Cube_S1 = CUBE_S1;
  constexpr uint32_t Tile_S1 = TILE_S1;
  constexpr uint32_t kTileFactor = Tile_S1 / Cube_S1;
  constexpr uint32_t Vec_S0 = Cube_S0 / VEC_CORES / kTileFactor;
  constexpr int QKP_CV_FIFO = QKPipe::RingFiFo::SLOT_NUM;
  static_assert(QKP_CV_FIFO >= 1, "QKP_CV_FIFO must be >= 1");
  static_assert(Tile_S1 % Cube_S1 == 0, "TILE_S1 must be divisible by CUBE_S1");
  static_assert(Cube_S0 % (VEC_CORES * kTileFactor) == 0,
                "Vec rows must divide evenly across tile slices");
  const bool initFlag = (tile_id == 0);
  if constexpr (DAV_VEC) {
    const size_t subblock_base_rows = static_cast<size_t>(Cube_S0 / VEC_CORES) *
                                      static_cast<size_t>(get_subblockid());
    const size_t local_row_offset = static_cast<size_t>(row_slice * Vec_S0);
    const size_t row_offset = subblock_base_rows + local_row_offset;
    const int s0_index = blk_idx * Cube_S0 + row_offset;
    const int s1_index = tile_id * static_cast<int>(Tile_S1);
    wait_flag(PIPE_V, PIPE_MTE2, pTileEventId);

    if (row_slice == 0) {
      TPOP<QKPipe, QKVecSlotGlobal, TileSplitAxis::TILE_UP_DOWN>(
          qkPipe, qkVecSlotGlobal);
    }

    const uint32_t qk_buf_idx = static_cast<uint32_t>(tile_id % QKP_CV_FIFO);
    const size_t qk_base_elems =
        static_cast<size_t>(qk_buf_idx) * static_cast<size_t>(kTileFactor) *
        static_cast<size_t>(Cube_S0) * static_cast<size_t>(Cube_S1);
    __gm__ float *qk_ptr = qk_tile_fifo + qk_base_elems +
                           row_offset * static_cast<size_t>(Cube_S1);
    using QKLoadGlobal =
        GlobalTensor<float, pto::Shape<1, 1, 1, Vec_S0, Cube_S1>,
                     pto::Stride<1, 1, 1, Cube_S1, 1>>;
    using TileDataFSub = Tile<TileType::Vec, float, Vec_S0, Tile_S1,
                              BLayout::RowMajor, Vec_S0, Cube_S1>;
    for (int sub_col = 0; sub_col < static_cast<int>(kTileFactor); ++sub_col) {
      const int sub_s1_index = s1_index + sub_col * static_cast<int>(Cube_S1);
      if constexpr (CAUSAL_MASK) {
        // compute_qk skips causal subtiles that are fully above this row
        // block's triangle. Do not read their unwritten fp16 FIFO bytes; the
        // softmax causal mask overwrites those lanes before the row reduction
        // sees them.
        if (sub_s1_index > blk_idx * static_cast<int>(Cube_S0)) {
          continue;
        }
      }
      QKLoadGlobal qkLoadGlobal(qk_ptr + static_cast<size_t>(sub_col) *
                                             static_cast<size_t>(Cube_S0) *
                                             static_cast<size_t>(Cube_S1));
      TileDataFSub qkSub;
      TASSIGN(qkSub, (uint64_t)qkVecTile.data() +
                         static_cast<uint64_t>(sub_col) *
                             static_cast<uint64_t>(Cube_S1) * sizeof(float));
      TLOAD(qkSub, qkLoadGlobal);
    }
    if (row_slice == static_cast<int>(kTileFactor) - 1) {
      TFREE<QKPipe, QKVecSlotGlobal, TileSplitAxis::TILE_UP_DOWN>(
          qkPipe, qkVecSlotGlobal);
    }

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // Extract per-slice views into the per-core reduce tiles so each slice
    // writes into its row range
    using ReduceSliceTile =
        Tile<TileType::Vec, float, Vec_S0, 1, BLayout::ColMajor, Vec_S0, 1>;
    // reduce tiles live per vector core; offset only by row_slice within the
    // core (no subblock stride)
    const size_t reduce_slice_rows = static_cast<size_t>(row_slice * Vec_S0);
    const uint64_t reduce_row_byte_offset = reduce_slice_rows * sizeof(float);

    ReduceSliceTile m1_local_max_slice;
    ReduceSliceTile l1_local_sum_slice;
    ReduceSliceTile m2_global_max_slice;
    ReduceSliceTile l2_global_sum_slice;
    ReduceSliceTile l1_exp_max_slice;

    TASSIGN(m1_local_max_slice,
            (uint64_t)m1_local_max.data() + reduce_row_byte_offset);
    TASSIGN(l1_local_sum_slice,
            (uint64_t)l1_local_sum.data() + reduce_row_byte_offset);
    TASSIGN(m2_global_max_slice,
            (uint64_t)m2_global_max.data() + reduce_row_byte_offset);
    TASSIGN(l2_global_sum_slice,
            (uint64_t)l2_global_sum.data() + reduce_row_byte_offset);
    TASSIGN(l1_exp_max_slice,
            (uint64_t)l1_exp_max_ififo.data() + reduce_row_byte_offset);

    // Extract current slice state from full-length reduce tiles
    wait_flag(PIPE_MTE3, PIPE_V, pTileEventId);
    if (initFlag) {
      pto_macro_fa_softmax<true, HEAD_SIZE, CAUSAL_MASK>(
          x_expT, qkVecTile, m1_local_max_slice, l1_local_sum_slice,
          m2_global_max_slice, l2_global_sum_slice, l1_exp_max_slice,
          input_reduce_tmp, qkVecTile, triu, causal_e, s0_index, s1_index);
    } else {
      pto_macro_fa_softmax<false, HEAD_SIZE, CAUSAL_MASK>(
          x_expT, qkVecTile, m1_local_max_slice, l1_local_sum_slice,
          m2_global_max_slice, l2_global_sum_slice, l1_exp_max_slice,
          input_reduce_tmp, qkVecTile, triu, causal_e, s0_index, s1_index);
    }

    set_flag(PIPE_V, PIPE_MTE2, pTileEventId);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    if (row_slice == 0) {
      TALLOC<PPipe, PVecSlotGlobal, TileSplitAxis::TILE_UP_DOWN>(
          pPipe, pVecSlotGlobal);
    }
    using PStoreGlobal =
        GlobalTensor<half, pto::Shape<1, 1, 1, Vec_S0, Cube_S1>,
                     pto::Stride<1, 1, 1, Cube_S1, 1>>;
    using TileDataHSub = Tile<TileType::Vec, half, Vec_S0, Tile_S1,
                              BLayout::RowMajor, Vec_S0, Cube_S1>;
    __gm__ half *p_ptr =
        p_tile_fifo + qk_base_elems + row_offset * static_cast<size_t>(Cube_S1);
    for (int sub_col = 0; sub_col < static_cast<int>(kTileFactor); ++sub_col) {
      PStoreGlobal pStoreGlobal(p_ptr + static_cast<size_t>(sub_col) *
                                            static_cast<size_t>(Cube_S0) *
                                            static_cast<size_t>(Cube_S1));
      TileDataHSub xExpSub;
      TASSIGN(xExpSub, (uint64_t)x_expT.data() +
                           static_cast<uint64_t>(sub_col) *
                               static_cast<uint64_t>(Cube_S1) * sizeof(half));
      TSTORE(pStoreGlobal, xExpSub);
    }
    if (row_slice == static_cast<int>(kTileFactor) - 1) {
      TPUSH<PPipe, PVecSlotGlobal, TileSplitAxis::TILE_UP_DOWN>(pPipe,
                                                                pVecSlotGlobal);
    }

    set_flag(PIPE_MTE3, PIPE_V, pTileEventId);
    if constexpr (INTERMEDIATE_CHECK) {
      // On the final row_slice, emit the exp_max for this subblock only
      // (Cube_S0 / VEC_CORES rows)
      if (row_slice == static_cast<int>(kTileFactor) - 1) {
        constexpr uint32_t SubblockRows = Cube_S0 / VEC_CORES;
        using GlobalPMaxFloatSub =
            GlobalTensor<float, pto::Shape<1, 1, 1, 1, SubblockRows>,
                         pto::Stride<1, 1, 1, Cube_S0, 1>>;
        using ExpMaxSub = Tile<TileType::Vec, float, 1, SubblockRows,
                               BLayout::RowMajor, 1, SubblockRows>;
        const size_t base_elems_pmax =
            static_cast<size_t>(tile_id % QKP_CV_FIFO) *
                static_cast<size_t>(Cube_S0) +
            subblock_base_rows;
        __gm__ float *p_ptr_fp32 = exp_max_ififo + base_elems_pmax;
        GlobalPMaxFloatSub pMaxGlobal(p_ptr_fp32);
        ExpMaxSub l1_exp_max_rowmajor;
        TRESHAPE(l1_exp_max_rowmajor, l1_exp_max_ififo);
        TSTORE(pMaxGlobal, l1_exp_max_rowmajor);
      }
    }
  }
}

template <typename PVPipe, int HEAD_SIZE, int CUBE_S0, int TILE_S1,
          int PV_CV_FIFO, int CV_FIFO_CONS_SYNC_PERIOD, bool INTERMEDIATE_CHECK,
          bool CAUSAL_MASK, typename TileOutT, typename ReduceTileF_T>
AICORE inline void compute_gu(PVPipe &pvPipe, int tile_id, int num_tiles,
                              __gm__ float *o_out, __gm__ float *o_parts_out,
                              TileOutT &runningOTile, TileOutT &pvVecTile,
                              ReduceTileF_T &l1_exp_max_ififo,
                              ReduceTileF_T &l2_global_sum, uint64_t guEventId,
                              int64_t o_seq_stride) {
  constexpr uint32_t Cube_S0 = CUBE_S0;
  constexpr uint32_t Vec_S0 = Cube_S0 / VEC_CORES;

  if constexpr (DAV_VEC) {
    wait_flag(PIPE_V, PIPE_MTE2, guEventId);
    const size_t subblock_base_rows = static_cast<size_t>(Cube_S0 / VEC_CORES) *
                                      static_cast<size_t>(get_subblockid());

    using PVVecGlobal =
        GlobalTensor<float, pto::Shape<1, 1, 1, Vec_S0, HEAD_SIZE>,
                     pto::Stride<1, 1, 1, HEAD_SIZE, 1>>;
    PVVecGlobal pvGlobal;
    TPOP<PVPipe, PVVecGlobal, TileSplitAxis::TILE_UP_DOWN>(pvPipe, pvGlobal);

    if (tile_id == 0) {
      // runningOTile is single-buffered and reused every row-block. With the
      // per-block pipe_barrier gone (continuous cross-block flow), this MTE2
      // reload must wait for the previous block's MTE3 store of runningOTile to
      // finish. Paired with the set_flag after the final-tile TSTORE below;
      // primed by a one-time set_flag before the work loop.
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      TLOAD(runningOTile, pvGlobal);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      // Single-tile case (num_tiles == 1): tile 0 is also the last tile and
      // must run the final softmax normalization. This is needed for the
      // non-causal path too, not only causal — otherwise a single-S1-tile (S1
      // == TILE_S1) output is left divided-by-nothing.
      if (tile_id == num_tiles - 1) {
        pto_macro_fa_gu_single_and_last_tile(runningOTile, l2_global_sum);
      }
    } else {
      TLOAD(pvVecTile, pvGlobal);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

      if (tile_id < num_tiles - 1) {
        pto_macro_fa_gu<ReduceTileF_T, TileOutT>(runningOTile, pvVecTile,
                                                 l1_exp_max_ififo);
      } else {
        pto_macro_fa_gu_last<ReduceTileF_T, TileOutT>(
            runningOTile, pvVecTile, l1_exp_max_ififo, l2_global_sum);
      }
    }
    TFREE<PVPipe, PVVecGlobal, TileSplitAxis::TILE_UP_DOWN>(pvPipe, pvGlobal);

    set_flag(PIPE_V, PIPE_MTE2, guEventId);

    if (tile_id == num_tiles - 1) {
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      // Runtime seq stride (see compute_qk): BNSD -> HEAD_SIZE, BSND ->
      // num_q_heads * HEAD_SIZE.
      using GlobalOutT =
          GlobalTensor<float, pto::Shape<1, 1, 1, Vec_S0, HEAD_SIZE>,
                       pto::Stride<1, 1, 1, pto::DYNAMIC, 1>>;
      GlobalOutT outGlobal(
          (__gm__ float *)(o_out + subblock_base_rows * o_seq_stride),
          typename GlobalOutT::Shape{},
          typename GlobalOutT::Stride(o_seq_stride));
      TSTORE(outGlobal, runningOTile);
      // Signal that this block's runningOTile store is issued so the next
      // block's tile-0 reload (wait_flag above) can safely overwrite the buffer
      // once the store completes.
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
  }
}

template <int HEAD_SIZE, int CUBE_S0, int CUBE_S1, int TILE_S1,
          int CV_FIFO_SIZE, bool INTERMEDIATE_CHECK, bool CAUSAL_MASK,
          int CV_FIFO_CONS_SYNC_PERIOD>
AICORE inline void runTFAImpl(
    uint32_t S0, uint32_t S1, uint32_t qk_preload, uint32_t batch,
    uint32_t num_q_heads, uint32_t num_kv_heads, int64_t q_batch_stride,
    int64_t q_head_stride, int64_t q_seq_stride, int64_t kv_batch_stride,
    int64_t kv_head_stride, int64_t kv_seq_stride, __gm__ uint64_t *ffts_addr,
    __gm__ half *q, __gm__ half *k, __gm__ half *v, __gm__ half *p_tile_fifo,
    __gm__ float *exp_max_ififo, __gm__ float *global_sum_out,
    __gm__ float *exp_max_out, __gm__ float *o_out, __gm__ float *o_parts_out,
    __gm__ float *qk_tile_fifo, __gm__ float *pv_tile_fifo,
    __gm__ uint8_t *cv_comm_buf, __gm__ uint8_t *profile_buf) {
  uint64_t tStart = get_sys_cnt();

  set_ffts_base_addr((uint64_t)ffts_addr);
  // NOTE: the cube/vec pipeline entry flags are set per row-block inside the
  // LPT loop below (and drained at the end of each block), so a single core can
  // process several row-blocks in sequence. Reset the vector mask at kernel
  // entry so a stale mask left by a previous launch cannot corrupt vector
  // addressing on back-to-back launches.
  if constexpr (DAV_VEC) {
    set_mask_norm();
    set_vector_mask(-1, -1);
  }

  // Rename dimensions for clarity: S0 (rows total), Cube_S0 (per-block rows),
  // S1 (cols), HEAD_SIZE (inner)
  constexpr uint32_t Cube_S0 = CUBE_S0;
  const uint32_t block_rows = S0 / CUBE_S0;  // runtime: S0 is a kernel arg now
  constexpr uint32_t Cube_S1 = CUBE_S1;      // per-tile S1 chunk
  constexpr uint32_t Tile_S1 = TILE_S1;      // logical tile along S1
  static_assert(Tile_S1 % Cube_S1 == 0, "TILE_S1 must be divisible by CUBE_S1");
  constexpr uint32_t kTileFactor = Tile_S1 / Cube_S1;  // sub-tiles per TILE_S1
  constexpr uint32_t Cube_HEAD = HEAD_SIZE;
  constexpr uint32_t Vec_S0 = Cube_S0 / VEC_CORES / kTileFactor;
  constexpr uint32_t VecGuRows = Cube_S0 / VEC_CORES;
  static_assert(Cube_S0 % (VEC_CORES * kTileFactor) == 0,
                "Vec rows must divide evenly across tile slices");

  // --------------------------
  // Tuning knobs (pipeline)
  //
  // qkPreloadNum controls how many (QK -> P) tiles we warm up before entering
  // the steady-state loop.
  // - Larger preload improves overlap (Cube/VEC concurrency) for long S1.
  // - Larger preload increases FIFO footprint (qkGlobalTensorNBuffers /
  // pvGlobalTensorNBuffers / guGlobalTensorNBuffers). Runtime warmup depth (see
  // header). Callers validate the range; the kernel trusts it.
  const uint32_t qkPreloadNum = qk_preload;

  // Buffer counts for optional double-buffering (default 1)
  // - srcVecTNBuffers/xexpVecTNBuffers: Vec ping-pong for QK load and x_exp
  // output
  // - *MatTNBuffers: L1 ping-pong for Cube stage (K/P/V)
  // Keep these small (1-2) unless you have measured stall bubbles that require
  // deeper buffering.
  constexpr uint32_t srcVecTNBuffers = 2;
  constexpr uint32_t xexpVecTNBuffers = 2;
  constexpr uint32_t outOTileNBuffers = 2;
  constexpr uint32_t qMatTNBuffers = 1;
  constexpr uint32_t kMatTNBuffers = 2;
  constexpr uint32_t pMatTNBuffers = 2;
  constexpr uint32_t vMatTNBuffers = 2;
  constexpr uint32_t qkp_tile_fifo_size = CV_FIFO_SIZE;
  constexpr uint32_t pv_tile_fifo_size = CV_FIFO_SIZE;
  static_assert(CV_FIFO_CONS_SYNC_PERIOD >= 1,
                "CV_FIFO_CONS_SYNC_PERIOD must be >= 1");
  // qkPreloadNum is runtime now, so its bounds (1 <= n <= CV_FIFO_SIZE, and n >
  // 1 unless kTileFactor == 1) are validated on the host in tfa_run before
  // launch.

  // Define tile types for first QK matmul
  using TileMatQData =
      Tile<TileType::Mat, half, Cube_S0, HEAD_SIZE, BLayout::ColMajor, Cube_S0,
           HEAD_SIZE, SLayout::RowMajor, 512>;
  using TileMatKData =
      Tile<TileType::Mat, half, HEAD_SIZE, Cube_S1, BLayout::RowMajor,
           HEAD_SIZE, Cube_S1, SLayout::ColMajor, 512>;
  // Accumulator rows must match Cube_S0 (per-block rows), not logical S0
  using TileQKData = TileAcc<float, Cube_S0, Cube_S1, Cube_S0, Cube_S1>;

  TileMatQData qMatTile[qMatTNBuffers];
  TileMatKData kMatTile[kMatTNBuffers];
  TileQKData qkAccTile;

  // Define tile types for second PV matmul
  using TileMatPData =
      Tile<TileType::Mat, half, Cube_S0, Cube_S1, BLayout::ColMajor, Cube_S0,
           Cube_S1, SLayout::RowMajor, 512>;
  using TileMatVData =
      Tile<TileType::Mat, half, Cube_S1, HEAD_SIZE, BLayout::ColMajor, Cube_S1,
           HEAD_SIZE, SLayout::RowMajor, 512>;
  using TilePVData = TileAcc<float, Cube_S0, HEAD_SIZE, Cube_S0, HEAD_SIZE>;

  TileMatPData pMatTile[pMatTNBuffers];
  TileMatVData vMatTile[vMatTNBuffers];
  TilePVData pvAccTile;

  allocate_cube_tile_buffers(qMatTile, kMatTile, pMatTile, vMatTile);

  // Assign accumulator tiles using ping-pong helper. qk starts at 0, pv starts
  // at 1.
  assign_running_acc_tile(qkAccTile, 0);
  assign_running_acc_tile(pvAccTile, 1);

  // Define tile types for FA softmax P computation
  // UB offsets for softmax tiles
  // Define per-tile vector tiles sized to Cube_S1
  using TileDataF_T = Tile<TileType::Vec, float, Vec_S0, Tile_S1,
                           BLayout::RowMajor, Vec_S0, Tile_S1>;
  using TileDataH_T = Tile<TileType::Vec, half, Vec_S0, Tile_S1,
                           BLayout::RowMajor, Vec_S0, Tile_S1>;
  constexpr uint32_t SubblockRows = Cube_S0 / VEC_CORES;
  // Reduce tiles cover one vector core's rows (Cube_S0 / VEC_CORES); slices are
  // extracted per row_slice
  using ReduceTileF_T = Tile<TileType::Vec, float, SubblockRows, 1,
                             BLayout::ColMajor, SubblockRows, 1>;

  TileDataF_T qkVecTile[srcVecTNBuffers];
  ReduceTileF_T m1_local_max;
  TileDataF_T input_reduce_tmp;
  TileDataF_T triu;
  ReduceTileF_T l1_local_sum;
  ReduceTileF_T m2_global_max;
  ReduceTileF_T l2_global_sum;
  ReduceTileF_T l1_exp_max_ififo[qkp_tile_fifo_size];
  TileDataH_T x_expT[xexpVecTNBuffers];
  // Precomputed causal mask base E[i][j] = i - j (fp16, exact for our tile
  // sizes). The per-diagonal additive mask is min(base_phase + E, 0) *
  // kCausalMaskNeg, replacing the old per-row scalar TTRI.
  TileDataH_T causal_e;

  using TileOutGuT = Tile<TileType::Vec, float, VecGuRows, HEAD_SIZE,
                          BLayout::RowMajor, VecGuRows, HEAD_SIZE>;
  TileOutGuT pvVecTile[outOTileNBuffers];
  TileOutGuT runningOTile;
  allocate_vec_tile_buffers<TileDataF_T, ReduceTileF_T, TileDataH_T, TileOutGuT,
                            srcVecTNBuffers, xexpVecTNBuffers,
                            outOTileNBuffers>(
      qkVecTile, m1_local_max, input_reduce_tmp, l1_local_sum, m2_global_max,
      l2_global_sum, l1_exp_max_ififo, x_expT, pvVecTile, runningOTile, triu,
      causal_e);

  // Generate E[i][j] = i - j once (vec only, causal only). Row 0 is
  // [0,-1,-2,...] via a descending TCI; row i is row 0 + i. This is a one-time
  // scalar TCI + Vec_S0-1 vector adds, amortized over the whole kernel — the
  // hot path then just adds a scalar and clamps (no per-diagonal-tile scalar
  // loop).
  if constexpr (DAV_VEC && CAUSAL_MASK) {
    // Build E in fp32 (scalar half ops are illegal on aicore) in the triu
    // scratch, then cast to fp16 once. triu is free here because compute_p only
    // uses it later inside the work loop.
    const uint64_t e_scratch = (uint64_t)triu.data();
    using ERowF =
        Tile<TileType::Vec, float, 1, Tile_S1, BLayout::RowMajor, 1, Tile_S1>;
    using EBlockF = Tile<TileType::Vec, float, Vec_S0, Tile_S1,
                         BLayout::RowMajor, Vec_S0, Tile_S1>;
    ERowF e_row0;
    TASSIGN(e_row0, e_scratch);
    TCI<ERowF, float, 1>(e_row0, 0.0f);  // e_row0[j] = 0 - j
    pipe_barrier(PIPE_V);
    for (int i = 1; i < static_cast<int>(Vec_S0); ++i) {
      ERowF e_rowi;
      TASSIGN(e_rowi,
              e_scratch + static_cast<uint64_t>(i) * Tile_S1 * sizeof(float));
      TADDS(e_rowi, e_row0, static_cast<float>(i));  // e_rowi[j] = i - j
    }
    pipe_barrier(PIPE_V);
    EBlockF e_block;
    TASSIGN(e_block, e_scratch);
    TCVT(causal_e, e_block, RoundMode::CAST_ROUND);  // fp32 E -> fp16 causal_e
    pipe_barrier(PIPE_V);
  }

  // This core's id in [0, n_cores). The host launches n_cores = min(block_rows,
  // kFaMaxCores) cores; each loops over the row-blocks the LPT scheduler below
  // assigns to it. comm_slot and every FIFO / scratch buffer are indexed by
  // this id (not the logical row) and reused across the core's blocks.
#if defined(__DAV_C220_CUBE__) || \
    defined(__DAV_C220_VEC__)  // A5 defined macro, don't need to reassign
  const int block_idx = get_block_idx();
#endif
  // Total independent row-blocks across the whole (batch x q_head x row_block)
  // grid. Each is a self-contained online-softmax chain; the LPT loop below
  // hands them out to the launched cores.
  const uint32_t total_row_blocks = batch * num_q_heads * block_rows;
  const uint32_t n_cores =
      (total_row_blocks < static_cast<uint32_t>(kFaMaxCores))
          ? total_row_blocks
          : static_cast<uint32_t>(kFaMaxCores);

  // n_cores <= kFaMaxCores < kCvMaxCores, so this stays on the direct comm_slot
  // == block_idx path.
  const bool use_cv_comm = (!INTERMEDIATE_CHECK) &&
                           (n_cores >= static_cast<uint32_t>(pto::kCvMaxCores));
  int comm_slot = block_idx;

  if (use_cv_comm) {
    comm_slot = pto::TSYNC_CVID(block_idx, cv_comm_buf);
  }
  __gm__ uint64_t *profile_entry = nullptr;
  if (profile_buf != nullptr) {
    std::size_t profile_block_base =
        static_cast<std::size_t>(block_idx) * kFaProfileBytesPerBlock;
    std::size_t profile_offset = profile_block_base;
    if constexpr (DAV_VEC) {
      profile_offset += (static_cast<std::size_t>(get_subblockid()) + 1U) *
                        1024U;  // vec subblock 0/1 use 2nd/3rd KB
    }
    profile_entry =
        reinterpret_cast<__gm__ uint64_t *>(profile_buf + profile_offset);
    profile_entry[0] = tStart;
  }
  const size_t p_fifo_block_stride = static_cast<size_t>(qkp_tile_fifo_size) *
                                     static_cast<size_t>(Cube_S0) *
                                     static_cast<size_t>(Tile_S1);
  const size_t p_max_fifo_block_stride =
      static_cast<size_t>(qkp_tile_fifo_size) * static_cast<size_t>(Cube_S0);
  const size_t qk_fifo_block_stride = p_fifo_block_stride;
  const size_t pv_fifo_block_stride = static_cast<size_t>(pv_tile_fifo_size) *
                                      static_cast<size_t>(Cube_S0) *
                                      static_cast<size_t>(HEAD_SIZE);

  // FIFO / scratch blocks are indexed by comm_slot (this core), fixed across
  // all its row-blocks.
  __gm__ half *p_tile_fifo_block =
      p_tile_fifo + static_cast<size_t>(comm_slot) * p_fifo_block_stride;
  __gm__ float *exp_max_ififo_block =
      exp_max_ififo + static_cast<size_t>(comm_slot) * p_max_fifo_block_stride;
  __gm__ float *qk_tile_fifo_block =
      qk_tile_fifo + static_cast<size_t>(comm_slot) * qk_fifo_block_stride;
  __gm__ float *pv_tile_fifo_block =
      pv_tile_fifo + static_cast<size_t>(comm_slot) * pv_fifo_block_stride;

  int p_gu_src_pingpong_id = 0;  // shared ping-pong for softmax vec tiles, pv
                                 // output tiles, and GU input tiles
  int k_src_pingpong_id = 0;     // separate ping-pong for K tiles
  int pv_src_pingpong_id = 0;    // separate ping-pong for P V tiles

  int qkAccTileEvtID = 0;
  int pvAccTileEvtID = 0;

  // FIFO definitions
  constexpr uint8_t FiFoDepth = CV_FIFO_SIZE;
#if defined(__DAV_C220_CUBE__) || defined(__DAV_C220_VEC__)
  constexpr uint8_t QK_PIPE_DIR = Direction::DIR_C2V;
  constexpr uint8_t P_PIPE_DIR = Direction::DIR_V2C;
  constexpr uint8_t PV_PIPE_DIR = Direction::DIR_C2V;
#elif defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
  constexpr uint8_t QK_PIPE_DIR = Direction::DIR_C2V_GM;
  constexpr uint8_t P_PIPE_DIR = Direction::DIR_V2C_GM;
  constexpr uint8_t PV_PIPE_DIR = Direction::DIR_C2V_GM;
#endif

  using QKPipe =
      TPipe<BUF0_QK_READY, QK_PIPE_DIR, Cube_S0 * Tile_S1 * sizeof(float),
            FiFoDepth, 2, false, UF_ENABLE ? true : false>;
  QKPipe qkPipe(qk_tile_fifo_block, (uint32_t)(uint64_t)qkVecTile[0].data(),
                0x0);

  // pFiFo, pProd, pCons
  using PPipe = TPipe<BUF1_SM_READY, P_PIPE_DIR,
                      Cube_S0 * Tile_S1 * sizeof(half), FiFoDepth>;
  PPipe pPipe(p_tile_fifo_block, 0x0, (uint32_t)(uint64_t)pMatTile[0].data());

  // pvFiFo, pvProd, pvCons
  using PVPipe =
      TPipe<UPDATE_READY, PV_PIPE_DIR, Cube_S0 * HEAD_SIZE * sizeof(float),
            FiFoDepth, 2, false, UF_ENABLE ? true : false>;
  PVPipe pvPipe(pv_tile_fifo_block, (uint32_t)(uint64_t)pvVecTile[0].data(),
                0x0);

  using QKSlotGlobal =
      GlobalTensor<float, pto::Shape<1, 1, 1, Cube_S0, Tile_S1>,
                   pto::Stride<1, 1, 1, Tile_S1, 1>>;
  using QKVecSlotGlobal =
      GlobalTensor<float, pto::Shape<1, 1, 1, VecGuRows, Tile_S1>,
                   pto::Stride<1, 1, 1, Tile_S1, 1>>;
  using PSlotGlobal = GlobalTensor<half, pto::Shape<1, 1, 1, Cube_S0, Tile_S1>,
                                   pto::Stride<1, 1, 1, Tile_S1, 1>>;
  using PVecSlotGlobal =
      GlobalTensor<half, pto::Shape<1, 1, 1, VecGuRows, Tile_S1>,
                   pto::Stride<1, 1, 1, Tile_S1, 1>>;
  using PVSlotGlobal =
      GlobalTensor<float, pto::Shape<1, 1, 1, Cube_S0, HEAD_SIZE>,
                   pto::Stride<1, 1, 1, HEAD_SIZE, 1>>;
  QKSlotGlobal qkSlotGlobal;
  QKVecSlotGlobal qkVecSlotGlobal;
  PSlotGlobal pSlotGlobal;
  PVecSlotGlobal pVecSlotGlobal;
  PVSlotGlobal pvSlotGlobal;

  // ---------------------------------------------------------------------------------------------
  // Load-balanced work assignment (persistent kernel). Unit of work = one
  // (batch, q_head, row-block) triple: a self-contained online-softmax chain
  // over that head's K/V.
  //
  // We assign with a greedy longest-processing-time (LPT) pass: walk the work
  // items heaviest -> lightest and give each to the currently least-loaded
  // core. Under the causal mask a row-block's cost grows with its row index
  // (block r sweeps min(1 + r*CUBE_S0/TILE_S1, full_tiles) key tiles); LPT
  // balances that triangle and degenerates to round-robin for the non-causal
  // rectangle. The argmin runs per work item, but it is only an O(n_cores<=24)
  // scalar scan next to a full matmul pipeline — negligible — and it must stay
  // per-item (not precomputed once per problem and reused): when block_rows is
  // not a multiple of n_cores, the leftover blocks of each problem have to land
  // on DIFFERENT cores across the B*H problems, which only the running
  // core_load[] accumulation does. Reusing one problem's map instead piles
  // every problem's leftovers on the same cores (e.g. block_rows=32, n_cores=24
  // -> 8 cores get 2 blocks and 16 get 1, every problem -> ~0.67 efficiency).
  //
  // Emission is problem-major (all of one (batch, q_head)'s row-blocks before
  // the next) so the cores running concurrently stay on one problem's K/V in L2
  // — a single large problem otherwise loses ~20% throughput when batches are
  // interleaved. Because every problem is identical, walking them in order
  // keeps the running loads balanced while preserving that locality.
  // ---------------------------------------------------------------------------------------------
  const uint32_t full_tiles = S1 / Tile_S1;
  uint32_t core_load[kFaMaxCores];
  for (uint32_t c = 0; c < n_cores; ++c) {
    core_load[c] = 0;
  }

  // --- One-time pipeline credits for continuous cross-block flow. ---
  // Previously these ping-pong credits were set at each row-block's entry and
  // reclaimed by a full drain + pipe_barrier(PIPE_ALL) at its exit, forcing a
  // fill/drain bubble at every boundary. We now prime them once here and drain
  // once after the whole work loop, so a core's next row-block can start
  // filling (QK + softmax) while the current block's PV/GU tail drains. The
  // QK/P/PV GM FIFOs stay correct across the boundary because of the
  // QK->P->PV->GU data dependency: a block's QK is fully consumed (by that
  // block's PV, via vec's P) before the same core issues the next block's QK,
  // so the per-block tile_id%SLOT_NUM addressing can never alias a live slot.
  // The only single-buffered tiles that outlive a boundary are Q (cube) and
  // runningOTile (vec); each gets a dedicated ping-pong handshake (EVENT_ID4 on
  // MTE1->MTE2 for Q, EVENT_ID0 on MTE3->MTE2 for O).
  if constexpr (DAV_CUBE) {
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
    set_flag(PIPE_MTE1, PIPE_MTE2,
             EVENT_ID4);  // Q buffer-free credit (paired per row-block)
  }
  if constexpr (DAV_VEC) {
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    set_flag(
        PIPE_MTE3, PIPE_MTE2,
        EVENT_ID0);  // runningOTile store->reload credit (paired per block)
  }

  for (uint32_t bh = 0; bh < batch * num_q_heads; ++bh) {
    for (int logical_block = static_cast<int>(block_rows) - 1;
         logical_block >= 0; --logical_block) {
      // num_tiles_s1 == the key tiles this row-block sweeps: rows past S1
      // attend every key, so a far-down causal block is capped at full_tiles
      // (the non-causal width).
      uint32_t blk_weight = full_tiles;
      if constexpr (CAUSAL_MASK) {
        const uint32_t diag_tiles =
            1u + (static_cast<uint32_t>(logical_block) * Cube_S0) / Tile_S1;
        blk_weight = diag_tiles < full_tiles ? diag_tiles : full_tiles;
      }
      // Scheduling weight != tile count: every row-block carries a fixed cost
      // (Q load, softmax init, GU, and — under causal — the diagonal-tile
      // masking) that a pure tile-count weight ignores. For causal the
      // near-diagonal blocks are short (1-2 tiles), so that fixed cost
      // dominates them and a tile-count LPT piles too many onto one core.
      // Adding kSchedFixed models it so LPT counts short blocks closer to their
      // true cost. Non-causal weights are uniform, so the constant does not
      // change that balance. (num_tiles_s1 below stays the real tile count —
      // this only steers LPT.)
      constexpr uint32_t kSchedFixed = 2u;
      const uint32_t sched_weight = blk_weight + kSchedFixed;
      // LPT step: give this triple to the currently least-loaded core.
      uint32_t best_core = 0;
      for (uint32_t c = 1; c < n_cores; ++c) {
        if (core_load[c] < core_load[best_core]) {
          best_core = c;
        }
      }
      core_load[best_core] += sched_weight;
      if (best_core != static_cast<uint32_t>(block_idx)) {
        continue;  // this work item belongs to another core
      }

      // --- Per-block setup for the work item this core just claimed. ---
      // GQA: consecutive groups of (num_q_heads / num_kv_heads) query heads
      // share one kv head.
      const uint32_t b = bh / num_q_heads;
      const uint32_t q_head = bh % num_q_heads;
      const uint32_t kv_head = q_head / (num_q_heads / num_kv_heads);
      // Base of this problem. Q and O use the q-side strides; K and V use the
      // kv-side strides. The head dim is contiguous (stride 1); only the
      // sequence-axis stride is layout-dependent.
      const size_t q_base =
          static_cast<size_t>(b) * static_cast<size_t>(q_batch_stride) +
          static_cast<size_t>(q_head) * static_cast<size_t>(q_head_stride);
      const size_t kv_base =
          static_cast<size_t>(b) * static_cast<size_t>(kv_batch_stride) +
          static_cast<size_t>(kv_head) * static_cast<size_t>(kv_head_stride);

      const int block_offset_rows = logical_block * static_cast<int>(Cube_S0);
      __gm__ half *q_block = q + q_base +
                             static_cast<size_t>(block_offset_rows) *
                                 static_cast<size_t>(q_seq_stride);
      __gm__ half *k_head = k + kv_base;
      __gm__ half *v_head = v + kv_base;
      __gm__ float *global_sum_block = nullptr;
      __gm__ float *exp_max_block = nullptr;
      if constexpr (INTERMEDIATE_CHECK) {
        global_sum_block = global_sum_out + block_offset_rows;
        exp_max_block = exp_max_out + block_offset_rows;
      }
      __gm__ float *o_out_block = o_out + q_base +
                                  static_cast<size_t>(block_offset_rows) *
                                      static_cast<size_t>(q_seq_stride);
      __gm__ float *o_parts_block = nullptr;
      // Actual key tiles this row-block sweeps == blk_weight from the scheduler
      // above; the cap keeps an S0 > S1 causal block from walking off the end
      // of K/V (its rows attend every key).
      const int num_tiles_s1 = static_cast<int>(blk_weight);

      // Fresh ping-pong indices for this row-block. The rolling credits primed
      // once before the work loop keep the K/V/P/softmax ping-pong buffers
      // protected across the boundary (no per-block barrier); only Q and
      // runningOTile need the extra per-block handshakes.
      k_src_pingpong_id = 0;
      p_gu_src_pingpong_id = 0;
      pv_src_pingpong_id = 0;

      // Q is single-buffered and reused every row-block: wait until the
      // previous block's QK matmuls released it before reloading Q (paired with
      // the set_flag after this block's steady loop).
      if constexpr (DAV_CUBE) {
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
      }

      // QK and P pre-computation (tile_id based)
      for (int preload_tile = 0;
           preload_tile < static_cast<int>(qkPreloadNum) &&
           preload_tile < num_tiles_s1;
           ++preload_tile) {
        if constexpr (DAV_CUBE) {
          for (int sub_tile = 0; sub_tile < static_cast<int>(kTileFactor);
               ++sub_tile) {
            qkAccTileEvtID = assign_running_acc_tile(qkAccTile);
            compute_qk<QKPipe, HEAD_SIZE, CUBE_S0, CUBE_S1, Tile_S1,
                       CV_FIFO_CONS_SYNC_PERIOD, INTERMEDIATE_CHECK,
                       CAUSAL_MASK>(
                qkPipe, preload_tile, sub_tile, q_block, k_head,
                qk_tile_fifo_block, qMatTile[0],
                kMatTile[k_src_pingpong_id % kMatTNBuffers], qkAccTile,
                qkSlotGlobal, k_src_pingpong_id % kMatTNBuffers, qkAccTileEvtID,
                logical_block, q_seq_stride, kv_seq_stride);
            k_src_pingpong_id++;
          }
        }
        if constexpr (DAV_VEC) {
          for (int row_slice = 0; row_slice < static_cast<int>(kTileFactor);
               ++row_slice) {
            // Init only on the very first S1 tile; row_slice partitions rows
            // within that tile
            compute_p<QKPipe, PPipe, HEAD_SIZE, CUBE_S0, CUBE_S1, Tile_S1,
                      CV_FIFO_CONS_SYNC_PERIOD, INTERMEDIATE_CHECK,
                      CAUSAL_MASK>(
                qkPipe, pPipe, preload_tile, row_slice, exp_max_ififo_block,
                qk_tile_fifo_block, p_tile_fifo_block, global_sum_block,
                exp_max_block,
                qkVecTile[p_gu_src_pingpong_id % srcVecTNBuffers],
                x_expT[p_gu_src_pingpong_id % xexpVecTNBuffers],
                input_reduce_tmp, m1_local_max, l1_local_sum, m2_global_max,
                l2_global_sum,
                l1_exp_max_ififo[preload_tile % qkp_tile_fifo_size], triu,
                causal_e, qkVecSlotGlobal, pVecSlotGlobal,
                p_gu_src_pingpong_id % xexpVecTNBuffers, logical_block);
            p_gu_src_pingpong_id++;
          }
        }
      }

      for (int tile_id = 0; tile_id < num_tiles_s1; ++tile_id) {
        int next_qk_tile =
            (tile_id + static_cast<int>(qkPreloadNum) >= num_tiles_s1)
                ? -1
                : (tile_id + static_cast<int>(qkPreloadNum));

        if (next_qk_tile != -1)
          qkAccTileEvtID = assign_running_acc_tile(qkAccTile);
        pvAccTileEvtID = assign_running_acc_tile(pvAccTile);

        for (int sub_tile = 0; sub_tile < static_cast<int>(kTileFactor);
             ++sub_tile) {
          if constexpr (DAV_CUBE) {
            if (next_qk_tile != -1) {
              compute_qk<QKPipe, HEAD_SIZE, CUBE_S0, CUBE_S1, Tile_S1,
                         CV_FIFO_CONS_SYNC_PERIOD, INTERMEDIATE_CHECK,
                         CAUSAL_MASK>(
                  qkPipe, next_qk_tile, sub_tile, q_block, k_head,
                  qk_tile_fifo_block, qMatTile[0],
                  kMatTile[k_src_pingpong_id % kMatTNBuffers], qkAccTile,
                  qkSlotGlobal, k_src_pingpong_id % kMatTNBuffers,
                  qkAccTileEvtID, logical_block, q_seq_stride, kv_seq_stride);
              k_src_pingpong_id++;
            }
          }

          if constexpr (DAV_VEC) {
            if (next_qk_tile != -1) {
              compute_p<QKPipe, PPipe, HEAD_SIZE, CUBE_S0, CUBE_S1, Tile_S1,
                        CV_FIFO_CONS_SYNC_PERIOD, INTERMEDIATE_CHECK,
                        CAUSAL_MASK>(
                  qkPipe, pPipe, next_qk_tile, sub_tile, exp_max_ififo_block,
                  qk_tile_fifo_block, p_tile_fifo_block, global_sum_block,
                  exp_max_block,
                  qkVecTile[p_gu_src_pingpong_id % srcVecTNBuffers],
                  x_expT[p_gu_src_pingpong_id % xexpVecTNBuffers],
                  input_reduce_tmp, m1_local_max, l1_local_sum, m2_global_max,
                  l2_global_sum,
                  l1_exp_max_ififo[next_qk_tile % qkp_tile_fifo_size], triu,
                  causal_e, qkVecSlotGlobal, pVecSlotGlobal,
                  p_gu_src_pingpong_id % xexpVecTNBuffers, logical_block);
              p_gu_src_pingpong_id++;
            }
          }

          if constexpr (DAV_CUBE) {
            compute_pv<PPipe, PVPipe, HEAD_SIZE, CUBE_S0, CUBE_S1, Tile_S1,
                       pv_tile_fifo_size, CV_FIFO_CONS_SYNC_PERIOD,
                       INTERMEDIATE_CHECK, CAUSAL_MASK>(
                pPipe, pvPipe, tile_id, sub_tile, v_head, p_tile_fifo_block,
                pMatTile[pv_src_pingpong_id % pMatTNBuffers],
                vMatTile[pv_src_pingpong_id % vMatTNBuffers], pvAccTile,
                pSlotGlobal, pvSlotGlobal,
                pv_src_pingpong_id % vMatTNBuffers + PV_EVENT_ID0,
                pvAccTileEvtID, logical_block, kv_seq_stride);
            pv_src_pingpong_id++;
          }
        }

        if constexpr (DAV_VEC) {
          compute_gu<PVPipe, HEAD_SIZE, CUBE_S0, Tile_S1, pv_tile_fifo_size,
                     CV_FIFO_CONS_SYNC_PERIOD, INTERMEDIATE_CHECK, CAUSAL_MASK>(
              pvPipe, tile_id, num_tiles_s1, o_out_block, o_parts_block,
              runningOTile, pvVecTile[p_gu_src_pingpong_id % outOTileNBuffers],
              l1_exp_max_ififo[tile_id % qkp_tile_fifo_size], l2_global_sum,
              p_gu_src_pingpong_id % outOTileNBuffers, q_seq_stride);
          p_gu_src_pingpong_id++;
        }
      }  // end steady-state tile loop for this row-block

      // ---- Per-block boundary: release Q for the next row-block. ----
      // No pipe_barrier / full drain here: the rolling ping-pong credits and
      // the QK->P->PV->GU FIFO dependency carry correctness across the
      // boundary, letting the next block's fill overlap this block's PV/GU
      // tail. This set (on MTE1, after all of this block's QK matmuls have read
      // Q) pairs with the wait_flag(EVENT_ID4) at the top of the next block.
      if constexpr (DAV_CUBE) {
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
      }
    }  // end logical_block loop for this (batch, q_head)
  }  // end (batch, q_head) problem loop

  // --- One-time drain: reclaim the credits primed before the work loop. ---
  if constexpr (DAV_CUBE) {
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
  }
  if constexpr (DAV_VEC) {
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }

  pipe_barrier(PIPE_ALL);
  uint64_t tEnd = get_sys_cnt();
  if (profile_entry != nullptr) {
    profile_entry[1] = tEnd;
  }
#ifdef _DEBUG
  if constexpr (DAV_CUBE) {
    cce::printf("Core %d Cube Block %d, Start @%d End @%d (%d us)\n",
                get_coreid(), block_idx, int(tStart), int(tEnd),
                int(tEnd - tStart) * 20 / 1000);
  } else {
    cce::printf(
        "Core %d Vec Block %d, SubBlock %d, Start @%d End @%d (%d us)\n",
        get_coreid(), block_idx, int(get_subblockid()), int(tStart), int(tEnd),
        int(tEnd - tStart) * 20 / 1000);
  }
#endif
}

// extern "C" kernel entry points. runTFAImpl above is a templated device
// function; launching a
// __global__ template makes it show up in profilers (msprof) under its mangled
// name
// (_Z6runTFAILi128E...). These C-linkage wrappers give the two launched kernels
// readable symbols. Tiling is fixed for this build; INTERMEDIATE_CHECK stays
// false (flip it here to dump per-tile exp_max for precision debugging), so
// only the causal / non-causal split needs its own kernel.
extern "C" __global__ AICORE void fa_fp16(
    __gm__ uint8_t *ffts_addr, __gm__ uint8_t *q, __gm__ uint8_t *k,
    __gm__ uint8_t *v, __gm__ uint8_t *p_tile_fifo,
    __gm__ uint8_t *exp_max_ififo, __gm__ uint8_t *o_out,
    __gm__ uint8_t *qk_tile_fifo, __gm__ uint8_t *pv_tile_fifo, uint32_t S0,
    uint32_t S1, uint32_t qk_preload, uint32_t batch, uint32_t num_q_heads,
    uint32_t num_kv_heads, int64_t q_batch_stride, int64_t q_head_stride,
    int64_t q_seq_stride, int64_t kv_batch_stride, int64_t kv_head_stride,
    int64_t kv_seq_stride) {
  runTFAImpl<128, 128, kFaCubeS1, kFaTileS1, kFaCvFifoSize, false, false,
             kFaCvFifoConsSyncPeriod>(
      S0, S1, qk_preload, batch, num_q_heads, num_kv_heads, q_batch_stride,
      q_head_stride, q_seq_stride, kv_batch_stride, kv_head_stride,
      kv_seq_stride, reinterpret_cast<__gm__ uint64_t *>(ffts_addr),
      reinterpret_cast<__gm__ half *>(q), reinterpret_cast<__gm__ half *>(k),
      reinterpret_cast<__gm__ half *>(v),
      reinterpret_cast<__gm__ half *>(p_tile_fifo),
      reinterpret_cast<__gm__ float *>(exp_max_ififo), nullptr, nullptr,
      reinterpret_cast<__gm__ float *>(o_out), nullptr,
      reinterpret_cast<__gm__ float *>(qk_tile_fifo),
      reinterpret_cast<__gm__ float *>(pv_tile_fifo), nullptr, nullptr);
}

extern "C" __global__ AICORE void fa_causal_fp16(
    __gm__ uint8_t *ffts_addr, __gm__ uint8_t *q, __gm__ uint8_t *k,
    __gm__ uint8_t *v, __gm__ uint8_t *p_tile_fifo,
    __gm__ uint8_t *exp_max_ififo, __gm__ uint8_t *o_out,
    __gm__ uint8_t *qk_tile_fifo, __gm__ uint8_t *pv_tile_fifo, uint32_t S0,
    uint32_t S1, uint32_t qk_preload, uint32_t batch, uint32_t num_q_heads,
    uint32_t num_kv_heads, int64_t q_batch_stride, int64_t q_head_stride,
    int64_t q_seq_stride, int64_t kv_batch_stride, int64_t kv_head_stride,
    int64_t kv_seq_stride) {
  runTFAImpl<128, 128, kFaCubeS1, kFaTileS1, kFaCvFifoSize, false, true,
             kFaCvFifoConsSyncPeriod>(
      S0, S1, qk_preload, batch, num_q_heads, num_kv_heads, q_batch_stride,
      q_head_stride, q_seq_stride, kv_batch_stride, kv_head_stride,
      kv_seq_stride, reinterpret_cast<__gm__ uint64_t *>(ffts_addr),
      reinterpret_cast<__gm__ half *>(q), reinterpret_cast<__gm__ half *>(k),
      reinterpret_cast<__gm__ half *>(v),
      reinterpret_cast<__gm__ half *>(p_tile_fifo),
      reinterpret_cast<__gm__ float *>(exp_max_ififo), nullptr, nullptr,
      reinterpret_cast<__gm__ float *>(o_out), nullptr,
      reinterpret_cast<__gm__ float *>(qk_tile_fifo),
      reinterpret_cast<__gm__ float *>(pv_tile_fifo), nullptr, nullptr);
}

// Host-callable launch shim. The PyTorch wrapper owns validation, workspace
// allocation, and stream selection; this file keeps the compiler-specific
// kernel launch syntax alongside the device entry points.
#ifndef __COSTMODEL
extern "C" void pto_launch_fa_fp16(
    uint32_t block_dim, void *stream, void *ffts_addr, void *q, void *k,
    void *v, void *p_tile_fifo, void *exp_max_ififo, void *o_out,
    void *qk_tile_fifo, void *pv_tile_fifo, uint32_t s0, uint32_t s1,
    uint32_t qk_preload, uint32_t batch, uint32_t num_q_heads,
    uint32_t num_kv_heads, int64_t q_batch_stride, int64_t q_head_stride,
    int64_t q_seq_stride, int64_t kv_batch_stride, int64_t kv_head_stride,
    int64_t kv_seq_stride, bool causal) {
  if (causal) {
    fa_causal_fp16<<<block_dim, nullptr, stream>>>(
        (__gm__ uint8_t *)ffts_addr, (__gm__ uint8_t *)q, (__gm__ uint8_t *)k,
        (__gm__ uint8_t *)v, (__gm__ uint8_t *)p_tile_fifo,
        (__gm__ uint8_t *)exp_max_ififo, (__gm__ uint8_t *)o_out,
        (__gm__ uint8_t *)qk_tile_fifo, (__gm__ uint8_t *)pv_tile_fifo, s0, s1,
        qk_preload, batch, num_q_heads, num_kv_heads, q_batch_stride,
        q_head_stride, q_seq_stride, kv_batch_stride, kv_head_stride,
        kv_seq_stride);
  } else {
    fa_fp16<<<block_dim, nullptr, stream>>>(
        (__gm__ uint8_t *)ffts_addr, (__gm__ uint8_t *)q, (__gm__ uint8_t *)k,
        (__gm__ uint8_t *)v, (__gm__ uint8_t *)p_tile_fifo,
        (__gm__ uint8_t *)exp_max_ififo, (__gm__ uint8_t *)o_out,
        (__gm__ uint8_t *)qk_tile_fifo, (__gm__ uint8_t *)pv_tile_fifo, s0, s1,
        qk_preload, batch, num_q_heads, num_kv_heads, q_batch_stride,
        q_head_stride, q_seq_stride, kv_batch_stride, kv_head_stride,
        kv_seq_stride);
  }
}
#endif  // __COSTMODEL
