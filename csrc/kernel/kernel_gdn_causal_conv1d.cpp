/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/

#include "kernel_utils.h"

using namespace pto;
using namespace kernel_utils;

#define DIV_ROUNDUP(x, y) (((x) + (y) - 1) / (y))
#define ALIGN_UP(x, y) (DIV_ROUNDUP((x), (y)) * (y))

// (ringSize, maxTileWidth).
// Must match FOR_EACH_RING_SIZE in torch_gdn_causal_conv1d.h.
// A larger ring needs a smaller tile to fit the 192 KiB UB.
#define FOR_EACH_RING_SIZE(DO) \
  DO(2, 4096) DO(4, 3072) DO(8, 1536) DO(16, 896) DO(32, 384) DO(64, 128)

// ===========================================================================
// DEPTHWISE fused causal conv1d + bias + (optional) SiLU  (per-channel, any K)
//
//   y[b,i,c] = act( bias[c] + sum_{k} W[k,c] * x_ext[b, i-K+1+k, c] )
//
// where x_ext[-K+1..-1] = conv_states[b, 0..K-2, c] (when hasConvStates=1)
// or zeros (when hasConvStates=0), and x_ext[0..L-1] = x[b, 0..L-1, c].
//
// Template parameter RS (compile-time, power of two >= K) is the accumulator
// ring size; K is passed at runtime (K <= RS). The host dispatches to the
// RS = roundUpToPow2(K) variant so any K in [2..64] is served.
//
// Weights/bias arrive in IoElemType (fp16/bf16) and are cast to fp32 on
// device. fp16/bf16 I/O, fp32 accumulate.
// ===========================================================================

namespace csilu {

constexpr uint32_t UB_BYTES_PER_CORE = 192u * 1024u;

template <typename TileT>
AICORE inline void applySiluToTile(TileT& dst, TileT& src, TileT& scratch) {
  using ElemType = typename TileT::DType;
  TMULS(scratch, src, (ElemType)-1);
  PipeBarrierVec();
  TEXP(scratch, scratch);
  PipeBarrierVec();
  TADDS(scratch, scratch, (ElemType)1);
  PipeBarrierVec();
  TDIV(dst, src, scratch);
}

// Process ONE work unit: outputs [outputRowStart, outputRowEnd) for channels
// [channelTileBase, channelTileBase+tileChannelCount) of one batch element.
//
// K is the runtime filter width (<= RS). Weights/bias are IoElemType (native
// dtype) and are cast to fp32 on device in a staging area. When hasConvStates
// is set, history rows (j < 0) are read from convStates at seqConvStatesOffset.
template <typename IoElemType, uint32_t RS, uint32_t MAX_W>
AICORE inline void processWorkUnit(
    __gm__ IoElemType* input, __gm__ IoElemType* output,
    __gm__ IoElemType* weights, __gm__ IoElemType* bias,
    __gm__ IoElemType* convStates, uint32_t channels,
    uint64_t sequenceRowOffset, uint64_t seqConvStatesOffset,
    uint32_t channelTileBase, int32_t tileChannelCount, uint32_t outputRowStart,
    uint32_t outputRowEnd, uint32_t K, uint32_t applyActivation,
    uint32_t hasBias, uint32_t hasConvStates) {
  using GlobalShape = pto::Shape<1, 1, 1, 1, DYNAMIC>;
  using GlobalStride = pto::Stride<1, 1, 1, 1, 1>;
  using GlobalIoTensor =
      pto::GlobalTensor<IoElemType, GlobalShape, GlobalStride>;
  using IoTile =
      Tile<TileType::Vec, IoElemType, 1, MAX_W, BLayout::RowMajor, 1, DYNAMIC>;
  using AccumTile =
      Tile<TileType::Vec, float, 1, MAX_W, BLayout::RowMajor, 1, DYNAMIC>;

  constexpr uint32_t accumTileBytes = MAX_W * sizeof(float);
  constexpr uint32_t ioTileBytes = MAX_W * sizeof(IoElemType);

  // UB layout (all offsets compile-time; AT = accumTileBytes):
  //   [0 .. RS-1]           RS fp32 weight tiles              → k * AT
  //   [RS]                  1 fp32 bias tile                  → ubBiasOffset
  //   [RS+1 .. 2*RS]        RS fp32 accumulator ring          → ubAccumRingBase + slot * AT
  //   [2*RS+1 .. 3*RS-1]    RS-1 fp32 partial-product tiles   → ubProductBase + (k-1) * AT
  //   [3*RS]                1 fp32 input scratch              → ubInputFp32
  //   I/O region (4 × ioTileBytes): in[0], out[0], out[1], in[1] → ubIoBase + {0,1,2,3} * ioTileBytes
  constexpr uint32_t ubBiasOffset = RS * accumTileBytes;
  constexpr uint32_t ubAccumRingBase = (RS + 1u) * accumTileBytes;
  constexpr uint32_t ubProductBase = (2u * RS + 1u) * accumTileBytes;
  constexpr uint32_t ubInputFp32 = (3u * RS) * accumTileBytes;
  constexpr uint32_t ubIoBase = (3u * RS + 1u) * accumTileBytes;
  static_assert(
      ubIoBase + 4u * ioTileBytes <= UB_BYTES_PER_CORE,
      "conv1d UB exceeds UB_BYTES_PER_CORE: lower RS/MAX_W or raise it");

  const uint32_t ubOutputOffset[2] = {ubIoBase + ioTileBytes,
                                      ubIoBase + 2u * ioTileBytes};
  const uint32_t ubInputOffset[2] = {ubIoBase, ubIoBase + 3u * ioTileBytes};

  // Staging area for native (IoElemType) weight/bias before casting to fp32.
  // The accumulators+products+xin_f region (ubAccumRingBase onwards) is idle
  // here and is used as scratch. RS+1 io-tiles fit in 2*RS accum-tiles since
  // (RS+1)*sizeof(IoElemType) <= 2*RS*sizeof(float) for any RS >= 1.
  constexpr uint32_t ubStageBase = ubAccumRingBase;
  static_assert(
      (RS + 1u) * ioTileBytes <= 2u * RS * accumTileBytes,
      "conv1d: native weight/bias staging does not fit scratch region");

  // Drain any V activity from a prior task that wrote the same UB staging area,
  // so the MTE2 loads below don't race with that task's output TCVT (WAR
  // hazard).
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

  // Load weights and bias.
  for (uint32_t k = 0; k < K; ++k) {
    GlobalIoTensor wGm(weights + (uint64_t)k * channels + channelTileBase,
                       {tileChannelCount});
    IoTile wStage(tileChannelCount);
    TASSIGN(wStage, ubStageBase + k * ioTileBytes);
    TLOAD(wStage, wGm);
  }
  if (hasBias) {
    GlobalIoTensor bGm(bias + channelTileBase, {tileChannelCount});
    IoTile bStage(tileChannelCount);
    TASSIGN(bStage, ubStageBase + K * ioTileBytes);
    TLOAD(bStage, bGm);
  }
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);

  // Cast weight and bias to fp32 in their final UB positions.
  for (uint32_t k = 0; k < K; ++k) {
    IoTile wStage(tileChannelCount);
    AccumTile wFp32(tileChannelCount);
    TASSIGN(wStage, ubStageBase + k * ioTileBytes);
    TASSIGN(wFp32, k * accumTileBytes);
    TCVT(wFp32, wStage, pto::RoundMode::CAST_NONE);
  }
  if (hasBias) {
    IoTile bStage(tileChannelCount);
    AccumTile bFp32(tileChannelCount);
    TASSIGN(bStage, ubStageBase + K * ioTileBytes);
    TASSIGN(bFp32, ubBiasOffset);
    TCVT(bFp32, bStage, pto::RoundMode::CAST_NONE);
  }
  // The cast TCVTs (V) finish before the first TMUL/TCVT in the loop reuses
  // this region — the double-buffer event flags below enforce the ordering.

  // Double-buffered input: two load slots with independent events.
  // EVENT_ID3 is reused here (the weight/bias load above consumed it already).
  const event_t inputBufferEvent[2] = {EVENT_ID0, EVENT_ID3};
  const event_t outputBufferEvent[2] = {EVENT_ID1, EVENT_ID2};
  // input slots are initially free
  set_flag(PIPE_V, PIPE_MTE2, inputBufferEvent[0]);
  set_flag(PIPE_V, PIPE_MTE2, inputBufferEvent[1]);
  // output slots are initially free
  set_flag(PIPE_MTE3, PIPE_V, outputBufferEvent[0]);
  set_flag(PIPE_MTE3, PIPE_V, outputBufferEvent[1]);

  // Signed loop variable: j < 0 indexes into conv_states (history rows).
  const int32_t halo = (int32_t)K - 1;
  // zeroPad: when there is no initial state, the first K-1 "history" positions
  // are zero; we start the loop at j=0 and treat the accumulators as starting
  // fresh (the startAll shortcut below initialises all of them from j=0).
  const bool zeroPad = (outputRowStart == 0u) && !hasConvStates;
  int32_t jstart;
  if (outputRowStart == 0u)
    jstart = hasConvStates ? -halo : 0;
  else
    jstart = (int32_t)outputRowStart - halo;

  // PROLOGUE: issue the first load before the loop so iteration 0 can prefetch.
  if (jstart < (int32_t)outputRowEnd) {
    IoTile prologueTile(tileChannelCount);
    TASSIGN(prologueTile, ubInputOffset[0]);
    wait_flag(PIPE_V, PIPE_MTE2, inputBufferEvent[0]);
    if (jstart >= 0) {
      GlobalIoTensor rowGm(input + sequenceRowOffset +
                               (uint64_t)jstart * channels + channelTileBase,
                           {tileChannelCount});
      TLOAD(prologueTile, rowGm);
    } else {
      const int32_t stateRow = halo + jstart;
      GlobalIoTensor stateGm(convStates + seqConvStatesOffset +
                               (uint64_t)stateRow * channels + channelTileBase,
                           {tileChannelCount});
      TLOAD(prologueTile, stateGm);
    }
    set_flag(PIPE_MTE2, PIPE_V, inputBufferEvent[0]);
  }

  for (int32_t j = jstart; j < (int32_t)outputRowEnd; ++j) {
    const uint32_t bufferIndex = (uint32_t)(j - jstart) & 1u;
    IoTile inputTileIo(tileChannelCount);
    AccumTile inputTileFp32(tileChannelCount);
    TASSIGN(inputTileIo, ubInputOffset[bufferIndex]);
    TASSIGN(inputTileFp32, ubInputFp32);

    // (1) Consume the row loaded by the prologue / previous prefetch.
    wait_flag(PIPE_MTE2, PIPE_V, inputBufferEvent[bufferIndex]);
    TCVT(inputTileFp32, inputTileIo, pto::RoundMode::CAST_NONE);
    set_flag(PIPE_V, PIPE_MTE2,
             inputBufferEvent[bufferIndex]);  // slot free again

    // (2) Prefetch the next row into the other buffer.
    if (j + 1 < (int32_t)outputRowEnd) {
      const uint32_t nextBuf = bufferIndex ^ 1u;
      const int32_t jnext = j + 1;
      IoTile nextTile(tileChannelCount);
      TASSIGN(nextTile, ubInputOffset[nextBuf]);
      wait_flag(PIPE_V, PIPE_MTE2, inputBufferEvent[nextBuf]);
      if (jnext >= 0) {
        GlobalIoTensor rowGm(input + sequenceRowOffset +
                                 (uint64_t)jnext * channels + channelTileBase,
                             {tileChannelCount});
        TLOAD(nextTile, rowGm);
      } else {
        const int32_t stateRow = halo + jnext;
        GlobalIoTensor stateGm(convStates + seqConvStatesOffset +
                                 (uint64_t)stateRow * channels +
                                 channelTileBase,
                             {tileChannelCount});
        TLOAD(nextTile, stateGm);
      }
      set_flag(PIPE_MTE2, PIPE_V, inputBufferEvent[nextBuf]);
    }

    PipeBarrierVec();

    // Scatter: row j contributes to up to K output rows. When zeroPad && j==0
    // all K accumulators are uninitialized, so we write them directly (no
    // product buffer needed).
    const bool startAll = zeroPad && (j == 0);
    for (uint32_t tapIndex = 0; tapIndex < K; ++tapIndex) {
      const int32_t outputRow = j + halo - (int32_t)tapIndex;
      if (outputRow < (int32_t)outputRowStart ||
          outputRow >= (int32_t)outputRowEnd)
        continue;
      AccumTile weightTile(tileChannelCount);
      TASSIGN(weightTile, tapIndex * accumTileBytes);
      if (startAll || tapIndex == 0) {
        AccumTile accumTile(tileChannelCount);
        TASSIGN(accumTile, ubAccumRingBase + ((uint32_t)outputRow & (RS - 1u)) *
                                                 accumTileBytes);
        TMUL(accumTile, inputTileFp32, weightTile);
      } else {
        AccumTile productTile(tileChannelCount);
        TASSIGN(productTile, ubProductBase + (tapIndex - 1u) * accumTileBytes);
        TMUL(productTile, inputTileFp32, weightTile);
      }
    }
    PipeBarrierVec();
    if (!startAll) {
      for (uint32_t tapIndex = 1; tapIndex < K; ++tapIndex) {
        const int32_t outputRow = j + halo - (int32_t)tapIndex;
        if (outputRow < (int32_t)outputRowStart ||
            outputRow >= (int32_t)outputRowEnd)
          continue;
        AccumTile accumTile(tileChannelCount);
        AccumTile productTile(tileChannelCount);
        TASSIGN(accumTile, ubAccumRingBase + ((uint32_t)outputRow & (RS - 1u)) *
                                                 accumTileBytes);
        TASSIGN(productTile, ubProductBase + (tapIndex - 1u) * accumTileBytes);
        TADD(accumTile, accumTile, productTile);
      }
    }
    PipeBarrierVec();

    if (j < (int32_t)outputRowStart)
      continue;  // halo row: primed accumulators only

    const uint32_t accumRingSlot = (uint32_t)j & (RS - 1u);
    const uint32_t outputBufIndex = (uint32_t)j & 1u;
    AccumTile accumTile(tileChannelCount);
    AccumTile siluScratch(tileChannelCount);
    IoTile outputTile(tileChannelCount);
    TASSIGN(accumTile, ubAccumRingBase + accumRingSlot * accumTileBytes);
    TASSIGN(siluScratch, ubProductBase);  // reuses product slot 0 — safe: scatter loop is done
    TASSIGN(outputTile, ubOutputOffset[outputBufIndex]);

    if (hasBias) {
      AccumTile bFp32(tileChannelCount);
      TASSIGN(bFp32, ubBiasOffset);
      TADD(accumTile, accumTile, bFp32);
      PipeBarrierVec();
    }
    if (applyActivation) {
      applySiluToTile(accumTile, accumTile, siluScratch);
      PipeBarrierVec();
    }
    wait_flag(PIPE_MTE3, PIPE_V, outputBufferEvent[outputBufIndex]);
    TCVT(outputTile, accumTile, pto::RoundMode::CAST_NONE);
    GlobalIoTensor outGm(
        output + sequenceRowOffset + (uint64_t)j * channels + channelTileBase,
        {tileChannelCount});
    set_flag(PIPE_V, PIPE_MTE3, outputBufferEvent[outputBufIndex]);
    wait_flag(PIPE_V, PIPE_MTE3, outputBufferEvent[outputBufIndex]);
    TSTORE(outGm, outputTile);
    set_flag(PIPE_MTE3, PIPE_V, outputBufferEvent[outputBufIndex]);
  }

  wait_flag(PIPE_V, PIPE_MTE2, inputBufferEvent[0]);
  wait_flag(PIPE_V, PIPE_MTE2, inputBufferEvent[1]);
  wait_flag(PIPE_MTE3, PIPE_V, outputBufferEvent[0]);
  wait_flag(PIPE_MTE3, PIPE_V, outputBufferEvent[1]);
}

// Batched driver: assigns work units to cores and calls processWorkUnit.
template <typename IoElemType, uint32_t RS, uint32_t MAX_W>
AICORE void runConvSiluBatched(
    __gm__ IoElemType* input, __gm__ IoElemType* output,
    __gm__ IoElemType* weights, __gm__ IoElemType* bias,
    __gm__ IoElemType* convStates, uint32_t batch, uint32_t seqLen,
    uint32_t channels, uint32_t stateLen, uint32_t K, uint32_t applyActivation,
    uint32_t hasBias, uint32_t hasConvStates) {
  static_assert(RS >= 2u, "RS (ring size) must be >= 2");

  set_mask_norm();
  set_vector_mask(-1, -1);

  const uint32_t coreCount = get_block_num();
  const uint32_t coreIndex = get_block_idx();
  if (seqLen == 0 || batch == 0 || channels == 0) return;

  // Minimum rows per sequence chunk. Smaller values expose more parallelism but
  // each chunk replays K-1 halo rows; 32 balances the two effects.
  constexpr uint32_t minSeqChunkLen = 32u < RS ? RS : 32u;
  // One aligned vector lane = 256 bytes / element size (128 for fp16/bf16).
  constexpr uint32_t channelsPerLane = 256u / sizeof(IoElemType);

  uint32_t workUnitsPerSequence = DIV_ROUNDUP(coreCount, batch);
  if (workUnitsPerSequence < 1) workUnitsPerSequence = 1;

  uint32_t maxSequenceChunks = DIV_ROUNDUP(seqLen, minSeqChunkLen);

  const uint32_t channelTilesForUbLimit = DIV_ROUNDUP(channels, MAX_W);
  const uint32_t channelTilesToFillCores =
      DIV_ROUNDUP(workUnitsPerSequence, maxSequenceChunks);
  uint32_t channelTileCount = channelTilesForUbLimit > channelTilesToFillCores
                                  ? channelTilesForUbLimit
                                  : channelTilesToFillCores;

  const uint32_t maxChannelTilesByLane = DIV_ROUNDUP(channels, channelsPerLane);
  if (channelTileCount > maxChannelTilesByLane)
    channelTileCount = maxChannelTilesByLane;

  uint32_t channelTileWidth =
      ALIGN_UP(DIV_ROUNDUP(channels, channelTileCount), channelsPerLane);
  if (channelTileWidth < channelsPerLane) channelTileWidth = channelsPerLane;
  if (channelTileWidth > MAX_W) channelTileWidth = MAX_W;
  if (channelTileWidth > channels) channelTileWidth = channels;

  channelTileCount = DIV_ROUNDUP(channels, channelTileWidth);
  uint32_t sequenceChunkCount =
      DIV_ROUNDUP(workUnitsPerSequence, channelTileCount);
  if (sequenceChunkCount > maxSequenceChunks)
    sequenceChunkCount = maxSequenceChunks;

  const uint32_t sequenceChunkLength = DIV_ROUNDUP(seqLen, sequenceChunkCount);
  const uint32_t totalWorkUnits = batch * channelTileCount * sequenceChunkCount;

  for (uint32_t workUnitIndex = coreIndex; workUnitIndex < totalWorkUnits;
       workUnitIndex += coreCount) {
    const uint32_t sequenceChunkIndex = workUnitIndex % sequenceChunkCount;
    const uint32_t channelBatchIndex = workUnitIndex / sequenceChunkCount;
    const uint32_t channelTileIndex = channelBatchIndex % channelTileCount;
    const uint32_t batchIndex = channelBatchIndex / channelTileCount;

    const uint32_t channelTileBase = channelTileIndex * channelTileWidth;
    const uint32_t remainingChannels = channels - channelTileBase;
    const int32_t tileChannelCount =
        (int32_t)(remainingChannels > channelTileWidth ? channelTileWidth
                                                       : remainingChannels);
    const uint32_t outputRowStart = sequenceChunkIndex * sequenceChunkLength;
    if (outputRowStart >= seqLen) continue;
    uint32_t outputRowEnd = outputRowStart + sequenceChunkLength;
    if (outputRowEnd > seqLen) outputRowEnd = seqLen;

    const uint64_t sequenceRowOffset = (uint64_t)batchIndex * seqLen * channels;
    const uint64_t seqConvStatesOffset =
        (uint64_t)batchIndex * stateLen * channels;

    processWorkUnit<IoElemType, RS, MAX_W>(
        input, output, weights, bias, convStates, channels, sequenceRowOffset,
        seqConvStatesOffset, channelTileBase, tileChannelCount, outputRowStart,
        outputRowEnd, K, applyActivation, hasBias, hasConvStates);
  }
}

}  // namespace csilu

// clang-format off
#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t*
#endif
// clang-format on

// Kernel entry parameters: 5 GM pointers + 8 uint32 scalars.
#define BATCHED_PARAMS                                                        \
  GM_ADDR input, GM_ADDR output, GM_ADDR weights, GM_ADDR bias,               \
      GM_ADDR convStates, uint32_t batch, uint32_t seqLen, uint32_t channels, \
      uint32_t stateLen, uint32_t K, uint32_t applyActivation,                \
      uint32_t hasBias, uint32_t hasConvStates

#if defined(__DAV_VEC__)
#define BATCHED_BODY(T, RS, MW)                                        \
  csilu::runConvSiluBatched<T, RS, MW>(                                \
      (__gm__ T*)input, (__gm__ T*)output, (__gm__ T*)weights,         \
      (__gm__ T*)bias, (__gm__ T*)convStates, batch, seqLen, channels, \
      stateLen, K, applyActivation, hasBias, hasConvStates)
#else  // cube pass: empty bodies; void the params to silence unused-variable
       // warnings.
#define BATCHED_BODY(T, RS, MW)                                           \
  (void)input, (void)output, (void)weights, (void)bias, (void)convStates, \
      (void)batch, (void)seqLen, (void)channels, (void)stateLen, (void)K, \
      (void)applyActivation, (void)hasBias, (void)hasConvStates
#endif

// clang-format off
#define DEF_ENTRY(SUF, T, RS, MW)                                               \
  extern "C" __global__ AICORE void gdn_causal_conv1d_##SUF(BATCHED_PARAMS) {   \
    BATCHED_BODY(T, RS, MW);                                                    \
  }
#define DEFINE_ENTRIES(ringSize, maxTileWidth)                        \
  DEF_ENTRY(fp16_rs##ringSize, half,        ringSize, maxTileWidth)   \
  DEF_ENTRY(bf16_rs##ringSize, bfloat16_t,  ringSize, maxTileWidth)
FOR_EACH_RING_SIZE(DEFINE_ENTRIES)
#undef DEFINE_ENTRIES
#undef DEF_ENTRY
#undef BATCHED_BODY
#undef BATCHED_PARAMS
// clang-format on
