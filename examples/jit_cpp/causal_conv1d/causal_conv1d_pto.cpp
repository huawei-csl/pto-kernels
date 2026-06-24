#include <pto/pto-inst.hpp>

using namespace pto;

#define DIV_ROUNDUP(x, y) (((x) + (y) - 1) / (y))
#define ALIGN_UP(x, y) (DIV_ROUNDUP((x), (y)) * (y))

// ===========================================================================
// DEPTHWISE fused causal conv1d + bias + (optional) SiLU  (per-channel, any K)
//
//   y[b,i,c] = act( bias[c] + sum_{k=max(0,K-1-i)..K-1} W[k,c] * x[b, i-K+1+k,
//   c] ), x[<0]=0
//
// Per-channel K-tap filter (Mamba/GDN short conv). x,y are [batch, seqLen,
// channels] row-major; `channels` is the lane axis, seqLen the conv axis.
// Weights W[K,channels] + bias[channels] are fp32 GM tensors. fp16 OR bf16 I/O,
// fp32 accumulate.
//
// Filter width K and per-tile channel width MAX_W are compile-time constants
// chosen at the call site as template parameters (no preprocessor config), e.g.
//   constexpr uint32_t K = CAUSAL_CONV_K, MAX_W = CAUSAL_CONV_MAX_W;
//   csilu::runConvSiluBatched<bfloat16_t, float, K, MAX_W>(...);
//
// 2-D-plus-batch work grid: workUnits = batch x sequenceChunkCount x
// channelTileCount. Each work unit produces outputs [batchIndex] x
// [outputRowStart,outputRowEnd) for channels
// [channelTileBase,channelTileBase+tileChannelCount), replaying K-1 causal halo
// rows to prime its accumulators. The grid fills all cores: batch supplies
// parallelism first, then channel tiles, then sequence chunks; the channel-tile
// width is widened to whole vector lanes for coalesced stores. (See
// processWorkUnit + runConvSiluBatched.)
//
// Generic in K (filter width) and MAX_W (tile width). accumRingSize = smallest
// power of two >= K is the accumulator ring size, so the K outputs in flight
// map to distinct ring slots via `idx & accumRingMask` (accumRingMask =
// accumRingSize - 1).
// UB layout (per lane, fp32 unless noted): K weights + bias(1) + accumRingSize
// accumulators + (K-1) partial products + input-as-fp32(1) =
// 2*K+accumRingSize+1 fp32 tiles; then the I/O region inputTile[0..1] + output0
// + output1 = 4 I/O tiles (input load double-buffered). A static_assert keeps
// the total within UB_BYTES_PER_CORE for the chosen K / MAX_W / dtypes. NOTE:
// uses the PTO tile-op API (<pto/pto-inst.hpp>); the `csilu` namespace avoids a
// clash with pto::detail.
// ===========================================================================

namespace csilu {

// Unified Buffer available per AIV core (Ascend 910B2 = 192 KiB). The UB
// static_assert in processWorkUnit checks the chosen layout fits; raise it for
// a next-gen NPU with a larger UB.
constexpr uint32_t UB_BYTES_PER_CORE = 192u * 1024u;

// Smallest power of two >= value.
AICORE constexpr uint32_t roundUpToPowerOfTwo(uint32_t value) {
  if (value != 0u) --value;

  value |= (value >> 1u);
  value |= (value >> 2u);
  value |= (value >> 4u);
  value |= (value >> 8u);
  value |= (value >> 16u);

  ++value;

  return value;
}

template <typename TileT>
AICORE inline void applySiluToTile(TileT& dst, TileT& src, TileT& scratch) {
  using ElemType = typename TileT::DType;
  TMULS(scratch, src, (ElemType)-1);
  pipe_barrier(PIPE_V);
  TEXP(scratch, scratch);
  pipe_barrier(PIPE_V);
  TADDS(scratch, scratch, (ElemType)1);
  pipe_barrier(PIPE_V);
  TDIV(dst, src, scratch);
}

// Process ONE work unit: outputs [outputRowStart,outputRowEnd) for channels
// [channelTileBase,channelTileBase+tileChannelCount) of the sequence whose
// first row is at element offset sequenceRowOffset. x[<0]=0 (no cache).
template <typename IoElemType, typename AccumElemType, uint32_t K,
          uint32_t MAX_W>
AICORE inline void processWorkUnit(
    __gm__ IoElemType* input, __gm__ IoElemType* output,
    __gm__ AccumElemType* weights, __gm__ AccumElemType* bias,
    uint32_t channels, uint64_t sequenceRowOffset, uint32_t channelTileBase,
    int32_t tileChannelCount, uint32_t outputRowStart, uint32_t outputRowEnd,
    uint32_t applyActivation) {
  using GlobalShape = pto::Shape<1, 1, 1, 1, DYNAMIC>;
  using GlobalStride = pto::Stride<1, 1, 1, 1, 1>;
  using GlobalIoTensor =
      pto::GlobalTensor<IoElemType, GlobalShape, GlobalStride>;
  using GlobalAccumTensor =
      pto::GlobalTensor<AccumElemType, GlobalShape, GlobalStride>;
  using IoTile =
      Tile<TileType::Vec, IoElemType, 1, MAX_W, BLayout::RowMajor, 1, DYNAMIC>;
  using AccumTile = Tile<TileType::Vec, AccumElemType, 1, MAX_W,
                         BLayout::RowMajor, 1, DYNAMIC>;

  constexpr uint32_t accumTileBytes = MAX_W * sizeof(AccumElemType);
  constexpr uint32_t ioTileBytes = MAX_W * sizeof(IoElemType);
  // accumulator ring (power of two >= K) so the K in-flight outputs never
  // alias.
  constexpr uint32_t accumRingSize = roundUpToPowerOfTwo(K);
  constexpr uint32_t accumRingMask =
      accumRingSize - 1u;  // ring-slot index mask
  static_assert(K <= accumRingSize, "accumulator ring must hold all K taps");

  // UB byte offsets. fp32 region: K weights (weight k at k*accumTileBytes) |
  // bias | accumRingSize accumulators | K-1 partial products | input-as-fp32.
  // Then the I/O region: 4 ioTileBytes-sized tiles (input load
  // double-buffered).
  constexpr uint32_t ubBiasOffset = K * accumTileBytes;
  constexpr uint32_t ubAccumRingBase = (K + 1u) * accumTileBytes;
  // partial product for tap k at ubProductBase + (k-1)*accumTileBytes; also
  // reused as the SiLU scratch tile once the products have been summed.
  constexpr uint32_t ubProductBase = (K + 1u + accumRingSize) * accumTileBytes;
  constexpr uint32_t ubInputFp32Offset =
      (2u * K + accumRingSize) * accumTileBytes;
  constexpr uint32_t ubIoRegionBase =
      (2u * K + accumRingSize + 1u) * accumTileBytes;
  static_assert(
      ubIoRegionBase + 4u * ioTileBytes <= UB_BYTES_PER_CORE,
      "conv1d UB exceeds UB_BYTES_PER_CORE: lower K/MAX_W or raise it");

  constexpr uint32_t ubOutputOffset[2] = {ubIoRegionBase + ioTileBytes,
                                          ubIoRegionBase + 2u * ioTileBytes};
  // input double-buffer: slot 0 before the outputs, slot 1 after.
  constexpr uint32_t ubInputOffset[2] = {ubIoRegionBase,
                                         ubIoRegionBase + 3u * ioTileBytes};

  const uint32_t firstInputRow =
      (outputRowStart > (K - 1)) ? (outputRowStart - (K - 1)) : 0u;

  // ---- per-channel weights + bias (resident for this work unit) ----
  for (uint32_t tapIndex = 0; tapIndex < K; ++tapIndex) {
    GlobalAccumTensor weightGm(
        weights + (uint64_t)tapIndex * channels + channelTileBase,
        {tileChannelCount});
    AccumTile weightTile(tileChannelCount);
    TASSIGN(weightTile, tapIndex * accumTileBytes);
    TLOAD(weightTile, weightGm);
  }
  {
    GlobalAccumTensor biasGm(bias + channelTileBase, {tileChannelCount});
    AccumTile biasTile(tileChannelCount);
    TASSIGN(biasTile, ubBiasOffset);
    TLOAD(biasTile, biasGm);
  }
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);

  // double-buffered input: two load slots with independent handshakes.
  // EVENT_ID3 is reused here (the weight load above already consumed it).
  // events are passed by mutable value to set_flag/wait_flag, so not constexpr.
  const event_t inputBufferEvent[2] = {EVENT_ID0, EVENT_ID3};
  set_flag(PIPE_V, PIPE_MTE2,
           inputBufferEvent[0]);  // input slot 0 initially free
  set_flag(PIPE_V, PIPE_MTE2,
           inputBufferEvent[1]);  // input slot 1 initially free
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);

  // PROLOGUE: issue the first input load (firstInputRow) so iteration 0 can
  // prefetch the next row.
  if (firstInputRow < outputRowEnd) {
    GlobalIoTensor inputRowGm(input + sequenceRowOffset +
                                  (uint64_t)firstInputRow * channels +
                                  channelTileBase,
                              {tileChannelCount});
    IoTile prologueInputTile(tileChannelCount);
    TASSIGN(prologueInputTile, ubInputOffset[0]);
    wait_flag(PIPE_V, PIPE_MTE2, inputBufferEvent[0]);
    TLOAD(prologueInputTile, inputRowGm);
    set_flag(PIPE_MTE2, PIPE_V, inputBufferEvent[0]);
  }

  for (uint32_t inputRow = firstInputRow; inputRow < outputRowEnd; ++inputRow) {
    const uint32_t bufferIndex = (inputRow - firstInputRow) & 1u;
    IoTile inputTileIo(tileChannelCount);
    AccumTile inputTileFp32(tileChannelCount);
    TASSIGN(inputTileIo, ubInputOffset[bufferIndex]);
    TASSIGN(inputTileFp32, ubInputFp32Offset);

    // (1) consume current row from buffer bufferIndex (loaded by the prologue /
    // previous prefetch).
    wait_flag(PIPE_MTE2, PIPE_V, inputBufferEvent[bufferIndex]);
    TCVT(inputTileFp32, inputTileIo, pto::RoundMode::CAST_NONE);
    set_flag(PIPE_V, PIPE_MTE2,
             inputBufferEvent[bufferIndex]);  // slot free again

    // (2) prefetch next row into the OTHER buffer; overlaps the compute below
    // without waiting on the buffer consumed this iteration.
    if (inputRow + 1 < outputRowEnd) {
      const uint32_t nextBufferIndex = bufferIndex ^ 1u;
      IoTile nextInputTile(tileChannelCount);
      TASSIGN(nextInputTile, ubInputOffset[nextBufferIndex]);
      GlobalIoTensor nextInputRowGm(input + sequenceRowOffset +
                                        (uint64_t)(inputRow + 1) * channels +
                                        channelTileBase,
                                    {tileChannelCount});
      wait_flag(PIPE_V, PIPE_MTE2, inputBufferEvent[nextBufferIndex]);
      TLOAD(nextInputTile, nextInputRowGm);
      set_flag(PIPE_MTE2, PIPE_V, inputBufferEvent[nextBufferIndex]);
    }

    pipe_barrier(PIPE_V);

    // scatter: this input row contributes to K output rows; form each
    // weight*input product (only for outputs in [outputRowStart,outputRowEnd)).
    for (uint32_t tapIndex = 0; tapIndex < K; ++tapIndex) {
      const uint32_t outputRow = inputRow + (K - 1) - tapIndex;
      if (outputRow < outputRowStart || outputRow >= outputRowEnd) continue;
      AccumTile weightTile(tileChannelCount);
      TASSIGN(weightTile, tapIndex * accumTileBytes);
      if (inputRow == 0 || tapIndex == 0) {
        // first contribution to this output's accumulator slot: initialise it.
        AccumTile accumTile(tileChannelCount);
        TASSIGN(accumTile,
                ubAccumRingBase + (outputRow & accumRingMask) * accumTileBytes);
        TMUL(accumTile, inputTileFp32, weightTile);
      } else {
        AccumTile productTile(tileChannelCount);
        TASSIGN(productTile, ubProductBase + (tapIndex - 1u) * accumTileBytes);
        TMUL(productTile, inputTileFp32, weightTile);
      }
    }
    pipe_barrier(PIPE_V);
    if (inputRow != 0) {
      for (uint32_t tapIndex = 1; tapIndex < K; ++tapIndex) {
        const uint32_t outputRow = inputRow + (K - 1) - tapIndex;
        if (outputRow < outputRowStart || outputRow >= outputRowEnd) continue;
        AccumTile accumTile(tileChannelCount);
        AccumTile productTile(tileChannelCount);
        TASSIGN(accumTile,
                ubAccumRingBase + (outputRow & accumRingMask) * accumTileBytes);
        TASSIGN(productTile, ubProductBase + (tapIndex - 1u) * accumTileBytes);
        TADD(accumTile, accumTile, productTile);
      }
    }
    pipe_barrier(PIPE_V);

    if (inputRow < outputRowStart)
      continue;  // halo row: primed accumulators only

    const uint32_t accumRingSlot = inputRow & accumRingMask;
    const uint32_t outputBufferIndex = inputRow & 1u;
    const event_t outputBufferEvent = (event_t)(1u + outputBufferIndex);
    AccumTile accumTile(tileChannelCount);
    AccumTile biasTile(tileChannelCount);
    AccumTile siluScratchTile(tileChannelCount);
    IoTile outputTile(tileChannelCount);
    TASSIGN(accumTile, ubAccumRingBase + accumRingSlot * accumTileBytes);
    TASSIGN(biasTile, ubBiasOffset);
    TASSIGN(siluScratchTile, ubProductBase);
    TASSIGN(outputTile, ubOutputOffset[outputBufferIndex]);

    TADD(accumTile, accumTile, biasTile);
    pipe_barrier(PIPE_V);
    if (applyActivation) {
      applySiluToTile(accumTile, accumTile, siluScratchTile);
      pipe_barrier(PIPE_V);
    }
    wait_flag(PIPE_MTE3, PIPE_V, outputBufferEvent);
    TCVT(outputTile, accumTile, pto::RoundMode::CAST_NONE);

    GlobalIoTensor outputRowGm(output + sequenceRowOffset +
                                   (uint64_t)inputRow * channels +
                                   channelTileBase,
                               {tileChannelCount});
    set_flag(PIPE_V, PIPE_MTE3, outputBufferEvent);
    wait_flag(PIPE_V, PIPE_MTE3, outputBufferEvent);
    TSTORE(outputRowGm, outputTile);
    set_flag(PIPE_MTE3, PIPE_V, outputBufferEvent);
  }

  wait_flag(PIPE_V, PIPE_MTE2, inputBufferEvent[0]);
  wait_flag(PIPE_V, PIPE_MTE2, inputBufferEvent[1]);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
}

template <typename IoElemType, typename AccumElemType, uint32_t K,
          uint32_t MAX_W>
AICORE void runConvSiluBatched(__gm__ IoElemType* input,
                               __gm__ IoElemType* output,
                               __gm__ AccumElemType* weights,
                               __gm__ AccumElemType* bias, uint32_t batch,
                               uint32_t seqLen, uint32_t channels,
                               uint32_t applyActivation) {
  static_assert(K >= 1u, "K (filter width) must be >= 1");

  set_mask_norm();
  set_vector_mask(-1, -1);

  const uint32_t coreCount = get_block_num();
  const uint32_t coreIndex = get_block_idx();
  if (seqLen == 0 || batch == 0 || channels == 0) return;

  // ---- grid tuning knobs ----
  // Minimum rows per sequence-chunk. Smaller chunks expose more parallelism but
  // each replays K-1 causal halo rows, so very small chunks waste compute; 32
  // is the balance point across the GDN sequence lengths.
  constexpr uint32_t minSeqChunkLen = 32u;
  // Channels per aligned vector lane = 256 B / element size (128 for
  // fp16/bf16). Channel tiles are sized in whole lanes for coalesced
  // loads/stores.
  constexpr uint32_t channelsPerLane = 256u / sizeof(IoElemType);

  // batch supplies parallelism first; each sequence then needs
  // workUnitsPerSequence (channel-tile x seq-chunk) units to keep all cores
  // busy.
  uint32_t workUnitsPerSequence = DIV_ROUNDUP(coreCount, batch);
  if (workUnitsPerSequence < 1) workUnitsPerSequence = 1;

  uint32_t maxSequenceChunks = DIV_ROUNDUP(seqLen, minSeqChunkLen);
  if (maxSequenceChunks < 1) maxSequenceChunks = 1;

  // channel tiles: enough that each tile fits MAX_W, and enough (with the chunk
  // count) to fill the per-sequence work-unit budget.
  const uint32_t channelTilesForUbLimit = DIV_ROUNDUP(channels, MAX_W);
  const uint32_t channelTilesToFillCores =
      DIV_ROUNDUP(workUnitsPerSequence, maxSequenceChunks);
  uint32_t channelTileCount = channelTilesForUbLimit > channelTilesToFillCores
                                  ? channelTilesForUbLimit
                                  : channelTilesToFillCores;

  const uint32_t maxChannelTilesByLane = DIV_ROUNDUP(channels, channelsPerLane);
  if (channelTileCount > maxChannelTilesByLane)
    channelTileCount = maxChannelTilesByLane;
  if (channelTileCount < 1) channelTileCount = 1;

  uint32_t channelTileWidth =
      ALIGN_UP(DIV_ROUNDUP(channels, channelTileCount), channelsPerLane);
  if (channelTileWidth < channelsPerLane) channelTileWidth = channelsPerLane;
  if (channelTileWidth > MAX_W) channelTileWidth = MAX_W;
  if (channelTileWidth > channels) channelTileWidth = channels;

  channelTileCount = DIV_ROUNDUP(channels, channelTileWidth);
  uint32_t sequenceChunkCount =
      DIV_ROUNDUP(workUnitsPerSequence, channelTileCount);
  if (sequenceChunkCount < 1) sequenceChunkCount = 1;
  if (sequenceChunkCount > maxSequenceChunks)
    sequenceChunkCount = maxSequenceChunks;

  const uint32_t sequenceChunkLength = DIV_ROUNDUP(seqLen, sequenceChunkCount);

  // Iterate the convolution (sequence) direction in the middle so input rows
  // and weights stay resident across chunks of the same channel tile.
  const uint32_t totalWorkUnits = batch * channelTileCount * sequenceChunkCount;
  for (uint32_t workUnitIndex = coreIndex; workUnitIndex < totalWorkUnits;
       workUnitIndex += coreCount) {
    const uint32_t sequenceChunkIndex = workUnitIndex % sequenceChunkCount;
    const uint32_t tileBatchIndex = workUnitIndex / sequenceChunkCount;
    const uint32_t channelTileIndex = tileBatchIndex % channelTileCount;
    const uint32_t batchIndex = tileBatchIndex / channelTileCount;
    const uint32_t channelTileBase = channelTileIndex * channelTileWidth;
    const uint32_t remainingChannels = channels - channelTileBase;
    const int32_t tileChannelCount = remainingChannels > channelTileWidth
                                         ? (int32_t)channelTileWidth
                                         : (int32_t)remainingChannels;
    const uint32_t outputRowStart = sequenceChunkIndex * sequenceChunkLength;
    if (outputRowStart >= seqLen) continue;
    uint32_t outputRowEnd = outputRowStart + sequenceChunkLength;
    if (outputRowEnd > seqLen) outputRowEnd = seqLen;
    const uint64_t sequenceRowOffset = (uint64_t)batchIndex * seqLen * channels;
    processWorkUnit<IoElemType, AccumElemType, K, MAX_W>(
        input, output, weights, bias, channels, sequenceRowOffset,
        channelTileBase, tileChannelCount, outputRowStart, outputRowEnd,
        applyActivation);
  }
}

}  // namespace csilu

// Filter width / per-tile channel width the entry points below are compiled at.
// Default to the K=4, MAX_W=3072 production configuration; the test suite
// recompiles this file at other widths via -DCAUSAL_CONV_K / -DCAUSAL_CONV_MAX_W
// (see test_causal_conv1d.py), so a plain build is byte-for-byte unaffected.
#ifndef CAUSAL_CONV_K
#define CAUSAL_CONV_K 4
#endif
#ifndef CAUSAL_CONV_MAX_W
#define CAUSAL_CONV_MAX_W 3072
#endif

// ---- single-sequence entry (back-compat: input,output [seqLen,channels] fp16,
// weights[K,channels]/bias[channels] fp32) ----
extern "C" __global__ AICORE void causal_conv1d_kernel(
    __gm__ uint8_t* input, __gm__ uint8_t* output, __gm__ uint8_t* weights,
    __gm__ uint8_t* bias, uint32_t seqLen, uint32_t channels) {
#if defined(__DAV_VEC__)
  constexpr uint32_t K = CAUSAL_CONV_K, MAX_W = CAUSAL_CONV_MAX_W;
  csilu::runConvSiluBatched<half, float, K, MAX_W>(
      (__gm__ half*)input, (__gm__ half*)output, (__gm__ float*)weights,
      (__gm__ float*)bias, 1u, seqLen, channels, 1u);
#else
  (void)input;
  (void)output;
  (void)weights;
  (void)bias;
  (void)seqLen;
  (void)channels;
#endif
}

// ---- batched fp16 entry: input,output [batch,seqLen,channels] fp16,
// weights/bias fp32 ----
extern "C" __global__ AICORE void causal_conv1d_batched_kernel(
    __gm__ uint8_t* input, __gm__ uint8_t* output, __gm__ uint8_t* weights,
    __gm__ uint8_t* bias, uint32_t batch, uint32_t seqLen, uint32_t channels,
    uint32_t applyActivation) {
#if defined(__DAV_VEC__)
  constexpr uint32_t K = CAUSAL_CONV_K, MAX_W = CAUSAL_CONV_MAX_W;
  csilu::runConvSiluBatched<half, float, K, MAX_W>(
      (__gm__ half*)input, (__gm__ half*)output, (__gm__ float*)weights,
      (__gm__ float*)bias, batch, seqLen, channels, applyActivation);
#else
  (void)input;
  (void)output;
  (void)weights;
  (void)bias;
  (void)batch;
  (void)seqLen;
  (void)channels;
  (void)applyActivation;
#endif
}

// ---- batched bf16 entry: input,output [batch,seqLen,channels] bf16,
// weights/bias fp32 ----
extern "C" __global__ AICORE void causal_conv1d_batched_bf16_kernel(
    __gm__ uint8_t* input, __gm__ uint8_t* output, __gm__ uint8_t* weights,
    __gm__ uint8_t* bias, uint32_t batch, uint32_t seqLen, uint32_t channels,
    uint32_t applyActivation) {
#if defined(__DAV_VEC__)
  constexpr uint32_t K = CAUSAL_CONV_K, MAX_W = CAUSAL_CONV_MAX_W;
  csilu::runConvSiluBatched<bfloat16_t, float, K, MAX_W>(
      (__gm__ bfloat16_t*)input, (__gm__ bfloat16_t*)output,
      (__gm__ float*)weights, (__gm__ float*)bias, batch, seqLen, channels,
      applyActivation);
#else
  (void)input;
  (void)output;
  (void)weights;
  (void)bias;
  (void)batch;
  (void)seqLen;
  (void)channels;
  (void)applyActivation;
#endif
}

extern "C" void call_kernel(uint32_t blockDim, void* stream, uint8_t* input,
                            uint8_t* output, uint8_t* weights, uint8_t* bias,
                            uint32_t seqLen, uint32_t channels) {
  causal_conv1d_kernel<<<blockDim * 2, nullptr, stream>>>(
      input, output, weights, bias, seqLen, channels);
}

extern "C" void call_kernel_batched(uint32_t blockDim, void* stream,
                                    uint8_t* input, uint8_t* output,
                                    uint8_t* weights, uint8_t* bias,
                                    uint32_t batch, uint32_t seqLen,
                                    uint32_t channels,
                                    uint32_t applyActivation) {
  causal_conv1d_batched_kernel<<<blockDim * 2, nullptr, stream>>>(
      input, output, weights, bias, batch, seqLen, channels, applyActivation);
}

extern "C" void call_kernel_batched_bf16(uint32_t blockDim, void* stream,
                                         uint8_t* input, uint8_t* output,
                                         uint8_t* weights, uint8_t* bias,
                                         uint32_t batch, uint32_t seqLen,
                                         uint32_t channels,
                                         uint32_t applyActivation) {
  causal_conv1d_batched_bf16_kernel<<<blockDim * 2, nullptr, stream>>>(
      input, output, weights, bias, batch, seqLen, channels, applyActivation);
}
