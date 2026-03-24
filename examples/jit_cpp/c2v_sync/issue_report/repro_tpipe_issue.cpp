// Reproducer: TPipe C2V producer path mismatch for this C2V case.
//
// Purpose:
// - Show that TPipe call style is convenient, but for this case we need producer
//   sync on PIPE_MTE3 to match reference behavior.
//
// Notes:
// - This file is for issue documentation.
// - It focuses on sync behavior, not full data movement.

#include <pto/pto-inst.hpp>
#include <pto/npu/a2a3/TPush.hpp>
#include <pto/npu/a2a3/TPop.hpp>
#include "runtime/rt.h"

using namespace pto;

#ifndef REPRO_USE_NATIVE_TPIPE_RECORD
#define REPRO_USE_NATIVE_TPIPE_RECORD 0
#endif

template <uint8_t FlagID>
struct MyC2VPipeRepro {
    struct Producer {
        AICORE inline void record() const
        {
            constexpr uint64_t kCfg = 1 | (2ULL << 4) | ((uint64_t)FlagID << 8);
            ffts_cross_core_sync(PIPE_MTE3, kCfg);
        }
    };

    struct Consumer {
        AICORE inline void wait() const
        {
            wait_flag_dev(FlagID);
        }
    };
};

extern "C" __global__ AICORE void repro_tpipe_issue(
    __gm__ float* __restrict__ gm_in,
    __gm__ float* __restrict__ gm_out,
    __gm__ uint8_t* __restrict__ ffts_addr,
    int32_t n)
{
    (void)gm_in;
    (void)gm_out;
    (void)n;
    set_ffts_base_addr((uint64_t)ffts_addr);

    using ProdTile = TileAcc<float, 16, 16>;
    using ConsTile = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, 8, 16>;
    using C2VPipe = TPipe<0, FIFOType::GM_FIFO, 1, 1, ProdTile, ConsTile>;

#ifdef __DAV_C220_CUBE__
    pipe_barrier(PIPE_MTE3);

    // Build-time switch:
    //  - REPRO_USE_NATIVE_TPIPE_RECORD=1: use upstream TPipe Producer::record()
    //  - REPRO_USE_NATIVE_TPIPE_RECORD=0: use workaround PIPE_MTE3 producer sync
#if REPRO_USE_NATIVE_TPIPE_RECORD
    typename C2VPipe::Producer prod;
    prod.record();
#else
    MyC2VPipeRepro<0>::Producer prod;
    prod.record();
#endif
#endif

#ifdef __DAV_C220_VEC__
    // Consumer API style remains the same.
    typename C2VPipe::Consumer cons;
    cons.wait();
#endif
}

extern "C" void call_kernel(
    uint32_t blockDim, void* stream,
    uint8_t* gm_in, uint8_t* gm_out, int32_t n)
{
    void* ffts_addr;
    uint32_t ffts_len;
    rtGetC2cCtrlAddr((uint64_t*)&ffts_addr, &ffts_len);

    repro_tpipe_issue<<<blockDim, nullptr, stream>>>(
        (__gm__ float*)gm_in,
        (__gm__ float*)gm_out,
        (__gm__ uint8_t*)ffts_addr,
        n);
}
