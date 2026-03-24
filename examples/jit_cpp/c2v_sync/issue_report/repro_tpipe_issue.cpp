// Reproducer: TPipe C2V producer path mismatch with real compute.
//
// Purpose:
// - Show that TPipe call style is convenient, but for this case we need producer
//   sync on PIPE_MTE3 to match reference behavior.
//
// Notes:
// - This file is for issue documentation.
// - It performs full C2V computation so numeric mismatch is observable.

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
    set_ffts_base_addr((uint64_t)ffts_addr);

    using ProdTile = TileAcc<float, 16, 16>;
    using ConsTile = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, 8, 16>;
    using C2VPipe = TPipe<0, FIFOType::GM_FIFO, 1, 1, ProdTile, ConsTile>;

#ifdef __DAV_C220_CUBE__
    set_padding(0);
    set_atomic_none();

    auto l1_buf = reinterpret_cast<__cbuf__ float*>((uintptr_t)0);
    copy_gm_to_cbuf(
        l1_buf,
        gm_in + get_block_idx() * n * 2,
        0,
        n * 2 / 256,
        32,
        0, 0, PAD_NONE
    );

    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);

    copy_cbuf_to_gm(
        gm_out + get_block_idx() * n * 2,
        l1_buf,
        0,
        n * 2 / 256,
        32,
        0, 0
    );

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

    pipe_barrier(PIPE_ALL);
#endif

#ifdef __DAV_C220_VEC__
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    int subblock_id = get_subblockid();
    int id = get_block_idx() * get_subblockdim() + subblock_id;
    auto ub_buf = reinterpret_cast<__ubuf__ float*>((uintptr_t)0);

    // Consumer API style remains the same.
    typename C2VPipe::Consumer cons;
    cons.wait();

    copy_gm_to_ubuf(
        ub_buf,
        gm_out + id * n,
        0,
        n / 256,
        32,
        0, 0
    );

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    constexpr int ADD_REPEAT = 128;
    constexpr int ADD_NUM_PER_REPEAT = ADD_REPEAT * 64;
    for (int i = 0; i < (n + ADD_NUM_PER_REPEAT - 1) / ADD_NUM_PER_REPEAT; i++) {
        vadds(
            ub_buf + i * ADD_NUM_PER_REPEAT,
            ub_buf + i * ADD_NUM_PER_REPEAT,
            (float)subblock_id,
            ADD_REPEAT,
            1, 1, 8, 8
        );
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    copy_ubuf_to_gm(
        gm_out + id * n,
        ub_buf,
        0,
        n / 256,
        32,
        0, 0
    );

    pipe_barrier(PIPE_ALL);
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
