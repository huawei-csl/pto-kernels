// Reproducer: TPipe C2V producer path mismatch with wrapper data path.
//
// Purpose:
// - Show that TPipe call style is convenient, but for this case we need producer
//   sync on PIPE_MTE3 to match reference behavior.
//
// Notes:
// - This file is for issue documentation.
// - Data movement/compute use wrappers (TLOAD/TSTORE/TADDS), so only sync path differs.

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
    using MatTile = Tile<TileType::Mat, float, 256, 256, BLayout::RowMajor, DYNAMIC, DYNAMIC>;
    using GMShape = Shape<1, 1, 1, DYNAMIC, 256>;
    using GlobalFP32 = GlobalTensor<float, GMShape, Stride<1, 1, 1, 256, 1>>;

    set_padding(0);
    set_atomic_none();

    int rows_2n = n * 2 / 256;
    GlobalFP32 globalIn(gm_in + get_block_idx() * n * 2, GMShape(1, 1, 1, rows_2n, 256));
    GlobalFP32 globalOut(gm_out + get_block_idx() * n * 2, GMShape(1, 1, 1, rows_2n, 256));
    MatTile l1_tile(rows_2n, 256);
    TASSIGN(l1_tile, (uint32_t)0x0);

    TLOAD(l1_tile, globalIn);

    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);

    TSTORE(globalOut, l1_tile);

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
    using VecTile = Tile<TileType::Vec, float, 256, 256, BLayout::RowMajor, DYNAMIC, DYNAMIC>;
    using GMShape = Shape<1, 1, 1, DYNAMIC, 256>;
    using GlobalFP32 = GlobalTensor<float, GMShape, Stride<1, 1, 1, 256, 1>>;

    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    int subblock_id = get_subblockid();
    int id = get_block_idx() * get_subblockdim() + subblock_id;
    int rows_n = n / 256;
    GlobalFP32 globalOut(gm_out + id * n, GMShape(1, 1, 1, rows_n, 256));
    VecTile ub_tile(rows_n, 256);
    TASSIGN(ub_tile, (uint32_t)0x0);

    // Consumer API style remains the same.
    typename C2VPipe::Consumer cons;
    cons.wait();

    TLOAD(ub_tile, globalOut);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TADDS(ub_tile, ub_tile, (float)subblock_id);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(globalOut, ub_tile);

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
