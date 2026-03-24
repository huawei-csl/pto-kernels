// Reproducer: TSYNC_Custom C2V behavior with wrapper data path.
//
// Data movement and compute use wrappers (TLOAD/TSTORE/TADDS).
// Sync intentionally uses high-level TSYNC_Custom API under test.

#include <pto/pto-inst.hpp>
// Compatibility shim: some pto-isa snapshots use CV_CORE_SYNC in
// TSync_Custom.hpp without exporting the symbol in this include surface.
#ifndef CV_CORE_SYNC
#define CV_CORE_SYNC 2
#endif
// Compatibility shim: avoid including TSync.hpp (it pulls event.hpp, which
// references PIPE_FIX not declared in this compile surface). Provide local
// getFFTSMsg + alias expected by TSync_Custom.hpp.
static AICORE inline uint16_t getFFTSMsg(uint16_t mode, uint16_t eventId, uint16_t baseConst = 0x1)
{
    return ((baseConst & 0xf) + ((mode & 0x3) << 4) + ((eventId & 0xf) << 8));
}
#ifndef _getFFTSMsg
#define _getFFTSMsg getFFTSMsg
#endif
#include <pto/npu/a2a3/custom/TSync_Custom.hpp>
#include "runtime/rt.h"

using namespace pto;

extern "C" __global__ AICORE void repro_tsync_issue(
    __gm__ float* __restrict__ gm_in,
    __gm__ float* __restrict__ gm_out,
    __gm__ uint8_t* __restrict__ ffts_addr,
    int32_t n)
{
    // Intentionally use high-level TSYNC wrapper under test.
    constexpr TSync_Custom<SyncOpType::TSTORE_C2GM, SyncOpType::TLOAD> c2v_sync = {0};

#ifdef __DAV_C220_CUBE__
    using MatTile = Tile<TileType::Mat, float, 256, 256, BLayout::RowMajor, DYNAMIC, DYNAMIC>;
    using GMShape = Shape<1, 1, 1, DYNAMIC, 256>;
    using GlobalFP32 = GlobalTensor<float, GMShape, Stride<1, 1, 1, 256, 1>>;

    set_ffts_base_addr((uint64_t)ffts_addr);
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

    // Producer-side TSYNC API under test.
    c2v_sync.record();

    pipe_barrier(PIPE_ALL);
#endif

#ifdef __DAV_C220_VEC__
    using VecTile = Tile<TileType::Vec, float, 256, 256, BLayout::RowMajor, DYNAMIC, DYNAMIC>;
    using GMShape = Shape<1, 1, 1, DYNAMIC, 256>;
    using GlobalFP32 = GlobalTensor<float, GMShape, Stride<1, 1, 1, 256, 1>>;

    set_atomic_none();
    set_mask_norm();
    set_ffts_base_addr((uint64_t)ffts_addr);
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    int subblock_id = get_subblockid();
    int id = get_block_idx() * get_subblockdim() + subblock_id;
    int rows_n = n / 256;
    GlobalFP32 globalOut(gm_out + id * n, GMShape(1, 1, 1, rows_n, 256));
    VecTile ub_tile(rows_n, 256);
    TASSIGN(ub_tile, (uint32_t)0x0);

    // Consumer-side TSYNC API under test.
    c2v_sync.wait();

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

    repro_tsync_issue<<<blockDim, nullptr, stream>>>(
        (__gm__ float*)gm_in,
        (__gm__ float*)gm_out,
        (__gm__ uint8_t*)ffts_addr,
        n);
}
