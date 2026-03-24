// Reproducer: TSYNC_Custom C2V behavior with real compute.
//
// This kernel performs the same C2V data flow as the main example:
// - cube: GM -> cbuf -> GM
// - vec:  GM -> ub, add subblock id, ub -> GM
// Sync intentionally uses high-level TSYNC API to expose the current issue.

#include <pto/pto-inst.hpp>
// Compatibility shim: some pto-isa snapshots use CV_CORE_SYNC in
// TSync_Custom.hpp without exporting the symbol in this include surface.
#ifndef CV_CORE_SYNC
#define CV_CORE_SYNC 2
#endif
// Compatibility shim: some snapshots call _getFFTSMsg while only getFFTSMsg
// is exported in TSync.hpp.
#include <pto/npu/a2a3/TSync.hpp>
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
    set_ffts_base_addr((uint64_t)ffts_addr);
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

    // Producer-side TSYNC API under test.
    c2v_sync.record();

    pipe_barrier(PIPE_ALL);
#endif

#ifdef __DAV_C220_VEC__
    set_atomic_none();
    set_mask_norm();
    set_ffts_base_addr((uint64_t)ffts_addr);
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    int subblock_id = get_subblockid();
    int id = get_block_idx() * get_subblockdim() + subblock_id;
    auto ub_buf = reinterpret_cast<__ubuf__ float*>((uintptr_t)0);

    // Consumer-side TSYNC API under test.
    c2v_sync.wait();

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

    repro_tsync_issue<<<blockDim, nullptr, stream>>>(
        (__gm__ float*)gm_in,
        (__gm__ float*)gm_out,
        (__gm__ uint8_t*)ffts_addr,
        n);
}
