// Reproducer: TSYNC/Event integration gap for this C2V case.
//
// Purpose:
// 1) Document that direct TSYNC.hpp/Event usage is not currently a drop-in here.
// 2) Keep a minimal known-good pattern: producer PIPE_MTE3 + consumer wait_flag_dev.
//
// Notes:
// - This file is for issue reproduction/documentation only.
// - It is intentionally minimal and mirrors only the sync-relevant path.

#include <pto/pto-inst.hpp>
#include "runtime/rt.h"

using namespace pto;

extern "C" __global__ AICORE void repro_tsync_issue(
    __gm__ float* __restrict__ gm_in,
    __gm__ float* __restrict__ gm_out,
    __gm__ uint8_t* __restrict__ ffts_addr,
    int32_t n)
{
#ifdef __DAV_C220_CUBE__
    (void)gm_in;
    (void)n;
    set_ffts_base_addr((uint64_t)ffts_addr);

    // Assume cube side already produced gm_out data here.
    pipe_barrier(PIPE_MTE3);

    // Known-good producer sync for this case (matches reference kernel).
    constexpr uint64_t kCfg = 1 | (2ULL << 4) | (0ULL << 8); // CV_CORES_SYNC, flag 0
    ffts_cross_core_sync(PIPE_MTE3, kCfg);
#endif

#ifdef __DAV_C220_VEC__
    (void)gm_in;
    (void)gm_out;
    (void)n;
    set_ffts_base_addr((uint64_t)ffts_addr);

    // Known-good consumer side.
    wait_flag_dev(0);
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
