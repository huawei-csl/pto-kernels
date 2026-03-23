// repro_builtin.cpp — Bug 1 reproducer: TSync_Custom::record() wrong output
//
// Kernel: cube copies gm_input (all 100.0f) → gm_output via TLOAD/TSTORE,
//         signals vector with TSync_Custom::record().
//         Vector waits with TSync_Custom::wait(), loads gm_output, adds
//         get_subblockid() to every element, stores back.
//
// Expected output: 100.0f + subblockid  (100.0 for sub 0, 101.0 for sub 1)
// Observed output:   0.0f + subblockid  (  0.0 for sub 0,   1.0 for sub 1)
//
// Root cause: TSync_Custom::record() calls
//   ffts_cross_core_sync(PIPE_FIX, _getFFTSMsg(CV_CORE_SYNC, flag_id))
// where flag_id is a runtime uint16_t struct member.  Bisheng requires the
// second argument of ffts_cross_core_sync to be a compile-time literal.
// Passing a non-literal compiles without error but generates wrong code:
// the sync signal fires with the wrong FFTS message, the vector proceeds
// without waiting, and reads stale (zero-initialised) gm_output.
//
// See repro_mytync.cpp for the fixed version using MyTSync<FlagID>.

#include <pto/pto-inst.hpp>
#include <pto/npu/a2a3/custom/TSync_Custom.hpp>
#include <pto/npu/a2a3/custom/TSyncCVID.hpp>
#include "runtime/rt.h"

using namespace pto;

extern "C" __global__ AICORE void repro_builtin(
    __gm__ float* __restrict__ gm_input,
    __gm__ float* __restrict__ gm_output,
    __gm__ uint8_t* __restrict__ ffts_addr,
    int32_t N)
{
#ifdef __DAV_C220_CUBE__
    using MatTile    = Tile<TileType::Mat, float, 256, 256, BLayout::RowMajor, DYNAMIC, DYNAMIC>;
    using GMShape    = Shape<1, 1, 1, DYNAMIC, 256>;
    using GlobalFP32 = GlobalTensor<float, GMShape, Stride<1, 1, 1, 256, 1>>;

    set_ffts_base_addr((uint64_t)ffts_addr);
    set_padding(0);
    set_atomic_none();

    // Each cube block owns 2*N floats (both vector sub-cores' data).
    int rows = N * 2 / 256;
    GlobalFP32 gIn (gm_input  + get_block_idx() * N * 2, GMShape(1, 1, 1, rows, 256));
    GlobalFP32 gOut(gm_output + get_block_idx() * N * 2, GMShape(1, 1, 1, rows, 256));

    MatTile l1(rows, 256);
    TASSIGN(l1, (uint32_t)0x0);

    TLOAD(l1, gIn);                              // GM → L1  (MTE2)
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(gOut, l1);                            // L1 → GM  (MTE3)

    pipe_barrier(PIPE_MTE3);

    // BUG: flag_id is a runtime uint16_t member of TSync_Custom.
    // _getFFTSMsg(CV_CORE_SYNC, flag_id) is NOT a compile-time literal.
    // Bisheng silently generates wrong code — vector does not wait.
    TSync_Custom<SyncOpType::TSTORE_C2GM, SyncOpType::TLOAD> sync{0};
    sync.record();  // ← wrong: runtime flag_id in ffts_cross_core_sync 2nd arg

    pipe_barrier(PIPE_ALL);
#endif  // __DAV_C220_CUBE__

#ifdef __DAV_C220_VEC__
    using VecTile    = Tile<TileType::Vec, float, 256, 256, BLayout::RowMajor, DYNAMIC, DYNAMIC>;
    using GMShape    = Shape<1, 1, 1, DYNAMIC, 256>;
    using GlobalFP32 = GlobalTensor<float, GMShape, Stride<1, 1, 1, 256, 1>>;

    set_atomic_none();
    set_mask_norm();
    set_ffts_base_addr((uint64_t)ffts_addr);
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    int id   = get_block_idx() * get_subblockdim() + get_subblockid();
    int rows = N / 256;

    GlobalFP32 gOut(gm_output + id * N, GMShape(1, 1, 1, rows, 256));
    VecTile ub(rows, 256);
    TASSIGN(ub, (uint32_t)0x0);

    TSync_Custom<SyncOpType::TSTORE_C2GM, SyncOpType::TLOAD> sync{0};
    sync.wait();  // wait_flag_dev(flag_id) — this call is fine

    TLOAD(ub, gOut);                             // GM → UB  (MTE2)
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TADDS(ub, ub, (float)get_subblockid());      // add 0 or 1

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(gOut, ub);                            // UB → GM  (MTE3)
    pipe_barrier(PIPE_ALL);
#endif  // __DAV_C220_VEC__
}

extern "C" void call_kernel(
    uint32_t blockDim, void* stream,
    uint8_t* gm_input, uint8_t* gm_output, int32_t N)
{
    void* ffts_addr;
    uint32_t ffts_len;
    rtGetC2cCtrlAddr((uint64_t*)&ffts_addr, &ffts_len);
    repro_builtin<<<blockDim, nullptr, stream>>>(
        (__gm__ float*)gm_input,
        (__gm__ float*)gm_output,
        (__gm__ uint8_t*)ffts_addr,
        N);
}
