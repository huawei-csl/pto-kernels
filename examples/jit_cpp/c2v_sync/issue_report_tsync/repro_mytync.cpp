// repro_mytync.cpp — Bug 1 fix: MyTSync<FlagID> correct output
//
// Identical kernel to repro_builtin.cpp, with one change: the sync calls
// use MyTSync<0> instead of TSync_Custom.
//
// MyTSync<FlagID> promotes FlagID to a template non-type parameter, so
// kMsg = 1 | (CV_CORE_SYNC << 4) | (FlagID << 8) is a compile-time constant
// that bisheng folds to the literal 0x21 = 33 at instantiation time.
// Both arguments of ffts_cross_core_sync are then compile-time literals,
// satisfying bisheng's requirement.
//
// Expected output: 100.0f + subblockid  (100.0 for sub 0, 101.0 for sub 1)

#include <pto/pto-inst.hpp>
#include "MyTSync.hpp"    // local bisheng-safe sync wrapper
#include "runtime/rt.h"

using namespace pto;

extern "C" __global__ AICORE void repro_mytync(
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

    // FIX: FlagID=0 is a template param → kMsg folds to literal 33 (0x21).
    // Both args of ffts_cross_core_sync are compile-time literals.
    MyTSync<0> sync;
    sync.record();  // → ffts_cross_core_sync(PIPE_FIX, 33)

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

    MyTSync<0> sync;
    sync.wait();  // → wait_flag_dev(0)

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
    repro_mytync<<<blockDim, nullptr, stream>>>(
        (__gm__ float*)gm_input,
        (__gm__ float*)gm_output,
        (__gm__ uint8_t*)ffts_addr,
        N);
}
