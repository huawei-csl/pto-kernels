// Ported from npu_kernels/c2v_sync_cce/sync_c2v_kernel.cpp
// Version: TSYNC — uses TLOAD / TSTORE / TADDS from pto-isa; MyTSync for sync
//
// Low-level intrinsic → pto-isa wrapper mapping
//   copy_gm_to_cbuf   → TLOAD  (Mat tile, GM → L1)
//   copy_cbuf_to_gm   → TSTORE (Mat tile, L1 → GM)
//   copy_gm_to_ubuf   → TLOAD  (Vec tile, GM → UB)
//   copy_ubuf_to_gm   → TSTORE (Vec tile, UB → GM)
//   vadds loop        → TADDS  (Vec tile, scalar add)
//   ffts_cross_core_sync + wait_flag_dev → MyTSync<FlagID> (see MyTSync.hpp)
//
// MyTSync<0> is a local bisheng-safe replacement for TSync_Custom (flag_id
// promoted to a template parameter so kMsg is a compile-time literal).
// See issue_report.md for the TSync_Custom / Event failure analysis.
//
// Note: MyTSync::record() emits on PIPE_FIX.  We drain PIPE_MTE3 first via
// pipe_barrier so GM writes are visible before the signal fires.
//
// Tile shape convention (both Vec and Mat):
//   cols = 256 floats = 1 KB per burst   (static, matches the original's burstLen=32 blocks)
//   rows = N/256 or 2*N/256              (dynamic via DYNAMIC valid dim)
// Static Rows=256 forces TADDS into "count mode" internally, which handles
// arbitrary runtime element counts without overflow in the repeat counter.
#include <pto/pto-inst.hpp>  // TLoad, TStore, TAddS
#include "MyTSync.hpp"       // bisheng-safe C2V sync wrapper (workaround)
#include "runtime/rt.h"      // rtGetC2cCtrlAddr

using namespace pto;

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

extern "C" __global__ AICORE void sync_c2v_tsync(
    __gm__ float * __restrict__ gm_input,
    __gm__ float * __restrict__ gm_output,
    __gm__ uint8_t * __restrict__ ffts_addr,
    int32_t N)
{
#ifdef __DAV_C220_CUBE__
    // Mat tile for cube AIC side: L1/cbuf, row-major ND layout.
    using MatTile  = Tile<TileType::Mat, float, 256, 256, BLayout::RowMajor, DYNAMIC, DYNAMIC>;
    // GlobalTensor: 1×1×1×rows×256, stride=(1,1,1,256,1) — contiguous burst layout.
    using GMShape  = Shape<1, 1, 1, DYNAMIC, 256>;
    using GlobalFP32 = GlobalTensor<float, GMShape, Stride<1, 1, 1, 256, 1>>;

    set_ffts_base_addr((uint64_t)ffts_addr);
    set_padding(0);
    set_atomic_none();

    // 2*N floats split into (2*N/256) row-bursts of 256 floats (1 KB each).
    int rows_2n = N * 2 / 256;
    GlobalFP32 globalIn (gm_input  + get_block_idx() * N * 2, GMShape(1, 1, 1, rows_2n, 256));
    GlobalFP32 globalOut(gm_output + get_block_idx() * N * 2, GMShape(1, 1, 1, rows_2n, 256));

    MatTile l1_tile(rows_2n, 256);
    TASSIGN(l1_tile, (uint32_t)0x0);  // place at cbuf base

    TLOAD(l1_tile, globalIn);   // GM → L1  (MTE2)

    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);

    TSTORE(globalOut, l1_tile); // L1 → GM  (MTE3)

    // Drain MTE3 (GM write) before emitting the PIPE_FIX signal.
    pipe_barrier(PIPE_MTE3);

    // Signal vector: data is ready in gm_output.
    MyTSync<0> c2v_sync;
    c2v_sync.record();

    pipe_barrier(PIPE_ALL);
#endif  // __DAV_C220_CUBE__

#ifdef __DAV_C220_VEC__
    // Vec tile for vector AIV side: UB, row-major ND layout.
    // Max static dims (256×256) ensure TADDS selects count mode for runtime sizes.
    using VecTile  = Tile<TileType::Vec, float, 256, 256, BLayout::RowMajor, DYNAMIC, DYNAMIC>;
    // GlobalTensor: 1×1×1×rows×256, stride=(1,1,1,256,1) — contiguous burst layout.
    using GMShape  = Shape<1, 1, 1, DYNAMIC, 256>;
    using GlobalFP32 = GlobalTensor<float, GMShape, Stride<1, 1, 1, 256, 1>>;

    set_atomic_none();
    set_mask_norm();
    set_ffts_base_addr((uint64_t)ffts_addr);
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    int id     = get_block_idx() * get_subblockdim() + get_subblockid();
    int rows_n = N / 256;  // N floats = rows_n bursts × 256 floats

    GlobalFP32 globalOut(gm_output + id * N, GMShape(1, 1, 1, rows_n, 256));
    VecTile ub_tile(rows_n, 256);
    TASSIGN(ub_tile, (uint32_t)0x0);  // place at UB base

    // Wait for cube's "data ready" signal.
    MyTSync<0> c2v_sync;
    c2v_sync.wait();

    TLOAD(ub_tile, globalOut);  // GM → UB  (MTE2)

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // Add sub-block index to every element.
    // Replaces the vadds loop from the original kernel.
    TADDS(ub_tile, ub_tile, (float)get_subblockid());

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(globalOut, ub_tile); // UB → GM  (MTE3)

    pipe_barrier(PIPE_ALL);
#endif  // __DAV_C220_VEC__
}

// ---------------------------------------------------------------------------
// Host-side launcher
// ---------------------------------------------------------------------------

extern "C" void call_kernel(
    uint32_t blockDim, void* stream,
    uint8_t* gm_input, uint8_t* gm_output, int32_t N)
{
    void *ffts_addr;
    uint32_t ffts_len;
    rtGetC2cCtrlAddr((uint64_t*)&ffts_addr, &ffts_len);

    sync_c2v_tsync<<<blockDim, nullptr, stream>>>(
        (__gm__ float *)gm_input,
        (__gm__ float *)gm_output,
        (__gm__ uint8_t *)ffts_addr,
        N
    );
}
