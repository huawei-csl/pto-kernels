// Ported from npu_kernels/c2v_sync_cce/sync_c2v_kernel.cpp
// Version: TSYNC — uses TLOAD / TSTORE / TADDS from pto-isa; raw intrinsics for sync
//
// Low-level intrinsic → pto-isa wrapper mapping
//   copy_gm_to_cbuf   → TLOAD  (Mat tile, GM → L1)
//   copy_cbuf_to_gm   → TSTORE (Mat tile, L1 → GM)
//   copy_gm_to_ubuf   → TLOAD  (Vec tile, GM → UB)
//   copy_ubuf_to_gm   → TSTORE (Vec tile, UB → GM)
//   vadds loop        → TADDS  (Vec tile, scalar add)
//
// Cross-core sync (ffts_cross_core_sync / wait_flag_dev) is called directly
// with literal arguments.  The pto-isa wrappers TSync_Custom and TSync.hpp Event
// both fail with the bisheng compiler because ffts_cross_core_sync requires all
// arguments to be compile-time literals:
//   • TSync_Custom::record()  — second arg derived from runtime uint16_t flag_id
//   • Event::Init()           — first arg is constexpr pipe_t, not a literal
// Neither satisfies the bisheng strict-literal requirement; only TPipe::Producer
// (which hard-codes PIPE_FIX + compile-time FlagID) passes.
// The sync message is:  1 | (mode=2 << 4) | (flag_id=0 << 8) = 0x21 = 33
//
// Note: ffts_cross_core_sync emits on PIPE_FIX.  We drain PIPE_MTE3 first via
// pipe_barrier so GM writes are visible before the signal fires.
//
// Tile shape convention (both Vec and Mat):
//   cols = 256 floats = 1 KB per burst   (static, matches the original's burstLen=32 blocks)
//   rows = N/256 or 2*N/256              (dynamic via DYNAMIC valid dim)
// Static Rows=256 forces TADDS into "count mode" internally, which handles
// arbitrary runtime element counts without overflow in the repeat counter.
#include <pto/pto-inst.hpp>  // TLoad, TStore, TAddS
#include "runtime/rt.h"      // rtGetC2cCtrlAddr

// FFTS message: base=1, mode=CV_CORE_SYNC=2, flag_id=0
// = 1 | (2 << 4) | (0 << 8) = 0x21
static constexpr uint64_t kC2VSyncMsg = 1u | (2u << 4u) | (0u << 8u);

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
    // kC2VSyncMsg = 1 | (CV_CORE_SYNC=2 << 4) | (flag_id=0 << 8) = 0x21 = 33
    // Both args must be compile-time literals for bisheng.
    ffts_cross_core_sync(PIPE_FIX, kC2VSyncMsg);

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
    // wait_flag_dev(flag_id=0) — literal arg, no bisheng restriction.
    wait_flag_dev(0);

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
