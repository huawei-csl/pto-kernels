// =============================================================================
// add_matmul_v2c.cpp  —  C = (A + B) @ D   [gm_pipe variant]
//
// ── API variant: GlobalData TSTORE/TLOAD via explicit slot views ───────────
//
// The "gm_pipe" concept: TSTORE and TLOAD operate on GlobalTensor slot views
// computed from the FIFO buffer, while synchronization uses raw ffts calls.
// This separates data-movement (explicit TSTORE/TLOAD) from FIFO sync, which
// is the defining property of the GlobalData path (TALLOC+TPUSH / TPOP+TFREE).
//
//  raw_flag (fixed workspace)         │  gm_pipe (explicit double-buffer slot view)
//  ─────────────────────────────────── ─────────────────────────────────────────────
//  Vec:  TSTORE(ws_half, a_ub)         │  slot = slot_view(fifo + slot_idx * SLOT_SIZE)
//        ffts_cross_core_sync(FLAG_V2C)│  TSTORE(slot, a_ub)
//                                      │  ffts_cross_core_sync(FLAG_V2C)
//  ─────────────────────────────────── ─────────────────────────────────────────────
//  Cube: wait_flag_dev(FLAG_V2C)       │  wait_flag_dev(FLAG_V2C)
//        TLOAD(ab_l1, ws_half)         │  slot = slot_view(fifo + slot_idx * SLOT_SIZE)
//        ffts_cross_core_sync(FLAG_C2V)│  TLOAD(ab_l1, slot)
//                                      │  ffts_cross_core_sync(FLAG_C2V)
//
// Key gm_pipe advantage over raw_flag:
//   • FIFO_DEPTH=2 double-buffer: Vec writes to slot (r%2) while Cube reads
//     slot ((r-1)%2) — they use different slots and can pipeline one iteration.
//   • The GlobalTensor slot view makes the data path explicit (vs implicit
//     workspace pointer) and can be typed at a sub-tile granularity.
//
// Note: we use raw ffts calls for sync rather than TALLOC/TPUSH/TPOP/TFREE
// because TALLOC on the Vec side (2 sub-blocks) requires careful tileIndex
// management that TPUSH(TileData) handles atomically but TALLOC+TPUSH(GlobalData)
// exposes to the programmer.  The C2V direction uses full TALLOC+TPUSH API since
// Cube has only one sub-block.
//
// Python: all float16.  Reference: (A + B) @ D
// =============================================================================

#define MEMORY_BASE
#include <pto/pto-inst.hpp>
#include <pto/common/fifo.hpp>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>

using namespace pto;

#ifdef __DAV_CUBE__
constexpr bool DAV_CUBE = true;
#else
constexpr bool DAV_CUBE = false;
#endif
#ifdef __DAV_VEC__
constexpr bool DAV_VEC = true;
#else
constexpr bool DAV_VEC = false;
#endif

#define TILE_SIZE 128
#define HALF_TILE  64
#define VEC_NUM     2

#ifdef __CCE_AICORE__

constexpr uint32_t L1_D_OFFSET  = 0u;
constexpr uint32_t L1_AB_OFFSET = TILE_SIZE * TILE_SIZE * sizeof(half);  // 32 KB
constexpr uint32_t L0_OFFSET    = 0u;
constexpr uint32_t UB_A_OFFSET  = 0u;
constexpr uint32_t UB_B_OFFSET  = HALF_TILE * TILE_SIZE * sizeof(half);  // 16 KB

constexpr uint32_t FIFO_DEPTH      = 2u;
constexpr uint32_t V2C_SLOT_SIZE   = TILE_SIZE * TILE_SIZE * sizeof(half); // 32 KB/slot
constexpr uint32_t V2C_FIFO_BYTES  = FIFO_DEPTH * V2C_SLOT_SIZE;           // 64 KB/core

// FFTS flag assignments (raw, not managed by TPipe)
// Use IDs 2 and 3 to avoid collision with matmul_add_c2v's TPipe<0> which
// internally occupies flags 0 (push/data-ready) and 1 (free/slot-returned).
constexpr uint32_t FLAG_V2C_DATA = 2;  // Vec signals Cube: slot written
constexpr uint32_t FLAG_V2C_FREE = 3;  // Cube signals Vec: slot consumed
// mode=2 (CV_CORES_SYNC): one Cube broadcast → both Vec sub-blocks unblock,
//                         both Vec sub-blocks signal → Cube unblocks.
constexpr uint32_t SIGNAL_MODE   = 2;

using TileL1 = Tile<TileType::Mat, half, TILE_SIZE, TILE_SIZE,
                    BLayout::ColMajor, TILE_SIZE, TILE_SIZE,
                    SLayout::RowMajor, 512, PadValue::Zero>;
using TileL0A = TileLeft<half,  TILE_SIZE, TILE_SIZE>;
using TileL0B = TileRight<half, TILE_SIZE, TILE_SIZE>;
using TileL0C = TileAcc<float,  TILE_SIZE, TILE_SIZE>;

using TileVecUB = Tile<TileType::Vec, half, HALF_TILE, TILE_SIZE,
                       BLayout::RowMajor, HALF_TILE, TILE_SIZE,
                       SLayout::NoneBox, 512, PadValue::Null>;

// Slot view types (the core of the gm_pipe approach).
// Vec writes T/2 rows per sub-block; Cube reads the full T×T slot.
using HalfSlotView =
    GlobalTensor<half,
                 TileShape2D<half, HALF_TILE, TILE_SIZE, Layout::ND>,
                 BaseShape2D<half, HALF_TILE, TILE_SIZE, Layout::ND>,
                 Layout::ND>;
using FullSlotView =
    GlobalTensor<half,
                 TileShape2D<half, TILE_SIZE, TILE_SIZE, Layout::ND>,
                 BaseShape2D<half, TILE_SIZE, TILE_SIZE, Layout::ND>,
                 Layout::ND>;

// Large-tensor GM accessors (non-FIFO data)
using TileGlobal =
    GlobalTensor<half,
                 TileShape2D<half, TILE_SIZE, TILE_SIZE, Layout::ND>,
                 BaseShape2D<half, TILE_SIZE, TILE_SIZE, Layout::ND>,
                 Layout::ND>;
using HalfTileGlobal =
    GlobalTensor<half,
                 TileShape2D<half, HALF_TILE, TILE_SIZE, Layout::ND>,
                 BaseShape2D<half, HALF_TILE, TILE_SIZE, Layout::ND>,
                 Layout::ND>;

template <pipe_t Src, pipe_t Dst>
AICORE inline void SetFlag(uint32_t id) { set_flag(Src, Dst, static_cast<event_t>(id)); }
template <pipe_t Src, pipe_t Dst>
AICORE inline void WaitFlag(uint32_t id) { wait_flag(Src, Dst, static_cast<event_t>(id)); }

AICORE void run_add_matmul_v2c(
    __gm__ half    *A, __gm__ half *B, __gm__ half *C, __gm__ half *D,
    __gm__ uint8_t *fifo_mem, int64_t batch, uint64_t ffts_addr)
{
    const int32_t cid       = static_cast<int32_t>(get_block_idx());
    const int32_t vid       = static_cast<int32_t>(get_subblockid());
    const int32_t num_cores = static_cast<int32_t>(block_num);

    set_ffts_base_addr(ffts_addr);

    const int32_t wave_rows  = num_cores * TILE_SIZE;
    const int32_t num_rounds = static_cast<int32_t>(batch) / wave_rows;

    // Per-core FIFO region: FIFO_DEPTH × SLOT_SIZE bytes.
    __gm__ uint8_t *core_fifo = fifo_mem + cid * V2C_FIFO_BYTES;

    TileL1  d_l1, ab_l1;
    TASSIGN(d_l1,  L1_D_OFFSET);
    TASSIGN(ab_l1, L1_AB_OFFSET);
    TileL0A ab_l0;  TileL0B d_l0;  TileL0C c_l0;
    TASSIGN(ab_l0, L0_OFFSET);
    TASSIGN(d_l0,  L0_OFFSET);
    TASSIGN(c_l0,  L0_OFFSET);
    TileVecUB a_ub, b_ub;
    TASSIGN(a_ub, UB_A_OFFSET);
    TASSIGN(b_ub, UB_B_OFFSET);

    // ── Cube: wait for slot, TLOAD via slot view, signal free ─────────────────
    if constexpr (DAV_CUBE) {
        TileGlobal d_global(D);
        TLOAD(d_l1, d_global);
        SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
        WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
        TMOV(d_l0, d_l1);
        SetFlag<PIPE_MTE1, PIPE_M>(0);
        WaitFlag<PIPE_MTE1, PIPE_M>(0);

        for (int32_t r = 0; r < num_rounds; ++r) {
            const int32_t row_c = r * wave_rows + cid * TILE_SIZE;

            // Wait for both Vec sub-blocks to write their half-tiles.
            wait_flag_dev(FLAG_V2C_DATA);

            // Compute the current slot view (explicit GlobalTensor — the gm_pipe pattern).
            const uint32_t slot_offset = static_cast<uint32_t>(r % FIFO_DEPTH) * V2C_SLOT_SIZE;
            FullSlotView slot_in(reinterpret_cast<__gm__ half *>(
                reinterpret_cast<uint64_t>(core_fifo) + slot_offset));

            // Explicit TLOAD from the GM slot into L1 (the gm_pipe data-move).
            TLOAD(ab_l1, slot_in);
            SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
            WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);  // MTE2→MTE1: TLOAD done before TMOV

            // Signal Vec: slot consumed, safe to write again next round.
            // Skip for the last round — no more Vec writes will happen.
            if (r + 1 < num_rounds) {
                ffts_cross_core_sync(PIPE_MTE2, 1u | (SIGNAL_MODE << 4) | (FLAG_V2C_FREE << 8));
            }

            // M→MTE1 (ab_l0): wait for the previous round's TMATMUL to finish
            // reading ab_l0 before MTE1 overwrites it with TMOV.
            // Skipped for r=0 — no previous TMATMUL to wait for.
            // Uses id=1 to avoid aliasing the MTE1→M flag (id=0).
            if (r > 0) {
                WaitFlag<PIPE_M, PIPE_MTE1>(1);
            }

            TMOV(ab_l0, ab_l1);
            SetFlag<PIPE_MTE1, PIPE_M>(0);
            WaitFlag<PIPE_MTE1, PIPE_M>(0);

            TMATMUL(c_l0, ab_l0, d_l0);
            // Signal MTE1: ab_l0 is no longer in use by M (TMATMUL done).
            // Consumed by WaitFlag<M,MTE1>(1) in the next round (or the drain
            // after the loop for the final round).
            SetFlag<PIPE_M, PIPE_MTE1>(1);

            SetFlag<PIPE_M, PIPE_FIX>(0);
            WaitFlag<PIPE_M, PIPE_FIX>(0);  // M→FIX: c_l0 ready for TSTORE

            TileGlobal c_global(C + row_c * TILE_SIZE);
            TSTORE(c_global, c_l0);
            // FIX→M (c_l0): wait for TSTORE to finish reading c_l0 before the
            // next TMATMUL writes it.  Uses id=1 to avoid aliasing M→FIX (id=0).
            SetFlag<PIPE_FIX, PIPE_M>(1);
            WaitFlag<PIPE_FIX, PIPE_M>(1);
        }
        // Drain the M→MTE1 token left over from the final round.
        // Without this, the next kernel call's round-1 WaitFlag<M,MTE1>(1) would
        // consume a stale token and skip waiting, risking an L0A conflict.
        WaitFlag<PIPE_M, PIPE_MTE1>(1);
    }

    // ── Vec: compute A+B, write to slot view, signal Cube ─────────────────────
    if constexpr (DAV_VEC) {
        set_mask_norm();
        set_vector_mask(-1, -1);

        for (int32_t r = 0; r < num_rounds; ++r) {
            const int32_t row_v = r * wave_rows + cid * TILE_SIZE + vid * HALF_TILE;

            HalfTileGlobal a_global(A + row_v * TILE_SIZE);
            TLOAD(a_ub, a_global);
            HalfTileGlobal b_global(B + row_v * TILE_SIZE);
            TLOAD(b_ub, b_global);
            pipe_barrier(PIPE_ALL);  // MTE2→V: TLOADs done before TADD

            TADD(a_ub, a_ub, b_ub);
            pipe_barrier(PIPE_ALL);  // V→MTE3: TADD done before TSTORE

            // Wait for Cube to free the slot before overwriting it.
            // TSTORE and ffts_cross_core_sync are both MTE3 — ordered in same pipe.
            if (r >= static_cast<int32_t>(FIFO_DEPTH)) {
                wait_flag_dev(FLAG_V2C_FREE);
                pipe_barrier(PIPE_ALL);
            }

            // Compute explicit slot view for this sub-block's half-tile region.
            const uint32_t slot_offset =
                static_cast<uint32_t>(r % FIFO_DEPTH) * V2C_SLOT_SIZE
                + static_cast<uint32_t>(vid) * HALF_TILE * TILE_SIZE * sizeof(half);
            HalfSlotView slot_out(reinterpret_cast<__gm__ half *>(
                reinterpret_cast<uint64_t>(core_fifo) + slot_offset));

            // Explicit TSTORE to the GM slot (the gm_pipe data-move).
            TSTORE(slot_out, a_ub);
            pipe_barrier(PIPE_ALL);  // MTE3: wait for DMA to complete before signaling Cube
            // Signal Cube: this sub-block has written its T/2 rows.
            // mode=2 → Cube unblocks after BOTH sub-blocks (vid=0 and vid=1) signal.
            ffts_cross_core_sync(PIPE_MTE3, 1u | (SIGNAL_MODE << 4) | (FLAG_V2C_DATA << 8));
        }
    }
}

#endif  // __CCE_AICORE__

extern "C" __global__ AICORE void add_matmul_v2c_kernel(
    __gm__ uint8_t *A, __gm__ uint8_t *B, __gm__ uint8_t *C,
    __gm__ uint8_t *D, __gm__ uint8_t *fifo_mem,
    int64_t batch, uint64_t ffts_addr)
{
    run_add_matmul_v2c(
        reinterpret_cast<__gm__ half *>(A), reinterpret_cast<__gm__ half *>(B),
        reinterpret_cast<__gm__ half *>(C), reinterpret_cast<__gm__ half *>(D),
        fifo_mem, batch, ffts_addr);
}

extern "C" void call(uint32_t block_dim, void *stream,
                     uint8_t *A, uint8_t *B, uint8_t *C,
                     uint8_t *D, uint8_t *fifo_mem, int64_t batch)
{
    uint32_t ffts_len  = 0;
    uint64_t ffts_addr = 0;
    rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
    add_matmul_v2c_kernel<<<block_dim, nullptr, stream>>>(
        A, B, C, D, fifo_mem, batch, ffts_addr);
}
