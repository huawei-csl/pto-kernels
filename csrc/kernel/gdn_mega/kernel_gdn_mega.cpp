// mega_kernel.cpp — GDN Mega-Kernel (group-value / GQA): all PTO stages in one launch
//
// Same pipeline as pto_mega_kernel, with value heads (H) and key heads (Hg)
// passed at runtime. H dispatches to finite compile-time specializations.
//
// Stages:
//   1. cumsum      (Vec)
//   2. transpose   (Vec)
//   3. kkt         (Cube+Vec)  — K has Hg heads; β,g,A use H value heads
//   4. solve_tril  (Cube)
//   5. wy_fast     (Vec+Cube)
//   6. chunk_h     (Cube+Vec)
//   7. chunk_o     (Cube+Vec)

#ifndef GDN_D
#define GDN_D 128
#endif
#ifndef GDN_C
#define GDN_C 128
#endif
#ifndef GDN_KERNEL_NAME
#define GDN_KERNEL_NAME gdn_mega
#endif
// Note the codegen parser does not support arguments of form "type *name", only "type* name"
// clang-format off
#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t*
#endif
// clang-format off

#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include <type_traits>
using namespace pto;

// ===================================================================
// Device-only helpers (shared with standard mega-kernel)
// ===================================================================
#ifdef __CCE_AICORE__

constexpr uint16_t GDN_SYNC_AIV_FLAG = 12;
constexpr uint16_t GDN_SYNC_AIC_FLAG = 11;
constexpr uint16_t GDN_SYNC_AIC_AIV_FLAG = 13;
constexpr uint16_t GDN_SYNC_AIV_ONLY_ALL = 14;
constexpr uint16_t SYNC_MODE_SHIFT_VALUE = 4;
constexpr uint16_t SYNC_FLAG_SHIFT_VALUE = 8;

// A2/A3 vs A5 compatibility helpers. Mirrors csrc/kernel/kernel_utils.h,
// duplicated here because that header would clash with the local GetffstMsg
// and SYNC_* definitions below.
constexpr uint16_t GDN_VEC_FLAG_OFFSET = 16;

/// The vector pipe barrier is required on A2/A3; A5's vector pipe handles
/// the dependency itself, so this compiles to nothing there.
AICORE inline void PipeBarrierVec() {
#if __CCE_AICORE__ == 220
    pipe_barrier(PIPE_V);
#endif
}

/// L0A goes ZZ (RowMajor) on A2/A3 and NZ (ColMajor) on A5. L0B is ZN
/// (RowMajor) on both.
AICORE constexpr inline pto::BLayout GdnOuterLayout(bool is_left)
{
#ifdef __DAV_C310__
    return is_left ? pto::BLayout::ColMajor : pto::BLayout::RowMajor;
#else
    return pto::BLayout::RowMajor;
#endif
}

/// A5 has two Vec sub-blocks whose flag ids differ by 16, so one logical
/// event signalled by Cube has two waiters.
template <pipe_t Pipe>
AICORE inline void GdnSignalBothVec(uint16_t flag)
{
    set_intra_block(Pipe, flag);
    set_intra_block(Pipe, flag + GDN_VEC_FLAG_OFFSET);
}

template <pipe_t Pipe>
AICORE inline void GdnWaitBothVec(uint16_t flag)
{
    wait_intra_block(Pipe, flag);
    wait_intra_block(Pipe, flag + GDN_VEC_FLAG_OFFSET);
}

AICORE inline uint16_t GetffstMsg(uint16_t mode, uint16_t flagId)
{
    return (0x1 + ((mode & 0x3) << SYNC_MODE_SHIFT_VALUE) + ((flagId & 0xf) << SYNC_FLAG_SHIFT_VALUE));
}

template <bool isAIVOnly = true>
AICORE inline void SyncAllImpl()
{
    pipe_barrier(PIPE_ALL);
    if constexpr (isAIVOnly) {
#if __CCE_AICORE__ == 220
        ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x0, GDN_SYNC_AIV_ONLY_ALL));
        wait_flag_dev(GDN_SYNC_AIV_ONLY_ALL);
#else
        // TODO(A5): see the note below; this is the Vec-only all-block barrier.
        set_intra_block(PIPE_MTE3, GDN_SYNC_AIV_ONLY_ALL);
        wait_intra_block(PIPE_MTE3, GDN_SYNC_AIV_ONLY_ALL);
        pipe_barrier(PIPE_ALL);
#endif
        return;
    }
#if __CCE_AICORE__ == 220
#if defined(__DAV_CUBE__)
    wait_flag_dev(GDN_SYNC_AIV_FLAG);
    ffts_cross_core_sync(PIPE_FIX, GetffstMsg(0x0, GDN_SYNC_AIC_FLAG));
    wait_flag_dev(GDN_SYNC_AIC_FLAG);
    ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, GDN_SYNC_AIC_AIV_FLAG));
#elif defined(__DAV_VEC__)
    ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, GDN_SYNC_AIV_FLAG));
    wait_flag_dev(GDN_SYNC_AIC_AIV_FLAG);
#endif
#else
    // TODO(A5): REVIEW BEFORE RUNNING ON HARDWARE.
    // On A5 the Cube and both Vec sub-blocks share one core, so the Cube<->Vec
    // half of this barrier becomes an intra-block handshake (below). The
    // middle step on A2/A3 is a *cross-block* Cube barrier driven by
    // GDN_SYNC_AIC_FLAG; wait_flag_dev does not exist on A5, and the correct
    // device-wide replacement has not been established. This version omits
    // that step, so it synchronises Cube with Vec within a block but does NOT
    // provide the cross-block guarantee the A2/A3 path does. If any caller
    // relies on an all-block barrier here, this will be wrong.
#if defined(__DAV_CUBE__)
    GdnWaitBothVec<PIPE_MTE2>(GDN_SYNC_AIV_FLAG);
    pipe_barrier(PIPE_ALL);
    GdnSignalBothVec<PIPE_FIX>(GDN_SYNC_AIC_AIV_FLAG);
#elif defined(__DAV_VEC__)
    pipe_barrier(PIPE_ALL);
    set_intra_block(PIPE_MTE3, GDN_SYNC_AIV_FLAG);
    wait_intra_block(PIPE_MTE3, GDN_SYNC_AIC_AIV_FLAG);
    pipe_barrier(PIPE_ALL);
#endif
#endif
}

template <typename T, int32_t H_val>
AICORE inline void mega_transpose_TH_to_HT(__gm__ T *src, __gm__ T *dst, int64_t T_len)
{
#if defined(__DAV_VEC__)
    if (get_subblockid() != 0) return;
    set_mask_norm();
    set_vector_mask(-1, -1);

    auto cid = get_block_idx();
    auto block_num = get_block_num();

    constexpr int32_t BLOCK = 128;
    constexpr int32_t H = static_cast<int32_t>(H_val);
    constexpr int32_t ES = static_cast<int32_t>(sizeof(T));
    constexpr int32_t AlignBytes = 32;
    constexpr int32_t AlignRows = AlignBytes / ES;
    constexpr int32_t MinTransposeCols = 16;
    constexpr int32_t AlignElems = (AlignRows > MinTransposeCols) ? AlignRows : MinTransposeCols;
    constexpr int32_t HP = ((H + AlignElems - 1) / AlignElems) * AlignElems;
    constexpr int32_t SRC_UB = 0;
    constexpr int32_t DST_UB = SRC_UB + BLOCK * HP * ES;
    constexpr int32_t TMP_UB = DST_UB + HP * BLOCK * ES;

    using UBSrcFull =
        Tile<TileType::Vec, T, BLOCK, HP, BLayout::RowMajor, BLOCK, HP, SLayout::NoneBox, 512, PadValue::Zero>;
    using UBSrcDyn =
        Tile<TileType::Vec, T, BLOCK, HP, BLayout::RowMajor, DYNAMIC, DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;
    using UBDst = Tile<TileType::Vec, T, HP, BLOCK, BLayout::RowMajor, HP, BLOCK, SLayout::NoneBox, 512>;
    using UBDstDyn = Tile<TileType::Vec, T, HP, BLOCK, BLayout::RowMajor, DYNAMIC, DYNAMIC, SLayout::NoneBox, 512>;
    using UBTmp = Tile<TileType::Vec, T, BLOCK, HP, BLayout::RowMajor, BLOCK, HP, SLayout::NoneBox, 512>;

    using UBRow = Tile<TileType::Vec, T, 1, BLOCK, BLayout::RowMajor, 1, BLOCK, SLayout::NoneBox, 512>;
    using UBRowDyn = Tile<TileType::Vec, T, 1, BLOCK, BLayout::RowMajor, DYNAMIC, DYNAMIC, SLayout::NoneBox, 512>;
    using UBHeadDyn = Tile<TileType::Vec, T, AlignRows, BLOCK, BLayout::ColMajor, 1, DYNAMIC, SLayout::NoneBox, 512>;

    using Gm2D = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using Gm1D = Shape<1, 1, 1, 1, DYNAMIC>;
    using GmSrcS = pto::Stride<1, 1, 1, H, 1>;
    using GmHeadS = pto::Stride<1, 1, 1, 1, H>;
    using GmS1 = pto::Stride<1, 1, 1, 1, 1>;

    if constexpr (H < MinTransposeCols) {
        int64_t num_tok_blocks = (T_len + BLOCK - 1) / BLOCK;
        for (int64_t bi = static_cast<int64_t>(cid); bi < num_tok_blocks; bi += static_cast<int64_t>(block_num)) {
            int64_t t0 = bi * BLOCK;
            int32_t valid = (t0 + BLOCK <= T_len) ? BLOCK : static_cast<int32_t>(T_len - t0);

            for (int32_t h = 0; h < H; ++h) {
                Gm1D gs;
                gs.shape[4] = valid;
                UBHeadDyn row(valid);
                TASSIGN(row, SRC_UB);
                {
                    // DN layout makes each token a one-element burst with stride H.
                    GlobalTensor<T, Gm1D, GmHeadS, Layout::DN> gm(src + t0 * H + h, gs);
                    TLOAD(row, gm);
                }
                set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                {
                    GlobalTensor<T, Gm1D, GmS1, Layout::DN> gm(dst + h * T_len + t0, gs);
                    TSTORE(gm, row);
                }
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            }
        }
        return;
    }

    UBSrcFull ub_src;
    TASSIGN(ub_src, SRC_UB);
    UBDst ub_dst;
    TASSIGN(ub_dst, DST_UB);
    UBTmp ub_tmp;
    TASSIGN(ub_tmp, TMP_UB);

    int64_t num_tok_blocks = (T_len + BLOCK - 1) / BLOCK;

    for (int64_t bi = static_cast<int64_t>(cid); bi < num_tok_blocks; bi += static_cast<int64_t>(block_num)) {
        int64_t t0 = bi * BLOCK;
        int32_t valid = (t0 + BLOCK <= T_len) ? BLOCK : static_cast<int32_t>(T_len - t0);

        {
            Gm2D gs;
            gs.shape[3] = valid;
            gs.shape[4] = H;
            GlobalTensor<T, Gm2D, GmSrcS> gm(src + t0 * H, gs);
            UBSrcDyn ld(valid, H);
            TASSIGN(ld, SRC_UB);
            TLOAD(ld, gm);
            if (valid != BLOCK || H != HP) TFILLPAD_INPLACE(ub_src, ld);
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        TTRANS(ub_dst, ub_src, ub_tmp);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        for (int32_t h = 0; h < H; ++h) {
            Gm1D gs;
            gs.shape[4] = valid;
            GlobalTensor<T, Gm1D, GmS1> gm(dst + h * T_len + t0, gs);
            UBRowDyn st(1, valid);
            TASSIGN(st, DST_UB + h * BLOCK * ES);
            TSTORE(gm, st);
        }
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }
#endif
}

template <int32_t H, int32_t C>
AICORE inline void mega_cast_fp32_to_fp16_bsnd(__gm__ float *src, __gm__ half *dst, uint32_t num_matrices,
                                               int64_t total_tokens)
{
#if defined(__DAV_VEC__)
    if (get_subblockid() != 0) return;
    set_mask_norm();
    set_vector_mask(-1, -1);

    auto cid = get_block_idx();
    auto block_num = get_block_num();

    constexpr int32_t F32_UB = 0;
    constexpr int32_t F16_UB = C * static_cast<int32_t>(sizeof(float));

    using SrcUB = Tile<TileType::Vec, float, 1, C, BLayout::RowMajor, 1, C, SLayout::NoneBox, 512, PadValue::Zero>;
    using DynSrcUB =
        Tile<TileType::Vec, float, 1, C, BLayout::RowMajor, DYNAMIC, DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;
    using DstUB = Tile<TileType::Vec, half, 1, C, BLayout::RowMajor, 1, C, SLayout::NoneBox, 512>;
    using DynDstUB = Tile<TileType::Vec, half, 1, C, BLayout::RowMajor, DYNAMIC, DYNAMIC, SLayout::NoneBox, 512>;
    using Gm1D = Shape<1, 1, 1, 1, DYNAMIC>;
    using GmS1 = pto::Stride<1, 1, 1, 1, 1>;

    SrcUB src_ub;
    TASSIGN(src_ub, F32_UB);
    DstUB dst_ub;
    TASSIGN(dst_ub, F16_UB);

    for (uint32_t m = cid; m < num_matrices; m += block_num) {
        uint32_t h = m % static_cast<uint32_t>(H);
        uint32_t chunk_idx = m / static_cast<uint32_t>(H);

        for (int64_t t = 0; t < total_tokens; ++t) {
            int64_t off = t * static_cast<int64_t>(H * C) + static_cast<int64_t>(h * C);

            {
                Gm1D gs;
                gs.shape[4] = C;
                GlobalTensor<float, Gm1D, GmS1> gm(src + off, gs);
                SrcUB ld;
                TASSIGN(ld, F32_UB);
                TLOAD(ld, gm);
            }
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            TCVT(dst_ub, src_ub, RoundMode::CAST_NONE);

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            {
                Gm1D gs;
                gs.shape[4] = C;
                GlobalTensor<half, Gm1D, GmS1> gm(dst + off, gs);
                DstUB st;
                TASSIGN(st, F16_UB);
                TSTORE(gm, st);
            }
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        }
    }
#endif
}

#endif  // __CCE_AICORE__

// ===================================================================
// Include original kernel implementations in separate namespaces.
// ===================================================================

namespace mk_cumsum {
#include "chunk_cumsum.cpp"
}

namespace mk_kkt {
#include "scaled_dot_kkt.cpp"
}

namespace mk_solve {
#include "tri_inverse_impl.cpp"
}

namespace mk_wy {
#include "wy_fast.cpp"
}

namespace mk_h {
#include "chunk_h.cpp"
}

namespace mk_o {
#include "chunk_o.cpp"
}

AICORE inline void mega_solve_tril(__gm__ half *out, __gm__ half *in, __gm__ half *minus_id, uint32_t matrix_size,
                                   uint32_t num_matrices, uint32_t num_bsnd_heads, __gm__ int32_t *cu_seqlens,
                                   uint32_t is_lower)
{
    if (num_matrices <= get_block_num())
        mk_solve::runKernelTriInvRecUnroll<half, float, GDN_C, 1, true, half>(out, in, minus_id, num_matrices,
                                                                              num_bsnd_heads, cu_seqlens, is_lower);
    else if (num_matrices <= 2u * get_block_num())
        mk_solve::runKernelTriInvRecUnroll<half, float, GDN_C, 2, true, half>(out, in, minus_id, num_matrices,
                                                                              num_bsnd_heads, cu_seqlens, is_lower);
    else
        mk_solve::runKernelTriInvRecUnroll<half, float, GDN_C, 4, true, half>(out, in, minus_id, num_matrices,
                                                                              num_bsnd_heads, cu_seqlens, is_lower);
}

template <int32_t H>
AICORE inline void mega_kernel_impl(GM_ADDR q_ptr, GM_ADDR k_ptr, GM_ADDR v_ptr, GM_ADDR g_in_ptr, GM_ADDR beta_ptr,
                                    GM_ADDR msk_lower_ptr, GM_ADDR msk_full_ptr, GM_ADDR minus_id_ptr,
                                    GM_ADDR cu_seqlens_ptr, GM_ADDR o_ptr, GM_ADDR g_sum_ptr, GM_ADDR g_t_ptr,
                                    GM_ADDR beta_t_ptr, GM_ADDR A_ptr, GM_ADDR A_inv_f32_ptr, GM_ADDR A_inv_ptr,
                                    GM_ADDR w_ptr, GM_ADDR u_ptr, GM_ADDR s_ptr, GM_ADDR v_new_ptr, GM_ADDR fs_ptr,
                                    GM_ADDR h0_ptr, int64_t has_initial_state, GM_ADDR kkt_ws_ptr,
                                    GM_ADDR wy_ws_a1_ptr, GM_ADDR wy_ws_a2_ptr, GM_ADDR h_ws_ptr,
                                    GM_ADDR o_ws_qk_ptr, GM_ADDR o_ws_qs_ptr, GM_ADDR o_ws_gated_ptr,
                                    uint32_t num_key_heads, int64_t batch_size, int64_t seq_len,
                                    int64_t total_tokens, uint32_t num_matrices)
{
    constexpr int32_t D = GDN_D;
    constexpr int32_t C = GDN_C;

    if (num_key_heads == 0 || (static_cast<uint32_t>(H) % num_key_heads) != 0) {
        return;
    }

    mk_cumsum::cumsum_kernel<H, C>(reinterpret_cast<__gm__ float *>(g_in_ptr),
                                   reinterpret_cast<__gm__ float *>(g_sum_ptr),
                                   reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr), batch_size, seq_len);

#ifdef MEGA_STOP_AFTER_CUMSUM
    pipe_barrier(PIPE_ALL);
    return;
#endif

    SyncAllImpl<false>();

#ifdef MEGA_STOP_AFTER_SYNC1
    return;
#endif

    mega_transpose_TH_to_HT<float, H>(reinterpret_cast<__gm__ float *>(g_sum_ptr),
                                      reinterpret_cast<__gm__ float *>(g_t_ptr), total_tokens);
    mega_transpose_TH_to_HT<half, H>(reinterpret_cast<__gm__ half *>(beta_ptr),
                                     reinterpret_cast<__gm__ half *>(beta_t_ptr), total_tokens);

#ifdef MEGA_STOP_AFTER_TRANSPOSE
    pipe_barrier(PIPE_ALL);
    return;
#endif

    SyncAllImpl<false>();

    mk_kkt::kkt_kernel<H, D, C>(
        reinterpret_cast<__gm__ half *>(k_ptr), reinterpret_cast<__gm__ half *>(beta_t_ptr),
        reinterpret_cast<__gm__ float *>(g_t_ptr), reinterpret_cast<__gm__ float *>(msk_lower_ptr),
        reinterpret_cast<__gm__ half *>(kkt_ws_ptr), reinterpret_cast<__gm__ half *>(A_ptr),
        reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr), batch_size, seq_len, total_tokens, num_key_heads);

#if defined(__DAV_CUBE__)
    pipe_barrier(PIPE_ALL);
#if __CCE_AICORE__ == 220
    wait_flag_dev(2);
    wait_flag_dev(3);
#else
    GdnWaitBothVec<PIPE_MTE2>(2);
    GdnWaitBothVec<PIPE_MTE2>(3);
    pipe_barrier(PIPE_ALL);
#endif
#endif

#ifdef MEGA_STOP_AFTER_KKT
    pipe_barrier(PIPE_ALL);
    return;
#endif

    SyncAllImpl<false>();

    mega_solve_tril(reinterpret_cast<__gm__ half *>(A_inv_ptr), reinterpret_cast<__gm__ half *>(A_ptr),
                    reinterpret_cast<__gm__ half *>(minus_id_ptr), C, num_matrices, H,
                    reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr), 1);

#ifdef MEGA_STOP_AFTER_SOLVE
    pipe_barrier(PIPE_ALL);
    return;
#endif

    SyncAllImpl<false>();

#ifdef MEGA_STOP_AFTER_CAST
    pipe_barrier(PIPE_ALL);
    return;
#endif

    SyncAllImpl<false>();

#ifdef MEGA_STOP_AFTER_SYNC_BEFORE_WY
    return;
#endif

    mk_wy::wy_fast_kernel<H, D, C>(
        reinterpret_cast<__gm__ half *>(k_ptr), reinterpret_cast<__gm__ half *>(v_ptr),
        reinterpret_cast<__gm__ half *>(beta_t_ptr), reinterpret_cast<__gm__ float *>(g_t_ptr),
        reinterpret_cast<__gm__ half *>(A_inv_ptr), reinterpret_cast<__gm__ half *>(wy_ws_a1_ptr),
        reinterpret_cast<__gm__ half *>(wy_ws_a2_ptr), reinterpret_cast<__gm__ half *>(w_ptr),
        reinterpret_cast<__gm__ half *>(u_ptr), reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr), batch_size, seq_len,
        total_tokens, num_key_heads);

#if defined(__DAV_VEC__)
    if (get_block_idx() < num_matrices) {
        pipe_barrier(PIPE_ALL);
#if __CCE_AICORE__ == 220
        wait_flag_dev(3);
        wait_flag_dev(4);
#else
        wait_intra_block(PIPE_MTE3, 3);
        wait_intra_block(PIPE_MTE3, 4);
        pipe_barrier(PIPE_ALL);
#endif
    }
#endif

#ifdef MEGA_STOP_AFTER_WY
    pipe_barrier(PIPE_ALL);
    return;
#endif

    SyncAllImpl<false>();

    mk_h::chunk_h_kernel<H, D, C>(
        reinterpret_cast<__gm__ half *>(k_ptr), reinterpret_cast<__gm__ half *>(w_ptr),
        reinterpret_cast<__gm__ half *>(u_ptr), reinterpret_cast<__gm__ float *>(g_t_ptr),
        reinterpret_cast<__gm__ half *>(s_ptr), reinterpret_cast<__gm__ half *>(v_new_ptr),
        reinterpret_cast<__gm__ half *>(fs_ptr), reinterpret_cast<__gm__ half *>(h0_ptr), has_initial_state,
        reinterpret_cast<__gm__ half *>(h_ws_ptr), reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr), batch_size,
        seq_len, total_tokens, num_key_heads);

#ifdef MEGA_STOP_AFTER_H
    pipe_barrier(PIPE_ALL);
    return;
#endif

    SyncAllImpl<false>();

    mk_o::chunk_o_kernel<H, D, C>(
        reinterpret_cast<__gm__ half *>(q_ptr), reinterpret_cast<__gm__ half *>(k_ptr),
        reinterpret_cast<__gm__ half *>(v_new_ptr), reinterpret_cast<__gm__ half *>(s_ptr),
        reinterpret_cast<__gm__ float *>(g_t_ptr), reinterpret_cast<__gm__ float *>(msk_full_ptr),
        reinterpret_cast<__gm__ half *>(o_ws_qk_ptr), reinterpret_cast<__gm__ half *>(o_ws_qs_ptr),
        reinterpret_cast<__gm__ half *>(o_ws_gated_ptr), reinterpret_cast<__gm__ half *>(o_ptr),
        reinterpret_cast<__gm__ int32_t *>(cu_seqlens_ptr), batch_size, seq_len, total_tokens, num_key_heads);

#if defined(__DAV_CUBE__)
    if (get_block_idx() < num_matrices) {
        pipe_barrier(PIPE_ALL);
#if __CCE_AICORE__ == 220
        wait_flag_dev(3);
#else
        GdnWaitBothVec<PIPE_MTE2>(3);
        pipe_barrier(PIPE_ALL);
#endif
    }
#endif
}

// Note the codegen parser does not support arguments of form "type *name", only "type* name"
extern "C" __global__ AICORE void
GDN_KERNEL_NAME(GM_ADDR q_ptr, GM_ADDR k_ptr, GM_ADDR v_ptr, GM_ADDR g_in_ptr, GM_ADDR beta_ptr, GM_ADDR msk_lower_ptr,
                GM_ADDR msk_full_ptr, GM_ADDR minus_id_ptr, GM_ADDR cu_seqlens_ptr, GM_ADDR o_ptr, GM_ADDR g_sum_ptr,
                GM_ADDR g_t_ptr, GM_ADDR beta_t_ptr, GM_ADDR A_ptr, GM_ADDR A_inv_f32_ptr, GM_ADDR A_inv_ptr,
                GM_ADDR w_ptr, GM_ADDR u_ptr, GM_ADDR s_ptr, GM_ADDR v_new_ptr, GM_ADDR fs_ptr, GM_ADDR h0_ptr,
                int64_t has_initial_state, GM_ADDR kkt_ws_ptr, GM_ADDR wy_ws_a1_ptr, GM_ADDR wy_ws_a2_ptr,
                GM_ADDR h_ws_ptr, GM_ADDR o_ws_qk_ptr, GM_ADDR o_ws_qs_ptr, GM_ADDR o_ws_gated_ptr,
                uint32_t num_heads, uint32_t num_key_heads, int64_t batch_size, int64_t seq_len,
                int64_t total_tokens, uint32_t num_matrices)
{
    // The fused pipeline runs chunk_h / chunk_o style mix stages and relies on
    // FFTS cross-core sync, so the launch must be marked as a mix kernel. The
    // staged GDN kernels in csrc/kernel/ use the same task type. Without this
    // the ASC toolchain warns that the type cannot be derived automatically.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

#define DISPATCH_MEGA_H(H)                                                                                         \
    case H:                                                                                                        \
        mega_kernel_impl<H>(q_ptr, k_ptr, v_ptr, g_in_ptr, beta_ptr, msk_lower_ptr, msk_full_ptr, minus_id_ptr,    \
                            cu_seqlens_ptr, o_ptr, g_sum_ptr, g_t_ptr, beta_t_ptr, A_ptr, A_inv_f32_ptr, A_inv_ptr, \
                            w_ptr, u_ptr, s_ptr, v_new_ptr, fs_ptr, h0_ptr, has_initial_state, kkt_ws_ptr,          \
                            wy_ws_a1_ptr, wy_ws_a2_ptr, h_ws_ptr, o_ws_qk_ptr, o_ws_qs_ptr, o_ws_gated_ptr,         \
                            num_key_heads, batch_size, seq_len, total_tokens, num_matrices);                       \
        return

    switch (num_heads) {
        DISPATCH_MEGA_H(8);
        DISPATCH_MEGA_H(12);
        DISPATCH_MEGA_H(16);
        DISPATCH_MEGA_H(24);
        DISPATCH_MEGA_H(32);
        DISPATCH_MEGA_H(48);
        DISPATCH_MEGA_H(64);
        default:
            return;
    }

#undef DISPATCH_MEGA_H
}

// Host-callable launch shim: the `<<<>>>` syntax is only understood by the
// kernel compiler, so the launch lives here rather than in the host wrapper
// under csrc/host/.
extern "C" void pto_launch_gdn_mega(
    uint32_t blockDim, void* stream, void* q, void* k, void* v, void* g_in,
    void* beta, void* msk_lower, void* msk_full, void* minus_id,
    void* cu_seqlens, void* o, void* g_sum, void* g_t, void* beta_t, void* A,
    void* A_inv_f32, void* A_inv, void* w, void* u, void* s, void* v_new,
    void* fs, void* h0, int64_t has_initial_state, void* kkt_ws,
    void* wy_ws_a1, void* wy_ws_a2, void* h_ws, void* o_ws_qk, void* o_ws_qs,
    void* o_ws_gated, uint32_t num_heads, uint32_t num_key_heads,
    int64_t batch_size, int64_t seq_len, int64_t total_tokens,
    uint32_t num_matrices) {
  gdn_mega<<<blockDim, nullptr, stream>>>(
      (GM_ADDR)q, (GM_ADDR)k, (GM_ADDR)v, (GM_ADDR)g_in, (GM_ADDR)beta,
      (GM_ADDR)msk_lower, (GM_ADDR)msk_full, (GM_ADDR)minus_id,
      (GM_ADDR)cu_seqlens, (GM_ADDR)o, (GM_ADDR)g_sum, (GM_ADDR)g_t,
      (GM_ADDR)beta_t, (GM_ADDR)A, (GM_ADDR)A_inv_f32, (GM_ADDR)A_inv,
      (GM_ADDR)w, (GM_ADDR)u, (GM_ADDR)s, (GM_ADDR)v_new, (GM_ADDR)fs,
      (GM_ADDR)h0, has_initial_state, (GM_ADDR)kkt_ws, (GM_ADDR)wy_ws_a1,
      (GM_ADDR)wy_ws_a2, (GM_ADDR)h_ws, (GM_ADDR)o_ws_qk, (GM_ADDR)o_ws_qs,
      (GM_ADDR)o_ws_gated, num_heads, num_key_heads, batch_size, seq_len,
      total_tokens, num_matrices);
}
