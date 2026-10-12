#include "cv_sync_common.hpp"

#ifdef __CCE_AICORE__

AICORE void run_stream_c2v(__gm__ half *A, __gm__ half *B, int32_t num_iters)
{
    const int32_t cid = static_cast<int32_t>(get_block_idx());

    TileL1 b_l1, a_l1;
    TASSIGN(b_l1, L1_0_OFFSET);
    TASSIGN(a_l1, L1_1_OFFSET);

    TileL0A a_l0;
    TileL0B b_l0;
    TileL0C c_l0;
    TASSIGN(a_l0, L0_OFFSET);
    TASSIGN(b_l0, L0_OFFSET);
    TASSIGN(c_l0, L0_OFFSET);

    TileVecFloat c_ub;
    TASSIGN(c_ub, UB_0_OFFSET);

#if defined(__DAV_CUBE__)
    TileGlobal b_global(B);
    TLOAD(b_l1, b_global);
    SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
    WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);

    TMOV(b_l0, b_l1);
    SetFlag<PIPE_MTE1, PIPE_M>(0);
    WaitFlag<PIPE_MTE1, PIPE_M>(0);

    TileGlobal a_global(A + cid * TILE_SIZE * TILE_SIZE);
    TLOAD(a_l1, a_global);
    SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
    WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);

    TMOV(a_l0, a_l1);
    SetFlag<PIPE_MTE1, PIPE_M>(0);
    WaitFlag<PIPE_MTE1, PIPE_M>(0);

    TMATMUL(c_l0, a_l0, b_l0);
    SetFlag<PIPE_M, PIPE_FIX>(0);
    WaitFlag<PIPE_M, PIPE_FIX>(0);

    for (int32_t r = 0; r < num_iters; ++r) {
        if (r > 0) {
            WaitBothVec<PIPE_FIX>(FLAG_FREE);
        }
        TMOV<TileVecFloat, TileL0C, AccToVecMode::DualModeSplitM>(c_ub, c_l0);
        pipe_barrier(PIPE_ALL);
        SignalBothVec<PIPE_FIX>(FLAG_READY);
    }
    if (num_iters > 0) {
        WaitBothVec<PIPE_FIX>(FLAG_FREE);
        pipe_barrier(PIPE_ALL);
    }
#endif

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    for (int32_t r = 0; r < num_iters; ++r) {
        wait_intra_block(PIPE_V, FLAG_READY);
        set_intra_block(PIPE_V, FLAG_FREE);
    }
#endif
}

#endif

extern "C" __global__ AICORE void stream_c2v_kernel(__gm__ uint8_t *A, __gm__ uint8_t *B, int32_t num_iters)
{
    run_stream_c2v(reinterpret_cast<__gm__ half *>(A), reinterpret_cast<__gm__ half *>(B), num_iters);
}

void LaunchStreamC2V(uint32_t block_dim, uint8_t *A, uint8_t *B, int32_t num_iters, void *stream)
{
    stream_c2v_kernel<<<block_dim, nullptr, stream>>>(A, B, num_iters);
}

