#include "cv_sync_common.hpp"

#ifdef __CCE_AICORE__

AICORE void run_stream_v2c(__gm__ half *A, __gm__ half *D, int32_t num_iters)
{
    const int32_t cid = static_cast<int32_t>(get_block_idx());
    const int32_t vid = static_cast<int32_t>(get_subblockid());
    const int32_t num_cores = static_cast<int32_t>(block_num);
    const int32_t wave_rows = num_cores * TILE_SIZE;

    TileL1Insert ws_l1;
    TASSIGN(ws_l1, L1_0_OFFSET);

    TileVec a_ub, d_ub;
    TileVecNZ a_nz;
    TASSIGN(a_ub, UB_0_OFFSET);
    TASSIGN(d_ub, UB_1_OFFSET);
    TASSIGN(a_nz, UB_2_OFFSET);

#if defined(__DAV_CUBE__)
    for (int32_t r = 0; r < num_iters; ++r) {
        WaitBothVec<PIPE_MTE1>(FLAG_V2C_READY);
        SignalBothVec<PIPE_MTE1>(FLAG_V2C_FREE);
    }
#endif

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    for (int32_t r = 0; r < num_iters; ++r) {
        const int32_t row_v = r * wave_rows + cid * TILE_SIZE + vid * HALF_TILE;

        HalfTileGlobal a_global(A + row_v * TILE_SIZE);
        HalfTileGlobal d_global(D + row_v * TILE_SIZE);
        TLOAD(a_ub, a_global);
        TLOAD(d_ub, d_global);
        SetFlag<PIPE_MTE2, PIPE_V>(0);
        WaitFlag<PIPE_MTE2, PIPE_V>(0);

        TADD(a_ub, a_ub, d_ub);
        SetFlag<PIPE_V, PIPE_MTE3>(0);
        WaitFlag<PIPE_V, PIPE_MTE3>(0);
        TMOV(a_nz, a_ub);
        SetFlag<PIPE_V, PIPE_MTE3>(0);
        WaitFlag<PIPE_V, PIPE_MTE3>(0);

        if (r > 0) {
            wait_intra_block(PIPE_MTE3, FLAG_V2C_FREE);
        }
        TINSERT(ws_l1, a_nz, static_cast<uint16_t>(vid * HALF_TILE), static_cast<uint16_t>(0));
        set_intra_block(PIPE_MTE3, FLAG_V2C_READY);
    }
    if (num_iters > 0) {
        wait_intra_block(PIPE_MTE3, FLAG_V2C_FREE);
    }
#endif
}

#endif

extern "C" __global__ AICORE void stream_v2c_kernel(__gm__ uint8_t *A, __gm__ uint8_t *D, int32_t num_iters)
{
    run_stream_v2c(reinterpret_cast<__gm__ half *>(A), reinterpret_cast<__gm__ half *>(D), num_iters);
}

void LaunchStreamV2C(uint32_t block_dim, uint8_t *A, uint8_t *D, int32_t num_iters, void *stream)
{
    stream_v2c_kernel<<<block_dim, nullptr, stream>>>(A, D, num_iters);
}

