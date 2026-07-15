#include "cv_sync_common.hpp"

#ifdef __CCE_AICORE__

AICORE void run_matmul_add_c2v(__gm__ half *A, __gm__ half *B, __gm__ float *C,
                               __gm__ float *D, int64_t batch) {
  const int32_t cid = static_cast<int32_t>(get_block_idx());
  const int32_t vid = static_cast<int32_t>(get_subblockid());
  const int32_t num_cores = static_cast<int32_t>(block_num);
  const int32_t wave_rows = num_cores * TILE_SIZE;
  const int32_t num_rounds = static_cast<int32_t>(batch) / wave_rows;

  TileL1 b_l1, a_l1;
  TASSIGN(b_l1, L1_0_OFFSET);
  TASSIGN(a_l1, L1_1_OFFSET);

  TileL0A a_l0;
  TileL0B b_l0;
  TileL0C c_l0;
  TASSIGN(a_l0, L0_OFFSET);
  TASSIGN(b_l0, L0_OFFSET);
  TASSIGN(c_l0, L0_OFFSET);

  TileVecFloat c_ub, d_ub;
  TASSIGN(c_ub, C2V_FLOAT_UB_BASE);
  TASSIGN(d_ub, UB_0_OFFSET);

#if defined(__DAV_CUBE__)
  TileGlobal b_global(B);
  TLOAD(b_l1, b_global);
  SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
  WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
  TMOV(b_l0, b_l1);
  SetFlag<PIPE_MTE1, PIPE_M>(0);
  WaitFlag<PIPE_MTE1, PIPE_M>(0);

  for (int32_t r = 0; r < num_rounds; ++r) {
    const int32_t row_c = r * wave_rows + cid * TILE_SIZE;
    TileGlobal a_global(A + row_c * TILE_SIZE);
    TLOAD(a_l1, a_global);
    SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
    WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);

    TMOV(a_l0, a_l1);
    SetFlag<PIPE_MTE1, PIPE_M>(0);
    WaitFlag<PIPE_MTE1, PIPE_M>(0);

    TMATMUL(c_l0, a_l0, b_l0);
    SetFlag<PIPE_M, PIPE_FIX>(0);
    WaitFlag<PIPE_M, PIPE_FIX>(0);

    if (r > 0) {
      WaitBothVec<PIPE_FIX>(FLAG_FREE);
    }
    TMOV<TileVecFloat, TileL0C, AccToVecMode::DualModeSplitM>(c_ub, c_l0);
    pipe_barrier(PIPE_ALL);
    SignalBothVec<PIPE_FIX>(FLAG_READY);
  }
  if (num_rounds > 0) {
    WaitBothVec<PIPE_FIX>(FLAG_FREE);
    pipe_barrier(PIPE_ALL);
  }
#endif

#if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  for (int32_t r = 0; r < num_rounds; ++r) {
    const int32_t row_v = r * wave_rows + cid * TILE_SIZE + vid * HALF_TILE;

    wait_intra_block(PIPE_V, FLAG_READY);

    HalfTileGlobalFloat d_global(D + row_v * TILE_SIZE);
    TLOAD(d_ub, d_global);
    SetFlag<PIPE_MTE2, PIPE_V>(0);
    WaitFlag<PIPE_MTE2, PIPE_V>(0);

    TADD(c_ub, c_ub, d_ub);
    SetFlag<PIPE_V, PIPE_MTE3>(0);
    WaitFlag<PIPE_V, PIPE_MTE3>(0);

    HalfTileGlobalFloat c_global(C + row_v * TILE_SIZE);
    TSTORE(c_global, c_ub);
    pipe_barrier(PIPE_ALL);
    set_intra_block(PIPE_V, FLAG_FREE);
  }
#endif
}

#endif

extern "C" __global__ AICORE void matmul_add_c2v_kernel(__gm__ uint8_t *A,
                                                        __gm__ uint8_t *B,
                                                        __gm__ uint8_t *C,
                                                        __gm__ uint8_t *D,
                                                        int64_t batch) {
  run_matmul_add_c2v(reinterpret_cast<__gm__ half *>(A),
                     reinterpret_cast<__gm__ half *>(B),
                     reinterpret_cast<__gm__ float *>(C),
                     reinterpret_cast<__gm__ float *>(D), batch);
}

void LaunchMatmulAddC2V(uint32_t block_dim, uint8_t *A, uint8_t *B, uint8_t *C,
                        uint8_t *D, int64_t batch, void *stream) {
  matmul_add_c2v_kernel<<<block_dim, nullptr, stream>>>(A, B, C, D, batch);
}
