#include "cv_sync_common.hpp"

#ifdef __CCE_AICORE__

AICORE void run_add_matmul_v2c(__gm__ half *A, __gm__ half *B, __gm__ half *C,
                               __gm__ half *D, int64_t batch) {
  const int32_t cid = static_cast<int32_t>(get_block_idx());
  const int32_t vid = static_cast<int32_t>(get_subblockid());
  const int32_t num_cores = static_cast<int32_t>(block_num);
  const int32_t wave_rows = num_cores * TILE_SIZE;
  const int32_t num_rounds = static_cast<int32_t>(batch) / wave_rows;
  constexpr uint16_t READY_FLAG = 10;
  constexpr uint16_t FREE_FLAG = 11;

  TileL1 d_l1, ab_l1;
  TileL1Insert ab_insert;
  TASSIGN(d_l1, L1_0_OFFSET);
  TASSIGN(ab_l1, L1_1_OFFSET);
  TASSIGN(ab_insert, L1_1_OFFSET);

  TileL0A ab_l0;
  TileL0B d_l0;
  TileL0C c_l0;
  TASSIGN(ab_l0, L0_OFFSET);
  TASSIGN(d_l0, L0_OFFSET);
  TASSIGN(c_l0, L0_OFFSET);

  TileVec a_ub, b_ub;
  TileVecNZ ab_nz;
  TASSIGN(a_ub, UB_0_OFFSET);
  TASSIGN(b_ub, UB_1_OFFSET);
  TASSIGN(ab_nz, UB_2_OFFSET);

#if defined(__DAV_CUBE__)
  TileGlobal d_global(D);
  TLOAD(d_l1, d_global);
  SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
  WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
  TMOV(d_l0, d_l1);
  SetFlag<PIPE_MTE1, PIPE_M>(0);
  WaitFlag<PIPE_MTE1, PIPE_M>(0);

  for (int32_t r = 0; r < num_rounds; ++r) {
    const int32_t row_c = r * wave_rows + cid * TILE_SIZE;

    WaitBothVec<PIPE_MTE1>(READY_FLAG);
    pipe_barrier(PIPE_ALL);
    TMOV(ab_l0, ab_l1);
    SetFlag<PIPE_MTE1, PIPE_M>(0);
    WaitFlag<PIPE_MTE1, PIPE_M>(0);
    pipe_barrier(PIPE_ALL);
    SignalBothVec<PIPE_MTE1>(FREE_FLAG);

    TMATMUL(c_l0, ab_l0, d_l0);
    SetFlag<PIPE_M, PIPE_FIX>(0);
    WaitFlag<PIPE_M, PIPE_FIX>(0);

    TileGlobal c_global(C + row_c * TILE_SIZE);
    TSTORE(c_global, c_l0);
    pipe_barrier(PIPE_ALL);
  }
#endif

#if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  for (int32_t r = 0; r < num_rounds; ++r) {
    const int32_t row_v = r * wave_rows + cid * TILE_SIZE + vid * HALF_TILE;

    HalfTileGlobal a_global(A + row_v * TILE_SIZE);
    HalfTileGlobal b_global(B + row_v * TILE_SIZE);
    TLOAD(a_ub, a_global);
    TLOAD(b_ub, b_global);
    SetFlag<PIPE_MTE2, PIPE_V>(0);
    WaitFlag<PIPE_MTE2, PIPE_V>(0);

    TADD(a_ub, a_ub, b_ub);
    SetFlag<PIPE_V, PIPE_MTE3>(0);
    WaitFlag<PIPE_V, PIPE_MTE3>(0);
    TMOV(ab_nz, a_ub);
    SetFlag<PIPE_V, PIPE_MTE3>(0);
    WaitFlag<PIPE_V, PIPE_MTE3>(0);

    if (r > 0) {
      wait_intra_block(PIPE_MTE3, FREE_FLAG);
      pipe_barrier(PIPE_ALL);
    }
    TINSERT(ab_insert, ab_nz, static_cast<uint16_t>(vid * HALF_TILE),
            static_cast<uint16_t>(0));
    pipe_barrier(PIPE_ALL);
    set_intra_block(PIPE_MTE3, READY_FLAG);
  }
  if (num_rounds > 0) {
    wait_intra_block(PIPE_MTE3, FREE_FLAG);
    pipe_barrier(PIPE_ALL);
  }
#endif
}

#endif

extern "C" __global__ AICORE void add_matmul_v2c_kernel(__gm__ uint8_t *A,
                                                        __gm__ uint8_t *B,
                                                        __gm__ uint8_t *C,
                                                        __gm__ uint8_t *D,
                                                        int64_t batch) {
  run_add_matmul_v2c(reinterpret_cast<__gm__ half *>(A),
                     reinterpret_cast<__gm__ half *>(B),
                     reinterpret_cast<__gm__ half *>(C),
                     reinterpret_cast<__gm__ half *>(D), batch);
}

void LaunchAddMatmulV2C(uint32_t block_dim, uint8_t *A, uint8_t *B, uint8_t *C,
                        uint8_t *D, int64_t batch, void *stream) {
  add_matmul_v2c_kernel<<<block_dim, nullptr, stream>>>(A, B, C, D, batch);
}
