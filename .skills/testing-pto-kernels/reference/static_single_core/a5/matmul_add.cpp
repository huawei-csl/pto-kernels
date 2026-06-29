#include <pto/pto-inst.hpp>

using namespace pto;

constexpr int TILE = 128;
constexpr int HALF_TILE = 64;
constexpr uint16_t FLAG_READY = 0;
constexpr uint16_t FLAG_FREE = 1;
constexpr uint16_t VEC_FLAG_OFFSET = 16;

#if defined(__CCE_AICORE__)
using Global =
    GlobalTensor<half, TileShape2D<half, TILE, TILE, Layout::ND>,
                 BaseShape2D<half, TILE, TILE, Layout::ND>, Layout::ND>;
using HalfFloat =
    GlobalTensor<float, TileShape2D<float, HALF_TILE, TILE, Layout::ND>,
                 BaseShape2D<float, HALF_TILE, TILE, Layout::ND>, Layout::ND>;
using L1Tile = Tile<TileType::Mat, half, TILE, TILE, BLayout::ColMajor, TILE,
                    TILE, SLayout::RowMajor, 512, PadValue::Zero>;
using Left = TileLeft<half, TILE, TILE>;
using Right = TileRight<half, TILE, TILE>;
using Acc = TileAcc<float, TILE, TILE>;
using VecF = Tile<TileType::Vec, float, HALF_TILE, TILE, BLayout::RowMajor,
                  HALF_TILE, TILE>;

template <pipe_t Src, pipe_t Dst>
AICORE inline void Sync(uint32_t id) {
  set_flag(Src, Dst, static_cast<event_t>(id));
  wait_flag(Src, Dst, static_cast<event_t>(id));
}

template <pipe_t Pipe>
AICORE inline void SignalBoth(uint16_t flag) {
  set_intra_block(Pipe, flag);
  set_intra_block(Pipe, flag + VEC_FLAG_OFFSET);
}

template <pipe_t Pipe>
AICORE inline void WaitBoth(uint16_t flag) {
  wait_intra_block(Pipe, flag);
  wait_intra_block(Pipe, flag + VEC_FLAG_OFFSET);
}
#endif

AICORE void run_static_matmul_add(__gm__ half *a, __gm__ half *b,
                                  __gm__ float *c, __gm__ float *d,
                                  int64_t batch) {
#if defined(__CCE_AICORE__)
  const int cid = static_cast<int>(get_block_idx());
  const int vid = static_cast<int>(get_subblockid());
  const int row_base = cid * TILE;
  if (row_base >= batch) return;

#if defined(__DAV_CUBE__)
  L1Tile a_l1, b_l1;
  Left a_l0;
  Right b_l0;
  Acc acc;
  VecF acc_v;
  TASSIGN(a_l1, 0x0);
  TASSIGN(b_l1, TILE * TILE * sizeof(half));
  TASSIGN(a_l0, 0x0);
  TASSIGN(b_l0, 0x0);
  TASSIGN(acc, 0x0);
  TASSIGN(acc_v, 0x20000);

  Global a_g(a + row_base * TILE), b_g(b);
  TLOAD(a_l1, a_g);
  TLOAD(b_l1, b_g);
  Sync<PIPE_MTE2, PIPE_MTE1>(0);
  TMOV(a_l0, a_l1);
  TMOV(b_l0, b_l1);
  Sync<PIPE_MTE1, PIPE_M>(0);
  TMATMUL(acc, a_l0, b_l0);
  Sync<PIPE_M, PIPE_FIX>(0);
  TMOV<VecF, Acc, AccToVecMode::DualModeSplitM>(acc_v, acc);
  pipe_barrier(PIPE_ALL);
  SignalBoth<PIPE_FIX>(FLAG_READY);
  WaitBoth<PIPE_FIX>(FLAG_FREE);
#endif

#if defined(__DAV_VEC__)
  VecF acc_v, d_v;
  TASSIGN(acc_v, 0x20000);
  TASSIGN(d_v, 0x0);
  wait_intra_block(PIPE_V, FLAG_READY);
  const int row = row_base + vid * HALF_TILE;
  HalfFloat d_g(d + row * TILE), c_g(c + row * TILE);
  TLOAD(d_v, d_g);
  Sync<PIPE_MTE2, PIPE_V>(0);
  TADD(acc_v, acc_v, d_v);
  Sync<PIPE_V, PIPE_MTE3>(0);
  TSTORE(c_g, acc_v);
  pipe_barrier(PIPE_ALL);
  set_intra_block(PIPE_V, FLAG_FREE);
#endif
#endif
}

extern "C" __global__ AICORE void static_matmul_add_kernel(__gm__ uint8_t *a,
                                                           __gm__ uint8_t *b,
                                                           __gm__ uint8_t *c,
                                                           __gm__ uint8_t *d,
                                                           int64_t batch) {
  run_static_matmul_add((__gm__ half *)a, (__gm__ half *)b, (__gm__ float *)c,
                        (__gm__ float *)d, batch);
}

extern "C" void call_static_matmul_add(uint32_t block_dim, void *stream,
                                       uint8_t *a, uint8_t *b, uint8_t *c,
                                       uint8_t *d, int64_t batch) {
  static_matmul_add_kernel<<<block_dim, nullptr, stream>>>(a, b, c, d, batch);
}
