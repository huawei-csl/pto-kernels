#include <runtime/rt_ffts.h>

#include <pto/pto-inst.hpp>

using namespace pto;

// Minimal static mix kernel: one Cube block computes A@B into GM workspace,
// then two Vector subblocks add D and store C. Shape is fixed to 128x128.
constexpr int TILE = 128;
constexpr int HALF = 64;
constexpr int VEC_NUM = 2;
constexpr int FLAG_C2V = 6;

#ifdef __CCE_AICORE__
using Global =
    GlobalTensor<half, TileShape2D<half, TILE, TILE, Layout::ND>,
                 BaseShape2D<half, TILE, TILE, Layout::ND>, Layout::ND>;
using HalfGlobal =
    GlobalTensor<half, TileShape2D<half, HALF, TILE, Layout::ND>,
                 BaseShape2D<half, HALF, TILE, Layout::ND>, Layout::ND>;
using MatTile = Tile<TileType::Mat, half, TILE, TILE, BLayout::ColMajor, TILE,
                     TILE, SLayout::RowMajor, 512>;
using Left = TileLeft<half, TILE, TILE>;
using Right = TileRight<half, TILE, TILE>;
using Acc = TileAcc<float, TILE, TILE>;
using Vec =
    Tile<TileType::Vec, half, HALF, TILE, BLayout::RowMajor, HALF, TILE>;

template <pipe_t Pipe>
inline AICORE void Signal(int32_t flag) {
  ffts_cross_core_sync(Pipe, 1 | (VEC_NUM << 4) | (flag << 8));
}
template <pipe_t Src, pipe_t Dst>
inline AICORE void Sync(uint32_t id) {
  set_flag(Src, Dst, static_cast<event_t>(id));
  wait_flag(Src, Dst, static_cast<event_t>(id));
}
#endif

__global__ AICORE void static_matmul_add(__gm__ half *a, __gm__ half *b,
                                         __gm__ half *c, __gm__ half *d,
                                         __gm__ half *workspace,
                                         uint64_t ffts_addr) {
#ifdef __CCE_AICORE__
  const int vid = get_subblockid();
  set_ffts_base_addr(ffts_addr);
#if defined(__DAV_C220_CUBE__)
  MatTile a_l1, b_l1;
  Left a_l0;
  Right b_l0;
  Acc acc;
  TASSIGN(a_l1, 0x0);
  TASSIGN(b_l1, TILE * TILE * sizeof(half));
  TASSIGN(a_l0, 0x0);
  TASSIGN(b_l0, 0x0);
  TASSIGN(acc, 0x0);
  Global a_g(a), b_g(b), ws_g(workspace);
  TLOAD(a_l1, a_g);
  TLOAD(b_l1, b_g);
  Sync<PIPE_MTE2, PIPE_MTE1>(0);
  TMOV(a_l0, a_l1);
  TMOV(b_l0, b_l1);
  Sync<PIPE_MTE1, PIPE_M>(0);
  TMATMUL(acc, a_l0, b_l0);
  Sync<PIPE_M, PIPE_FIX>(0);
  TSTORE(ws_g, acc);
  pipe_barrier(PIPE_ALL);
  Signal<PIPE_FIX>(FLAG_C2V);
#endif
#if defined(__DAV_C220_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  Vec ws_t, d_t;
  TASSIGN(ws_t, 0x0);
  TASSIGN(d_t, HALF * TILE * sizeof(half));
  wait_flag_dev(FLAG_C2V);
  const int row = vid * HALF;
  HalfGlobal ws_g(workspace + row * TILE), d_g(d + row * TILE),
      c_g(c + row * TILE);
  TLOAD(ws_t, ws_g);
  TLOAD(d_t, d_g);
  pipe_barrier(PIPE_ALL);
  TADD(ws_t, ws_t, d_t);
  Sync<PIPE_V, PIPE_MTE3>(0);
  TSTORE(c_g, ws_t);
  pipe_barrier(PIPE_ALL);
#endif
#endif
}

extern "C" void call(uint32_t block_dim, void *stream, uint8_t *a, uint8_t *b,
                     uint8_t *c, uint8_t *d, uint8_t *workspace,
                     int64_t batch) {
  (void)block_dim;
  (void)batch;
  uint32_t ffts_len = 0;
  uint64_t ffts_addr = 0;
  rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
  static_matmul_add<<<1, nullptr, stream>>>(
      (__gm__ half *)a, (__gm__ half *)b, (__gm__ half *)c, (__gm__ half *)d,
      (__gm__ half *)workspace, ffts_addr);
}
