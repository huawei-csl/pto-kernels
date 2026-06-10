#include <pto/pto-inst.hpp>

using namespace pto;

constexpr int TILE = 128;

#if defined(__CCE_AICORE__)
using Global = GlobalTensor<half, TileShape2D<half, TILE, TILE, Layout::ND>,
                            BaseShape2D<half, TILE, TILE, Layout::ND>, Layout::ND>;
using L1Tile = Tile<TileType::Mat, half, TILE, TILE, BLayout::ColMajor, TILE, TILE,
                    SLayout::RowMajor, 512>;
using Left = TileLeft<half, TILE, TILE>;
using Right = TileRight<half, TILE, TILE>;
using Acc = TileAcc<float, TILE, TILE>;

template <pipe_t Src, pipe_t Dst>
AICORE inline void Sync(uint32_t id) {
  set_flag(Src, Dst, static_cast<event_t>(id));
  wait_flag(Src, Dst, static_cast<event_t>(id));
}
#endif

__global__ AICORE void static_matmul_kernel(__gm__ half *a, __gm__ half *b,
                                            __gm__ half *c) {
#if defined(__DAV_C220_CUBE__)
  L1Tile a_l1, b_l1;
  Left a_l0;
  Right b_l0;
  Acc c_l0;
  TASSIGN(a_l1, 0x0);
  TASSIGN(b_l1, TILE * TILE * sizeof(half));
  TASSIGN(a_l0, 0x0);
  TASSIGN(b_l0, 0x0);
  TASSIGN(c_l0, 0x0);

  Global a_g(a), b_g(b), c_g(c);
  TLOAD(a_l1, a_g);
  TLOAD(b_l1, b_g);
  Sync<PIPE_MTE2, PIPE_MTE1>(0);
  TMOV(a_l0, a_l1);
  TMOV(b_l0, b_l1);
  Sync<PIPE_MTE1, PIPE_M>(0);
  TMATMUL(c_l0, a_l0, b_l0);
  Sync<PIPE_M, PIPE_FIX>(0);
  TSTORE(c_g, c_l0);
  pipe_barrier(PIPE_ALL);
#endif
}

extern "C" void call_matmul(uint32_t block_dim, void *stream, uint8_t *a,
                            uint8_t *b, uint8_t *c, uint32_t m) {
  (void)block_dim;
  (void)m;
  static_matmul_kernel<<<1, nullptr, stream>>>((__gm__ half *)a, (__gm__ half *)b,
                                               (__gm__ half *)c);
}
