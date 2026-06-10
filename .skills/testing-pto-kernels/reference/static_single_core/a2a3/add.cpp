#include <pto/pto-inst.hpp>

using namespace pto;

constexpr int ROWS = 64;
constexpr int COLS = 64;
constexpr uint32_t ELEMS = ROWS * COLS;

__global__ AICORE void static_add_kernel(__gm__ half *out, __gm__ half *x,
                                         __gm__ half *z) {
#if defined(__DAV_C220_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);

  using Shape5 = Shape<1, 1, 1, 1, ELEMS>;
  using Stride5 = Stride<1, 1, 1, 1, 1>;
  using Global = GlobalTensor<half, Shape5, Stride5>;
  using VecTile = Tile<TileType::Vec, half, 1, ELEMS, BLayout::RowMajor, -1, -1>;

  Global out_g(out);
  Global x_g(x);
  Global z_g(z);
  VecTile x_t(1, ELEMS), z_t(1, ELEMS), out_t(1, ELEMS);
  TASSIGN(x_t, 0x0);
  TASSIGN(z_t, 0x8000);
  TASSIGN(out_t, 0x10000);

  TLOAD(x_t, x_g);
  TLOAD(z_t, z_g);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TADD(out_t, x_t, z_t);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(out_g, out_t);
  pipe_barrier(PIPE_ALL);
#endif
}

extern "C" void call_static_add(uint32_t block_dim, void *stream, uint8_t *out,
                                uint8_t *x, uint8_t *z) {
  (void)block_dim;
  static_add_kernel<<<1, nullptr, stream>>>((__gm__ half *)out, (__gm__ half *)x,
                                            (__gm__ half *)z);
}
