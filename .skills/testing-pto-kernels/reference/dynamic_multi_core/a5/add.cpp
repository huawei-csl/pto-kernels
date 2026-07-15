#include <pto/pto-inst.hpp>

using namespace pto;

constexpr uint32_t UB_BYTES_PER_TILE = 32 * 1024;
constexpr uint32_t ELEMS_PER_TILE = UB_BYTES_PER_TILE / sizeof(half);
constexpr unsigned X_UB = 0x00000;
constexpr unsigned Z_UB = 0x08000;
constexpr unsigned Y_UB = 0x10000;

template <typename T>
AICORE void run_add(__gm__ T *y, __gm__ T *x, __gm__ T *z, uint32_t n) {
#if defined(__DAV_VEC__)
  const uint32_t num_cores = block_num;
  const uint32_t core = block_idx;
  const uint32_t per_core = (n + num_cores - 1) / num_cores;
  const uint32_t base = core * per_core;
  if (base >= n) return;

  uint32_t remaining_core = per_core;
  if (base + remaining_core > n) remaining_core = n - base;

  using Shape = pto::Shape<1, 1, 1, 1, ELEMS_PER_TILE>;
  using Stride = pto::Stride<1, 1, 1, 1, 1>;
  using Global = pto::GlobalTensor<T, Shape, Stride>;
  using VecTile =
      Tile<TileType::Vec, T, 1, ELEMS_PER_TILE, BLayout::RowMajor, -1, -1>;

  Global x_g(x + base);
  Global z_g(z + base);
  Global y_g(y + base);

  for (uint32_t done = 0; done < remaining_core; done += ELEMS_PER_TILE) {
    const uint32_t cols = (remaining_core - done > ELEMS_PER_TILE)
                              ? ELEMS_PER_TILE
                              : (remaining_core - done);
    VecTile x_t(1, cols), z_t(1, cols), y_t(1, cols);
    TASSIGN(x_t, X_UB);
    TASSIGN(z_t, Z_UB);
    TASSIGN(y_t, Y_UB);
    TASSIGN(x_g, x + base + done);
    TASSIGN(z_g, z + base + done);
    TASSIGN(y_g, y + base + done);

    TLOAD(x_t, x_g);
    TLOAD(z_t, z_g);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TADD(y_t, x_t, z_t);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(y_g, y_t);
    pipe_barrier(PIPE_ALL);
  }
#endif
}

extern "C" __global__ AICORE void add_kernel(__gm__ void *y, __gm__ void *x,
                                             __gm__ void *z, uint32_t n) {
  run_add<half>((__gm__ half *)y, (__gm__ half *)x, (__gm__ half *)z, n);
}

extern "C" void call_add(uint32_t block_dim, void *stream, uint8_t *y,
                         uint8_t *x, uint8_t *z, uint32_t n) {
  add_kernel<<<block_dim, nullptr, stream>>>(y, x, z, n);
}
