#include <pto/pto-inst.hpp>

using namespace pto;

#define DIV_ROUNDUP(x, y) (((x) + (y) - 1) / (y))

#if defined(__CHECK_FEATURE_AT_PRECOMPILE) || \
    (__CCE_AICORE__ == 220 && defined(__DAV_C220_CUBE__))

constexpr uint32_t M_TILE = 128;
constexpr uint32_t N = 128;
constexpr uint32_t K = 128;
constexpr uint32_t TILE_ELEMS = M_TILE * K;
constexpr unsigned L1_B = 0x0;
constexpr unsigned L1_A = L1_B + TILE_ELEMS * sizeof(half);

template <pipe_t SrcPipe, pipe_t DstPipe>
AICORE inline void SetFlag(uint32_t id) {
  set_flag(SrcPipe, DstPipe, static_cast<event_t>(id));
}

template <pipe_t SrcPipe, pipe_t DstPipe>
AICORE inline void WaitFlag(uint32_t id) {
  wait_flag(SrcPipe, DstPipe, static_cast<event_t>(id));
}

// simple_matmul: C[M,128] = A[M,128] @ B[128,128], M multiple of 128.
AICORE void run_simple_matmul(__gm__ half *a, __gm__ half *b, __gm__ half *c,
                              uint32_t m) {
  const uint32_t total_batches = m / M_TILE;
  const uint32_t core_id = get_block_idx();
  const uint32_t num_cores = block_num;
  const uint32_t batches_per_core = DIV_ROUNDUP(total_batches, num_cores);
  const uint32_t batch_start = batches_per_core * core_id;
  if (batch_start >= total_batches) return;

  uint32_t batches_to_process = batches_per_core;
  if (batch_start + batches_to_process > total_batches) {
    batches_to_process = total_batches - batch_start;
  }

  using TensorShape = TileShape2D<half, M_TILE, K, Layout::ND>;
  using TensorStrides = BaseShape2D<half, M_TILE, K, Layout::ND>;
  using GlobalIn = GlobalTensor<half, TensorShape, TensorStrides, Layout::ND>;
  using TensorShapeOut = TileShape2D<half, M_TILE, N, Layout::ND>;
  using TensorStridesOut = BaseShape2D<half, M_TILE, N, Layout::ND>;
  using GlobalOut =
      GlobalTensor<half, TensorShapeOut, TensorStridesOut, Layout::ND>;
  using TileL1 = Tile<TileType::Mat, half, M_TILE, K, BLayout::ColMajor, M_TILE,
                      K, SLayout::RowMajor, 512>;
  using TileL0A = TileLeft<half, M_TILE, K>;
  using TileL0B = TileRight<half, K, N>;
  using TileL0C = TileAcc<float, M_TILE, N>;

  TileL1 b_l1;
  TileL1 a_l1;
  TASSIGN(b_l1, L1_B);
  TASSIGN(a_l1, L1_A);

  TileL0A a_l0;
  TileL0B b_l0;
  TileL0C c_l0;
  TASSIGN(a_l0, 0x0);
  TASSIGN(b_l0, 0x0);
  TASSIGN(c_l0, 0x0);

  GlobalIn b_global(b);
  TASSIGN(b_global, b);
  TLOAD(b_l1, b_global);
  SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
  WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
  TMOV(b_l0, b_l1);
  SetFlag<PIPE_MTE1, PIPE_M>(0);
  WaitFlag<PIPE_MTE1, PIPE_M>(0);

  for (uint32_t i = 0; i < batches_to_process; ++i) {
    const uint32_t batch_idx = batch_start + i;
    const uint32_t gm_offset = batch_idx * TILE_ELEMS;
    GlobalIn a_global(a + gm_offset);
    GlobalOut c_global(c + gm_offset);
    TASSIGN(a_global, a + gm_offset);
    TASSIGN(c_global, c + gm_offset);

    TLOAD(a_l1, a_global);
    SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
    WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
    TMOV(a_l0, a_l1);
    SetFlag<PIPE_MTE1, PIPE_M>(0);
    WaitFlag<PIPE_MTE1, PIPE_M>(0);

    TMATMUL(c_l0, a_l0, b_l0);
    pipe_barrier(PIPE_ALL);
    SetFlag<PIPE_M, PIPE_FIX>(0);
    WaitFlag<PIPE_M, PIPE_FIX>(0);
    TSTORE(c_global, c_l0);
    pipe_barrier(PIPE_ALL);
  }
}

#endif

extern "C" __global__ AICORE void simple_matmul_kernel(
    __gm__ void *a, __gm__ void *b, __gm__ void *c, uint32_t m) {
#if defined(__CHECK_FEATURE_AT_PRECOMPILE) || \
    (__CCE_AICORE__ == 220 && defined(__DAV_C220_CUBE__))
  run_simple_matmul((__gm__ half *)a, (__gm__ half *)b, (__gm__ half *)c, m);
#endif
}

extern "C" void call_matmul(uint32_t block_dim, void *stream, uint8_t *a,
                            uint8_t *b, uint8_t *c, uint32_t m) {
  simple_matmul_kernel<<<block_dim, nullptr, stream>>>(a, b, c, m);
}
