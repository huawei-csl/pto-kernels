#include <pto/common/pto_tile.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

constexpr int M = 16;
constexpr int K = 16;
constexpr int N = 16;

__global__ AICORE void static_matmul_kernel(__gm__ float *out, __gm__ half *a,
                                            __gm__ half *b) {
#if defined(__DAV_CUBE__)
  using GlobalA = GlobalTensor<half, pto::Shape<1, 1, 1, M, K>,
                               pto::Stride<M * K, M * K, M * K, K, 1>>;
  using GlobalB = GlobalTensor<half, pto::Shape<1, 1, 1, K, N>,
                               pto::Stride<K * N, K * N, K * N, N, 1>>;
  using GlobalC = GlobalTensor<float, pto::Shape<1, 1, 1, M, N>,
                               pto::Stride<M * N, M * N, M * N, N, 1>>;

  GlobalA a_g(a);
  GlobalB b_g(b);
  GlobalC c_g(out);

  using MatA = Tile<TileType::Mat, half, M, K, BLayout::ColMajor, M, K,
                    SLayout::RowMajor, 512>;
  using MatB = Tile<TileType::Mat, half, K, N, BLayout::ColMajor, K, N,
                    SLayout::RowMajor, 512>;
  using Left = TileLeft<half, M, K, M, K>;
  using Right = TileRight<half, K, N, K, N>;
  using Acc = TileAcc<float, M, N, M, N>;

  MatA a_l1;
  MatB b_l1;
  Left a_l0;
  Right b_l0;
  Acc c_l0;
  TASSIGN(a_l1, 0x0);
  TASSIGN(b_l1, 0x10000);
  TASSIGN(a_l0, 0x0);
  TASSIGN(b_l0, 0x0);
  TASSIGN(c_l0, 0x0);

  TLOAD(a_l1, a_g);
  TLOAD(b_l1, b_g);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TMOV(a_l0, a_l1);
  TMOV(b_l0, b_l1);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  TMATMUL(c_l0, a_l0, b_l0);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TSTORE(c_g, c_l0);
#endif
}

extern "C" void call_static_matmul(uint32_t block_dim, void *stream,
                                   uint8_t *out, uint8_t *a, uint8_t *b) {
  (void)block_dim;
  static_matmul_kernel<<<1, nullptr, stream>>>(
      (__gm__ float *)out, (__gm__ half *)a, (__gm__ half *)b);
}
