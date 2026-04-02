#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void _kernel(__gm__ float* x, __gm__ float* y, int32_t batch, int32_t n_cols) {
  unsigned v5 = 1;
  unsigned v6 = 0;
  int32_t one = 1;
  int32_t zero = 0;
  int64_t v9 = 0;
  int64_t v10 = 32768;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t core_id = get_block_idx();
  int64_t local_aiv_id = get_subblockid();
  int64_t num_aiv_per_core = get_subblockdim();
  int64_t num_aiv_per_core_ = (int64_t) num_aiv_per_core;
  int64_t num_blocks = get_block_num();
  int32_t num_aiv_cores = (int32_t) ((int64_t) (uint64_t) ((int64_t) num_blocks) * (uint64_t) num_aiv_per_core_);
  int32_t rows_per_core_floor = batch / num_aiv_cores;
  int32_t rows_per_core = batch % num_aiv_cores != zero && batch < zero == num_aiv_cores < zero ? rows_per_core_floor + one : rows_per_core_floor;
  int32_t v19 = (int32_t) ((uint32_t) ((int32_t) (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) core_id) * (uint64_t) num_aiv_per_core_) + (uint64_t) ((int64_t) local_aiv_id))) * (uint32_t) v18);
  int32_t v20 = (int32_t) ((uint32_t) v19 + (uint32_t) v18);
  int32_t v21 = (int32_t) ((uint32_t) ((uint32_t) v20 < (uint32_t) batch ? v20 : batch) - (uint32_t) v19);
  int32_t v22 = (int32_t) ((uint32_t) batch * (uint32_t) n_cols);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  if (v21 > zero) {
    Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v23 = Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(n_cols);
    TASSIGN(v23, v9);
    Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v24 = Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(n_cols);
    TASSIGN(v24, v10);
    for (size_t v25 = (size_t) zero; v25 < ((size_t) v21); v25 += (size_t) one) {
      int32_t v26 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) v19 + (uint32_t) ((int32_t) v25)) * (uint32_t) n_cols);
      unsigned v27 = (unsigned) n_cols * v5;
      pto::Shape<1, 1, 1, 1, -1> v28 = pto::Shape<1, 1, 1, 1, -1>(n_cols);
      pto::Stride<-1, -1, -1, -1, 1> v29 = pto::Stride<-1, -1, -1, -1, 1>(v27, v27, v27, v27);
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v30 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(x + (v6 + (unsigned) v26 * (unsigned) one), v28, v29);
      unsigned v31 = (unsigned) n_cols * v5;
      pto::Shape<1, 1, 1, 1, -1> v32 = pto::Shape<1, 1, 1, 1, -1>(n_cols);
      pto::Stride<-1, -1, -1, -1, 1> v33 = pto::Stride<-1, -1, -1, -1, 1>(v31, v31, v31, v31);
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v34 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(y + (v6 + (unsigned) v26 * (unsigned) one), v32, v33);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      TLOAD(v23, v30);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      TABS(v24, v23);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      pipe_barrier(PIPE_MTE3);
      TSTORE(v34, v24);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    };
  }
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  #endif // __DAV_VEC__

  return;
}

