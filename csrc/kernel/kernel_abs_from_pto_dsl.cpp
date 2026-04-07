#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void _kernel(__gm__ float* x, __gm__ float* y, int32_t batch, int32_t n_cols) {
  unsigned uone = 1;
  unsigned uzero = 0;
  int32_t one = 1;
  int32_t zero = 0;
  int64_t UB_ZERO_ADDR = 0;
  int64_t UB_TILE_ADDR = 32768;
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
  int32_t row_offset_start = (int32_t) ((uint32_t) ((int32_t) (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) core_id) * (uint64_t) num_aiv_per_core_) + (uint64_t) ((int64_t) local_aiv_id))) * (uint32_t) rows_per_core);
  int32_t row_offset_end = (int32_t) ((uint32_t) row_offset_start + (uint32_t) rows_per_core);
  int32_t row_offset_end_clipped = (int32_t) ((uint32_t) ((uint32_t) row_offset_end < (uint32_t) batch ? row_offset_end : batch) - (uint32_t) row_offset_start);
  int32_t total_elements = (int32_t) ((uint32_t) batch * (uint32_t) n_cols);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  if (row_offset_end_clipped > zero) {
    Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> tile_x = Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(n_cols);
    TASSIGN(tile_x, UB_ZERO_ADDR);
    Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> tile_y = Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(n_cols);
    TASSIGN(tile_y, UB_TILE_ADDR);
    for (size_t i_row = (size_t) zero; i_row < ((size_t) row_offset_end_clipped); i_row += (size_t) one) {
      int32_t global_offset = (int32_t) ((uint32_t) ((int32_t) (uint32_t) row_offset_start + (uint32_t) ((int32_t) i_row)) * (uint32_t) n_cols);
      unsigned n_cols_ = (unsigned) n_cols * uone;
      pto::Shape<1, 1, 1, 1, -1> shape_x = pto::Shape<1, 1, 1, 1, -1>(n_cols);
      pto::Stride<-1, -1, -1, -1, 1> stride_x = pto::Stride<-1, -1, -1, -1, 1>(n_cols_, n_cols_, n_cols_, n_cols_);
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> global_x = GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(x + (uzero + (unsigned) global_offset * (unsigned) one), shape_x, stride_x);
      unsigned n_cols_ = (unsigned) n_cols * uone;
      pto::Shape<1, 1, 1, 1, -1> shape_y = pto::Shape<1, 1, 1, 1, -1>(n_cols);
      pto::Stride<-1, -1, -1, -1, 1> stride_y = pto::Stride<-1, -1, -1, -1, 1>(n_cols_, n_cols_, n_cols_, n_cols_);
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> global_y = GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(y + (uzero + (unsigned) global_offset * (unsigned) one), shape_y, stride_y);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      TLOAD(tile_x, global_x);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      TABS(tile_y, tile_x);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      pipe_barrier(PIPE_MTE3);
      TSTORE(global_y, tile_y);
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

