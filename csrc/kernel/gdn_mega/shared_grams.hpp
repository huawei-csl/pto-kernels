#pragma once

// Compute a key Gram matrix once per key head, concurrently with vector gate
// preparation. Packed row storage also fits arbitrarily short sequence tails.
template <int H, int C, int D>
AICORE inline void mega_shared_grams_serial(
    __gm__ half* k, __gm__ float* scratch, __gm__ float* scratch_upper,
    __gm__ int32_t* cu, uint32_t matrices, int key_heads, __gm__ half* q,
    __gm__ float* qk_out) {
#if defined(__DAV_CUBE__)
  using Mat = Tile<TileType::Mat, half, C, D, BLayout::ColMajor, C, D,
                   SLayout::RowMajor, 512>;
  using MatTail = Tile<TileType::Mat, half, C, D, BLayout::ColMajor, DYNAMIC,
                       DYNAMIC, SLayout::RowMajor, 512, PadValue::Zero>;
  using MatT = Tile<TileType::Mat, half, D, C, BLayout::RowMajor, D, C,
                    SLayout::ColMajor, 512>;
  using Shape2 = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using Stride2 = pto::Stride<1, 1, 1, DYNAMIC, 1>;
  Mat key;
  MatT kt;
  TileLeft<half, C, D> left;
  TileRight<half, D, C> right;
  TileAcc<float, C, C> acc;
  TASSIGN(key, 0);
  TASSIGN(left, 0);
  TASSIGN(right, 0);
  TASSIGN(acc, 0);
  for (uint32_t m = get_block_idx(); m < matrices / H * key_heads;
       m += get_block_num()) {
    auto info = mega_chunk_info(m, key_heads, C, cu);
    int valid = info.valid_size;
    int head = m % key_heads;
    int64_t pos = (info.bsnd_offset - head * C) / (key_heads * C);
    MatTail tail(valid, D);
    TASSIGN(tail, 0);
    GlobalTensor<half, Shape2, Stride2> input(k + (pos * key_heads + head) * D,
                                              {1, 1, 1, valid, D},
                                              {1, 1, 1, key_heads * D, 1});
    TLOAD(tail, input);
    pipe_barrier(PIPE_ALL);
    if (valid != C) TFILLPAD(tail, tail);
    TRESHAPE(kt, key);
    TEXTRACT(left, key, 0, 0);
    TEXTRACT(right, kt, 0, 0);
    pipe_barrier(PIPE_ALL);
    TMATMUL(acc, left, right);
    pipe_barrier(PIPE_ALL);
    TileAcc<float, C, C, DYNAMIC, DYNAMIC> result(valid, C);
    TASSIGN(result, 0);
    TileAcc<float, C, C / 2, DYNAMIC, C / 2> low(valid), high(valid);
    TASSIGN(low, 0);
    TASSIGN(high, C * C / 2 * sizeof(float));
    GlobalTensor<float, Shape2, Stride2> output(
        scratch + info.bsnd_offset / 2, {1, 1, 1, valid, C / 2},
        {1, 1, 1, key_heads * C / 2, 1});
    GlobalTensor<float, Shape2, Stride2> upper(
        scratch_upper + info.bsnd_offset / 2, {1, 1, 1, valid, C / 2},
        {1, 1, 1, key_heads * C / 2, 1});
    TSTORE(output, low);
    TSTORE(upper, high);
    pipe_barrier(PIPE_ALL);
    GlobalTensor<half, Shape2, Stride2> query_gm(
        q + (pos * key_heads + head) * D, {1, 1, 1, valid, D},
        {1, 1, 1, key_heads * D, 1});
    TLOAD(tail, query_gm);
    pipe_barrier(PIPE_ALL);
    if (valid != C) TFILLPAD(tail, tail);
    TEXTRACT(left, key, 0, 0);
    pipe_barrier(PIPE_ALL);
    // K^T remains in L0B while the Q tile replaces K in L1/L0A.
    TMATMUL(acc, left, right);
    pipe_barrier(PIPE_ALL);
    GlobalTensor<float, Shape2, Stride2> qk_gm(qk_out + info.bsnd_offset,
                                               {1, 1, 1, valid, C},
                                               {1, 1, 1, key_heads * C, 1});
    TSTORE(qk_gm, result);
    pipe_barrier(PIPE_ALL);
  }
#endif
}

// Share both raw Gram products across grouped value heads. Prefetch the next
// K/Q pair while the current pair is on the cube, and export K*K^T while Q*K^T
// computes. L1 inputs can be recycled as soon as MTE1 has read them.
template <int C, int D>
AICORE inline void mega_load_gram_pair(__gm__ half* k, __gm__ half* q,
                                       int64_t offset, int valid,
                                       int key_heads) {
#if defined(__DAV_CUBE__)
  using Tail = Tile<TileType::Mat, half, C, D, BLayout::ColMajor, DYNAMIC,
                    DYNAMIC, SLayout::RowMajor, 512, PadValue::Zero>;
  using S = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using Str = pto::Stride<1, 1, 1, DYNAMIC, 1>;
  Tail key(valid, D), query(valid, D);
  TASSIGN(key, 0);
  TASSIGN(query, C * D * sizeof(half));
  GlobalTensor<half, S, Str> kgm(k + offset, {1, 1, 1, valid, D},
                                 {1, 1, 1, key_heads * D, 1});
  GlobalTensor<half, S, Str> qgm(q + offset, {1, 1, 1, valid, D},
                                 {1, 1, 1, key_heads * D, 1});
  TLOAD(key, kgm);
  TLOAD(query, qgm);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif
}

template <int H, int C, int D>
AICORE inline void mega_shared_grams_pipelined(
    __gm__ half* k, __gm__ float* scratch, __gm__ float* scratch_upper,
    __gm__ int32_t* cu, uint32_t matrices, int key_heads, __gm__ half* q,
    __gm__ float* qk_out) {
#if defined(__DAV_CUBE__)
  using Mat = Tile<TileType::Mat, half, C, D, BLayout::ColMajor, C, D,
                   SLayout::RowMajor, 512>;
  using Tail = Tile<TileType::Mat, half, C, D, BLayout::ColMajor, DYNAMIC,
                    DYNAMIC, SLayout::RowMajor, 512, PadValue::Zero>;
  using Trans = Tile<TileType::Mat, half, D, C, BLayout::RowMajor, D, C,
                     SLayout::ColMajor, 512>;
  using S = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using Str = pto::Stride<1, 1, 1, DYNAMIC, 1>;
  Mat key, query;
  Trans kt;
  TileLeft<half, C, D> left, qleft;
  TileRight<half, D, C> right;
  TileAcc<float, C, C> kk, qk;
  TASSIGN(key, 0);
  TASSIGN(query, C * D * sizeof(half));
  TASSIGN(left, 0);
  TASSIGN(qleft, C * D * sizeof(half));
  TASSIGN(right, 0);
  TASSIGN(kk, 0);
  TASSIGN(qk, C * C * sizeof(float));
  uint32_t first = get_block_idx(), count = matrices / H * key_heads,
           step = get_block_num();
  if (first >= count) return;
  auto first_info = mega_chunk_info(first, key_heads, C, cu);
  mega_load_gram_pair<C, D>(k, q, first_info.bsnd_offset / C * D,
                            first_info.valid_size, key_heads);
  for (uint32_t m = first; m < count; m += step) {
    auto info = mega_chunk_info(m, key_heads, C, cu);
    int valid = info.valid_size;
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    if (valid != C) {
      Tail ktail(valid, D), qtail(valid, D);
      TASSIGN(ktail, 0);
      TASSIGN(qtail, C * D * sizeof(half));
      pipe_barrier(PIPE_ALL);
      TFILLPAD(ktail, ktail);
      TFILLPAD(qtail, qtail);
    }
    if (m != first) wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    TRESHAPE(kt, key);
    TMOV(left, key);
    TEXTRACT(right, kt, 0, 0);
    TMOV(qleft, query);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    if (m + step < count) {
      auto next = mega_chunk_info(m + step, key_heads, C, cu);
      mega_load_gram_pair<C, D>(k, q, next.bsnd_offset / C * D, next.valid_size,
                                key_heads);
    }
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    if (m != first) wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    TMATMUL(kk, left, right);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TMATMUL(qk, qleft, right);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    TileAcc<float, C, C, DYNAMIC, DYNAMIC> kresult(valid, C), qresult(valid, C);
    TASSIGN(kresult, 0);
    TASSIGN(qresult, C * C * sizeof(float));
    TileAcc<float, C, C / 2, DYNAMIC, C / 2> low(valid), high(valid);
    TASSIGN(low, 0);
    TASSIGN(high, C * C / 2 * sizeof(float));
    GlobalTensor<float, S, Str> kgm(scratch + info.bsnd_offset / 2,
                                    {1, 1, 1, valid, C / 2},
                                    {1, 1, 1, key_heads * C / 2, 1});
    GlobalTensor<float, S, Str> upper(scratch_upper + info.bsnd_offset / 2,
                                      {1, 1, 1, valid, C / 2},
                                      {1, 1, 1, key_heads * C / 2, 1});
    GlobalTensor<float, S, Str> qgm(qk_out + info.bsnd_offset,
                                    {1, 1, 1, valid, C},
                                    {1, 1, 1, key_heads * C, 1});
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TSTORE(kgm, low);
    TSTORE(upper, high);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
    TSTORE(qgm, qresult);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  }
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  pipe_barrier(PIPE_ALL);
#endif
}

template <int H, int C, int D>
AICORE inline void mega_shared_grams(__gm__ half* k, __gm__ float* scratch,
                                     __gm__ float* scratch_upper,
                                     __gm__ int32_t* cu, uint32_t matrices,
                                     int key_heads, __gm__ half* q,
                                     __gm__ float* qk_out) {
#if defined(__DAV_CUBE__)
  // A single wave has no next pair to prefetch; retain its shorter setup.
  if (matrices / H * key_heads > get_block_num()) {
    mega_shared_grams_pipelined<H, C, D>(k, scratch, scratch_upper, cu,
                                         matrices, key_heads, q, qk_out);
    return;
  }
  mega_shared_grams_serial<H, C, D>(k, scratch, scratch_upper, cu, matrices,
                                    key_heads, q, qk_out);
#endif
}
