#pragma once

// Parallel prefix scan and layout conversion for the fused GDN pipeline.
// Seven vector additions replace the serial 128-row accumulation chain.
template <int H, int C>
AICORE inline void mega_gate_scan(__gm__ float* g, __gm__ half* beta,
                                  __gm__ float* sums, __gm__ float* transposed,
                                  __gm__ half* beta_t, __gm__ int32_t* cu,
                                  int64_t sequences, int64_t tokens) {
#if defined(__DAV_VEC__)
  if (get_subblockid() != 0) return;
  set_mask_norm();
  set_vector_mask(-1, -1);
  constexpr int P = (H + 15) / 16 * 16;
  constexpr int Bytes = C * P * sizeof(float);
  using Shape2 = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using Stride2 = pto::Stride<1, 1, 1, DYNAMIC, 1>;
  using Full = Tile<TileType::Vec, float, C, P, BLayout::RowMajor, C, P,
                    SLayout::NoneBox, 512, PadValue::Zero>;
  using Part = Tile<TileType::Vec, float, C, P, BLayout::RowMajor, DYNAMIC,
                    DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;
  using Trans = Tile<TileType::Vec, float, P, C, BLayout::RowMajor, P, C>;
  using TransPart =
      Tile<TileType::Vec, float, P, C, BLayout::RowMajor, DYNAMIC, DYNAMIC>;
  Full buffers[2], lows[2], tmp;
  TASSIGN(buffers[0], 0);
  TASSIGN(buffers[1], Bytes);
  TASSIGN(lows[0], 2 * Bytes);
  TASSIGN(lows[1], 3 * Bytes);
  TASSIGN(tmp, 5 * Bytes);
  Trans trans;
  TASSIGN(trans, 2 * Bytes);
  int64_t chunk_index = 0;
  for (int64_t s = 0; s < sequences; ++s) {
    int64_t start = cu[s], end = cu[s + 1];
    for (int64_t pos = start; pos < end; pos += C, ++chunk_index) {
      if (chunk_index % get_block_num() != get_block_idx()) continue;
      int valid = static_cast<int>(min(static_cast<int64_t>(C), end - pos));
      Part input(valid, H);
      TASSIGN(input, 0);
      GlobalTensor<float, Shape2, Stride2> gm(g + pos * H, {1, 1, 1, valid, H},
                                              {1, 1, 1, H, 1});
      TLOAD(input, gm);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      TFILLPAD_INPLACE(buffers[0], input);
      PipeBarrierVec();
      TEXPANDS(lows[0], 0.0f);
      PipeBarrierVec();
      int current = 0;
      for (int offset = 1; offset < C; offset *= 2) {
        Part prefix(offset, P), dest_prefix(offset, P);
        TASSIGN(prefix, current * Bytes);
        TASSIGN(dest_prefix, (1 - current) * Bytes);
        TMOV(dest_prefix, prefix);
        Part lp(offset, P), ld(offset, P);
        TASSIGN(lp, (2 + current) * Bytes);
        TASSIGN(ld, (3 - current) * Bytes);
        TMOV(ld, lp);
        Part a(C - offset, P), b(C - offset, P), dest(C - offset, P);
        TASSIGN(a, current * Bytes + offset * P * sizeof(float));
        TASSIGN(b, current * Bytes);
        TASSIGN(dest, (1 - current) * Bytes + offset * P * sizeof(float));
        TADD(dest, a, b);
        PipeBarrierVec();
        // Error-free TwoSum of the high parts, plus their low parts.
        Part ar(C - offset, P), br(C - offset, P), la(C - offset, P),
            lb(C - offset, P), lo(C - offset, P);
        TASSIGN(ar, 4 * Bytes);
        TASSIGN(br, 5 * Bytes);
        TASSIGN(la, (2 + current) * Bytes + offset * P * sizeof(float));
        TASSIGN(lb, (2 + current) * Bytes);
        TASSIGN(lo, (3 - current) * Bytes + offset * P * sizeof(float));
        TSUB(br, dest, a);
        PipeBarrierVec();
        TSUB(ar, dest, br);
        PipeBarrierVec();
        TSUB(ar, a, ar);
        TSUB(br, b, br);
        PipeBarrierVec();
        TADD(ar, ar, br);
        TADD(lo, la, lb);
        PipeBarrierVec();
        TADD(lo, lo, ar);
        PipeBarrierVec();
        current = 1 - current;
      }
      // Reuse the dead ping-pong slots for transpose output/scratch.
      TASSIGN(trans, (1 - current) * Bytes);
      TASSIGN(tmp, (3 - current) * Bytes);
      TransPart trans_out(H, valid);
      TASSIGN(trans_out, (1 - current) * Bytes);
      GlobalTensor<float, Shape2, Stride2> high_gm(
          transposed + pos, {1, 1, 1, H, valid}, {1, 1, 1, tokens, 1});
      GlobalTensor<float, Shape2, Stride2> low_gm(
          sums + pos, {1, 1, 1, H, valid}, {1, 1, 1, tokens, 1});
      TTRANS(trans, buffers[current], tmp);
      pipe_barrier(PIPE_ALL);
      TSTORE(high_gm, trans_out);
      pipe_barrier(PIPE_ALL);
      TTRANS(trans, lows[current], tmp);
      pipe_barrier(PIPE_ALL);
      TSTORE(low_gm, trans_out);
      pipe_barrier(PIPE_ALL);
      using HalfFull = Tile<TileType::Vec, half, C, P, BLayout::RowMajor, C, P,
                            SLayout::NoneBox, 512, PadValue::Zero>;
      using HalfIn = Tile<TileType::Vec, half, C, P, BLayout::RowMajor, DYNAMIC,
                          DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;
      using HalfTrans =
          Tile<TileType::Vec, half, P, C, BLayout::RowMajor, P, C>;
      using HalfOut =
          Tile<TileType::Vec, half, P, C, BLayout::RowMajor, DYNAMIC, DYNAMIC>;
      HalfIn beta_in(valid, H);
      HalfFull beta_full, beta_tmp;
      HalfTrans beta_trans;
      HalfOut beta_out(H, valid);
      TASSIGN(beta_in, 0);
      TASSIGN(beta_full, 0);
      TASSIGN(beta_trans, Bytes);
      TASSIGN(beta_out, Bytes);
      TASSIGN(beta_tmp, 2 * Bytes);
      GlobalTensor<half, Shape2, Stride2> beta_gm(
          beta + pos * H, {1, 1, 1, valid, H}, {1, 1, 1, H, 1});
      TLOAD(beta_in, beta_gm);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      TFILLPAD_INPLACE(beta_full, beta_in);
      PipeBarrierVec();
      TTRANS(beta_trans, beta_full, beta_tmp);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      GlobalTensor<half, Shape2, Stride2> beta_out_gm(
          beta_t + pos, {1, 1, 1, H, valid}, {1, 1, 1, tokens, 1});
      TSTORE(beta_out_gm, beta_out);
      pipe_barrier(PIPE_ALL);
    }
  }
#endif
}
