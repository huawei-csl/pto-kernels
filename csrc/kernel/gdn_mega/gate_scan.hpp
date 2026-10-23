#pragma once

// Parallel prefix scan and layout conversion for the fused GDN pipeline.
// Seven vector additions replace the serial 128-row accumulation chain.
template <int H, int C>
AICORE inline void mega_gate_scan(__gm__ float* g, __gm__ half* beta,
                                 __gm__ float* sums, __gm__ float* transposed,
                                 __gm__ half* beta_t, __gm__ int32_t* cu,
                                 int64_t sequences, int64_t tokens)
{
#if defined(__DAV_VEC__)
    if (get_subblockid() != 0) return;
    set_mask_norm();
    set_vector_mask(-1, -1);
    constexpr int P = (H + 15) / 16 * 16;
    constexpr int Bytes = C * P * sizeof(float);
    using Shape2 = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using Stride2 = pto::Stride<1, 1, 1, DYNAMIC, 1>;
    using Full = Tile<TileType::Vec, float, C, P, BLayout::RowMajor, C, P, SLayout::NoneBox, 512, PadValue::Zero>;
    using Part = Tile<TileType::Vec, float, C, P, BLayout::RowMajor, DYNAMIC, DYNAMIC,
                      SLayout::NoneBox, 512, PadValue::Zero>;
    using Trans = Tile<TileType::Vec, float, P, C, BLayout::RowMajor, P, C>;
    using TransPart = Tile<TileType::Vec, float, P, C, BLayout::RowMajor, DYNAMIC, DYNAMIC>;
    Full buffers[2], tmp;
    TASSIGN(buffers[0], 0);
    TASSIGN(buffers[1], Bytes);
    TASSIGN(tmp, 3 * Bytes);
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
            GlobalTensor<float, Shape2, Stride2> gm(g + pos * H, {1, 1, 1, valid, H}, {1, 1, 1, H, 1});
            TLOAD(input, gm);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            TFILLPAD_INPLACE(buffers[0], input);
            PipeBarrierVec();
            int current = 0;
            for (int offset = 1; offset < C; offset *= 2) {
                Part prefix(offset, P), dest_prefix(offset, P);
                TASSIGN(prefix, current * Bytes);
                TASSIGN(dest_prefix, (1 - current) * Bytes);
                TMOV(dest_prefix, prefix);
                Part a(C - offset, P), b(C - offset, P), dest(C - offset, P);
                TASSIGN(a, current * Bytes + offset * P * sizeof(float));
                TASSIGN(b, current * Bytes);
                TASSIGN(dest, (1 - current) * Bytes + offset * P * sizeof(float));
                TADD(dest, a, b);
                PipeBarrierVec();
                current = 1 - current;
            }
            TTRANS(trans, buffers[current], tmp);
            PipeBarrierVec();
            Part sum_out(valid, H);
            TASSIGN(sum_out, current * Bytes);
            TransPart trans_out(H, valid);
            TASSIGN(trans_out, 2 * Bytes);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            GlobalTensor<float, Shape2, Stride2> sum_gm(sums + pos * H, {1, 1, 1, valid, H}, {1, 1, 1, H, 1});
            GlobalTensor<float, Shape2, Stride2> trans_gm(transposed + pos, {1, 1, 1, H, valid}, {1, 1, 1, tokens, 1});
            TSTORE(sum_gm, sum_out);
            TSTORE(trans_gm, trans_out);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            using HalfFull = Tile<TileType::Vec, half, C, P, BLayout::RowMajor, C, P, SLayout::NoneBox, 512, PadValue::Zero>;
            using HalfIn = Tile<TileType::Vec, half, C, P, BLayout::RowMajor, DYNAMIC, DYNAMIC,
                                SLayout::NoneBox, 512, PadValue::Zero>;
            using HalfTrans = Tile<TileType::Vec, half, P, C, BLayout::RowMajor, P, C>;
            using HalfOut = Tile<TileType::Vec, half, P, C, BLayout::RowMajor, DYNAMIC, DYNAMIC>;
            HalfIn beta_in(valid, H);
            HalfFull beta_full, beta_tmp;
            HalfTrans beta_trans;
            HalfOut beta_out(H, valid);
            TASSIGN(beta_in, 0); TASSIGN(beta_full, 0);
            TASSIGN(beta_trans, Bytes); TASSIGN(beta_out, Bytes);
            TASSIGN(beta_tmp, 2 * Bytes);
            GlobalTensor<half, Shape2, Stride2> beta_gm(beta + pos * H, {1, 1, 1, valid, H}, {1, 1, 1, H, 1});
            TLOAD(beta_in, beta_gm);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            TFILLPAD_INPLACE(beta_full, beta_in);
            PipeBarrierVec();
            TTRANS(beta_trans, beta_full, beta_tmp);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            GlobalTensor<half, Shape2, Stride2> beta_out_gm(beta_t + pos, {1, 1, 1, H, valid}, {1, 1, 1, tokens, 1});
            TSTORE(beta_out_gm, beta_out);
            pipe_barrier(PIPE_ALL);
        }
    }
#endif
}
