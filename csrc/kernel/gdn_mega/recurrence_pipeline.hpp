#pragma once

// Schedule the six products as one dependency graph. Each L0 operand bank
// holds two independent 128x128 tiles; the two FP32 accumulators fill L0C.
template <int H, int C, typename Mat, typename Acc>
AICORE inline void mega_recurrence_pipeline(Mat input, Mat u, Mat inverse, Mat state,
    Mat kt, Mat query, Mat attention, Mat residual, Mat decay, Acc acc, Acc out,
    __gm__ half* output, int valid, bool decay_zero)
{
#if defined(__DAV_CUBE__)
    constexpr int Bytes = C * C * sizeof(half);
    TileLeft<half, C, C> left, next_left;
    TileRight<half, C, C> right, next_right;
    TASSIGN(left, 0); TASSIGN(next_left, Bytes);
    TASSIGN(right, 0); TASSIGN(next_right, Bytes);
    TMOV(left, inverse); TMOV(right, u); TMOV(next_right, input);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0); wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    TMATMUL(acc, left, right);
    TMATMUL(out, left, next_right);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TMOV(input, out);
    set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    TMOV(right, state); TMOV(next_left, query);
    wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
    TMOV(left, input);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0); wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    TMATMUL_ACC(acc, acc, left, right); // R = U - W*S.
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    TMATMUL(out, next_left, right); // Q*S overlaps residual export.
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TMOV(residual, acc);
    set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    if (!decay_zero) {
        // The same schedule also retains finite incoming-state decay.
        // S remains resident in right while residual uses next_right.
        TMOV(left, decay);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0); wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        TMATMUL(acc, left, right);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0); wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    }
    using Trans = Tile<TileType::Mat, half, C, C, BLayout::RowMajor, C, C, SLayout::ColMajor, 512>;
    Trans kt_trans; TRESHAPE(kt_trans, kt);
    TEXTRACT(left, kt_trans, 0, 0);
    wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
    TMOV(next_right, residual);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0); wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    if (decay_zero) {
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        TMATMUL(acc, left, next_right);
    } else {
        TMATMUL_ACC(acc, acc, left, next_right);
    }
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    TMOV(next_left, attention);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0); wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
    TMATMUL_ACC(out, out, next_left, next_right);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TMOV(state, acc); // State export overlaps the final output product.
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
    TileAcc<float, C, C, DYNAMIC, C> tail(valid); TASSIGN(tail, C * C * sizeof(float));
    mega_store_packed<C, C, H * C>(out, tail, output, valid, C);
    pipe_barrier(PIPE_ALL);
#endif
}
