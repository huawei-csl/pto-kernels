#pragma once

// Order only the engines used by the on-chip inverse. Independent operand
// DMA on MTE2 can progress while these cube/FIX/MTE1 dependencies resolve.
AICORE inline void mega_inverse_sync()
{
#if defined(__DAV_CUBE__)
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0); wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0); wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0); wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
#else
    pipe_barrier(PIPE_ALL);
#endif
}

// Two independent diagonal solves share an instruction schedule, not data.
// Queue independent cube work before waiting for FIX exports, preserving the
// original FP32 accumulators and FP16 rounding at every polynomial step.
template <typename Mat>
AICORE inline void mega_inverse_interleaved(Mat input, Mat identity, int terms)
{
#if defined(__DAV_CUBE__)
    constexpr int B = 64, Bytes = B * B * sizeof(half), Scratch = 5 * 128 * 128 * sizeof(half);
    using Small = Tile<TileType::Mat, half, B, B, BLayout::ColMajor, B, B, SLayout::RowMajor, 512>;
    Small x0, x1, p0, p1;
    TASSIGN(x0, Scratch); TASSIGN(x1, Scratch + Bytes);
    TASSIGN(p0, Scratch + 2 * Bytes); TASSIGN(p1, Scratch + 3 * Bytes);
    TileLeft<half, B, B> a0, a1, power0, power1;
    TileRight<half, B, B> b0, b1, unit;
    TileAcc<float, B, B> X0, X1, P0, P1;
    TASSIGN(a0, 0); TASSIGN(a1, Bytes);
    TASSIGN(power0, 2 * Bytes); TASSIGN(power1, 3 * Bytes);
    TASSIGN(b0, 0); TASSIGN(b1, Bytes); TASSIGN(unit, 2 * Bytes);
    TASSIGN(X0, 0); TASSIGN(X1, 2 * Bytes);
    TASSIGN(P0, 4 * Bytes); TASSIGN(P1, 6 * Bytes);
    TEXTRACT(a0, input, 0, 0); TEXTRACT(b0, input, 0, 0);
    TEXTRACT(a1, input, B, B); TEXTRACT(b1, input, B, B);
    TEXTRACT(power0, identity, 0, 0); TEXTRACT(unit, identity, 0, 0);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0); wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    TMATMUL(P0, a0, b0); set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TMATMUL(P1, a1, b1); set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
    TMATMUL(X0, a0, unit);
    TMATMUL(X1, a1, unit);
    TMATMUL_ACC(X0, X0, power0, unit); set_flag(PIPE_M, PIPE_FIX, EVENT_ID2);
    TMATMUL_ACC(X1, X1, power0, unit); set_flag(PIPE_M, PIPE_FIX, EVENT_ID3);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0); TMOV(p0, P0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1); TMOV(p1, P1);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID2); TMOV(x0, X0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID3); TMOV(x1, X1);
    set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0); wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
    for (int degree = 2; degree < terms; degree *= 2) {
        TMOV(a0, x0); TMOV(a1, x1); TMOV(b0, p0); TMOV(b1, p1);
        if (degree * 2 < terms) { TMOV(power0, p0); TMOV(power1, p1); }
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0); wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        TMATMUL_ACC(X0, X0, a0, b0); set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        TMATMUL_ACC(X1, X1, a1, b1); set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
        if (degree * 2 < terms) {
            TMATMUL(P0, power0, b0); set_flag(PIPE_M, PIPE_FIX, EVENT_ID2);
            TMATMUL(P1, power1, b1); set_flag(PIPE_M, PIPE_FIX, EVENT_ID3);
        }
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        if (degree * 2 < terms) TMOV(x0, X0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
        if (degree * 2 < terms) {
            TMOV(x1, X1);
            wait_flag(PIPE_M, PIPE_FIX, EVENT_ID2); TMOV(p0, P0);
            wait_flag(PIPE_M, PIPE_FIX, EVENT_ID3); TMOV(p1, P1);
        }
        set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0); wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
    }
    TINSERT(input, X0, 0, 0); TINSERT(input, X1, B, B);
    set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0); wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
#endif
}

template <int B, typename Mat>
AICORE inline void mega_inverse_join(Mat negative_input, Mat result, int offset)
{
#if defined(__DAV_CUBE__)
    using Small = Tile<TileType::Mat, half, B, B, BLayout::ColMajor, B, B, SLayout::RowMajor, 512>;
    Small tmp;
    TASSIGN(tmp, 5 * 128 * 128 * sizeof(half));
    TileLeft<half, B, B> a;
    TileRight<half, B, B> b;
    TileAcc<float, B, B> c;
    TASSIGN(a, 0); TASSIGN(b, 0); TASSIGN(c, 0);
    TEXTRACT(a, result, offset + B, offset + B);
    TEXTRACT(b, negative_input, offset + B, offset);
    mega_inverse_sync();
    TMATMUL(c, a, b); mega_inverse_sync();
    TMOV(tmp, c); mega_inverse_sync();
    TMOV(a, tmp); TEXTRACT(b, result, offset, offset); mega_inverse_sync();
    TMATMUL(c, a, b); mega_inverse_sync();
    TINSERT(result, c, offset + B, offset); mega_inverse_sync();
#endif
}
