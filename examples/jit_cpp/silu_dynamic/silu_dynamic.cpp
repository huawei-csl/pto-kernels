#include <pto/pto-inst.hpp>
using namespace pto;

constexpr uint32_t BUFFER_NUM      = 2;
constexpr uint32_t UB_ALLOC_BYTES  = 48 * 1024;
constexpr uint32_t ELEMENTS_PER_TILE = UB_ALLOC_BYTES / 2;

constexpr unsigned X_PING   = 0x00000;
constexpr unsigned X_PONG   = (X_PING + 0x8000 + 0x100);
constexpr unsigned CAL_PING = 0x10000;
constexpr unsigned CAL_PONG = (CAL_PING + 0x8000 + 0x100);

// SiLU: y = x / (1 + exp(-x))
template <typename T>
AICORE void runTSilu(__gm__ T *y, __gm__ T *x, uint32_t num_elements)
{
#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)

    set_mask_norm();
    set_vector_mask(-1, -1);

    const uint32_t num_cores = block_num;
    const uint32_t elements_per_core = (num_elements + num_cores - 1) / num_cores; // ceil
    const uint32_t offset_this_core  = elements_per_core * block_idx;

    if (offset_this_core >= num_elements) return;

    uint32_t elements_to_process = elements_per_core;
    if (offset_this_core + elements_to_process > num_elements) {
        elements_to_process = num_elements - offset_this_core;
    }
    if (elements_to_process == 0) return;

    using ShapeDim5 = pto::Shape<1, 1, 1, 1, ELEMENTS_PER_TILE>;
    using StridDim5 = pto::Stride<1, 1, 1, 1, 1>;
    using GlobalData = pto::GlobalTensor<T, ShapeDim5, StridDim5>;

    GlobalData xGlobal(x + offset_this_core);
    GlobalData yGlobal(y + offset_this_core);

    using TileData = Tile<TileType::Vec, T,
                          1, ELEMENTS_PER_TILE,
                          BLayout::RowMajor,
                          -1, -1>;

    set_flag(PIPE_V,    PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V,    PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_V,    EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V,    EVENT_ID1);

    uint32_t x_offset = 0;
    uint32_t y_offset = 0;

    for (uint32_t num_processed = 0, ping = 1;
         num_processed < elements_to_process;
         num_processed += ELEMENTS_PER_TILE)
    {
        const uint32_t remaining = elements_to_process - num_processed;
        const uint32_t cur_cols  = (remaining >= ELEMENTS_PER_TILE) ? ELEMENTS_PER_TILE : remaining;

        const int8_t  buf = ping ? 0 : 1;
        const event_t ev  = ping ? (event_t)EVENT_ID0 : (event_t)EVENT_ID1;

        TileData xTile(1, cur_cols);
        TileData calTile(1, cur_cols);

        if (buf == 0) {
            TASSIGN(xTile,  X_PING);
            TASSIGN(calTile, CAL_PING);
        } else {
            TASSIGN(xTile,  X_PONG);
            TASSIGN(calTile, CAL_PONG);
        }

        TASSIGN(xGlobal, (x + offset_this_core + x_offset));
        TASSIGN(yGlobal, (y + offset_this_core + y_offset));

        wait_flag(PIPE_V, PIPE_MTE2, ev);
        TLOAD(xTile, xGlobal);
        pipe_barrier(PIPE_ALL);

        set_flag(PIPE_MTE2, PIPE_V, ev);
        wait_flag(PIPE_MTE2, PIPE_V, ev);

        wait_flag(PIPE_MTE3, PIPE_V, ev);

        TMULS(calTile, xTile, (T)-1);
        pipe_barrier(PIPE_ALL);

        TEXP(calTile, calTile);
        pipe_barrier(PIPE_ALL);

        TADDS(calTile, calTile, (T)1);
        pipe_barrier(PIPE_ALL);

        TDIV(calTile, xTile, calTile);
        pipe_barrier(PIPE_ALL);

        set_flag(PIPE_V, PIPE_MTE3, ev);
        wait_flag(PIPE_V, PIPE_MTE3, ev);

        TSTORE(yGlobal, calTile);
        pipe_barrier(PIPE_ALL);

        set_flag(PIPE_MTE3, PIPE_V, ev);
        set_flag(PIPE_V, PIPE_MTE2, ev);

        x_offset += cur_cols;
        y_offset += cur_cols;
        ping = 1 - ping;
    }

    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

#else
    // Cube branch: do nothing
#endif
}

__global__ AICORE void silu_custom(__gm__ void *x, __gm__ void *y, uint32_t num_elements)
{
    runTSilu<half>((__gm__ half *)y, (__gm__ half *)x, num_elements);
}

extern "C" void call_kernel(uint32_t blockDim, void *stream,
                            uint8_t *x, uint8_t *y, uint32_t num_elements)
{
    silu_custom<<<blockDim, nullptr, stream>>>(x, y, num_elements);
}
