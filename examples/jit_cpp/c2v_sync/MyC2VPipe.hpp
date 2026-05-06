#pragma once

#include <pto/pto-inst.hpp>
#include <pto/npu/a2a3/TPush.hpp>
#include <pto/npu/a2a3/TPop.hpp>

using namespace pto;

// MyC2VPipe mirrors TPush/TPop call style while keeping the validated
// producer signal path on PIPE_MTE3 for this kernel.
template <uint8_t FlagID>
struct MyC2VPipe {
    struct Producer {
        AICORE inline void record() const
        {
            uint64_t config = TPipe<FlagID, FIFOType::GM_FIFO, 1, 1, TileAcc<float, 16, 16>,
                                    Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, 8, 16>>::getFFTSMsgCfg(
                TSyncCVMode::CV_CORES_SYNC, FlagID);
            ffts_cross_core_sync(PIPE_MTE3, config);
        }
    };

    struct Consumer {
        AICORE inline void wait() const
        {
            wait_flag_dev(FlagID);
        }
    };
};
