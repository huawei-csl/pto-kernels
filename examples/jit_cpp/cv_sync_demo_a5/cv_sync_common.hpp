#pragma once

#include <pto/pto-inst.hpp>

using namespace pto;

constexpr int32_t TILE_SIZE = 128;
constexpr int32_t HALF_TILE = 64;
constexpr int32_t VEC_NUM = 2;

constexpr uint16_t FLAG_READY = 0;
constexpr uint16_t FLAG_FREE = 1;
constexpr uint16_t FLAG_V2C_READY = 2;
constexpr uint16_t FLAG_V2C_FREE = 3;
constexpr uint16_t VEC_FLAG_OFFSET = 16;

constexpr uint32_t L1_0_OFFSET = 0u;
constexpr uint32_t L1_1_OFFSET = TILE_SIZE * TILE_SIZE * sizeof(half);
constexpr uint32_t L1_2_OFFSET = 2 * TILE_SIZE * TILE_SIZE * sizeof(half);
constexpr uint32_t L0_OFFSET = 0u;
constexpr uint32_t UB_0_OFFSET = 0u;
constexpr uint32_t UB_1_OFFSET = HALF_TILE * TILE_SIZE * sizeof(half);
constexpr uint32_t UB_2_OFFSET = 2 * HALF_TILE * TILE_SIZE * sizeof(half);
constexpr uint32_t C2V_FLOAT_UB_BASE = 0x20000u;

#ifdef __CCE_AICORE__

using TileL1 = Tile<TileType::Mat, half, TILE_SIZE, TILE_SIZE, BLayout::ColMajor, TILE_SIZE, TILE_SIZE,
                    SLayout::RowMajor, 512, PadValue::Zero>;
using TileL1Insert = Tile<TileType::Mat, half, TILE_SIZE, TILE_SIZE, BLayout::RowMajor, TILE_SIZE, TILE_SIZE,
                          SLayout::RowMajor, 512, PadValue::Zero>;
using TileL0A = TileLeft<half, TILE_SIZE, TILE_SIZE>;
using TileL0B = TileRight<half, TILE_SIZE, TILE_SIZE>;
using TileL0C = TileAcc<float, TILE_SIZE, TILE_SIZE>;
using TileVec = Tile<TileType::Vec, half, HALF_TILE, TILE_SIZE, BLayout::RowMajor, HALF_TILE, TILE_SIZE,
                     SLayout::NoneBox, 512, PadValue::Null>;
using TileVecNZ = Tile<TileType::Vec, half, HALF_TILE, TILE_SIZE, BLayout::ColMajor, HALF_TILE, TILE_SIZE,
                       SLayout::RowMajor, 512, PadValue::Null>;
using TileVecFloat = Tile<TileType::Vec, float, HALF_TILE, TILE_SIZE, BLayout::RowMajor, HALF_TILE, TILE_SIZE,
                          SLayout::NoneBox, 512, PadValue::Null>;

using TileGlobal = GlobalTensor<half, TileShape2D<half, TILE_SIZE, TILE_SIZE, Layout::ND>,
                                BaseShape2D<half, TILE_SIZE, TILE_SIZE, Layout::ND>, Layout::ND>;
using HalfTileGlobal = GlobalTensor<half, TileShape2D<half, HALF_TILE, TILE_SIZE, Layout::ND>,
                                    BaseShape2D<half, HALF_TILE, TILE_SIZE, Layout::ND>, Layout::ND>;
using HalfTileGlobalFloat = GlobalTensor<float, TileShape2D<float, HALF_TILE, TILE_SIZE, Layout::ND>,
                                         BaseShape2D<float, HALF_TILE, TILE_SIZE, Layout::ND>, Layout::ND>;

template <pipe_t Src, pipe_t Dst>
AICORE inline void SetFlag(uint32_t id)
{
    set_flag(Src, Dst, static_cast<event_t>(id));
}

template <pipe_t Src, pipe_t Dst>
AICORE inline void WaitFlag(uint32_t id)
{
    wait_flag(Src, Dst, static_cast<event_t>(id));
}

template <pipe_t Pipe>
AICORE inline void SignalBothVec(uint16_t flag)
{
    set_intra_block(Pipe, flag);
    set_intra_block(Pipe, flag + VEC_FLAG_OFFSET);
}

template <pipe_t Pipe>
AICORE inline void WaitBothVec(uint16_t flag)
{
    wait_intra_block(Pipe, flag);
    wait_intra_block(Pipe, flag + VEC_FLAG_OFFSET);
}

#endif

