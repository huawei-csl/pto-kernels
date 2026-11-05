// Copyright (c) 2026 Huawei Technologies Co., Ltd. See repository LICENSE.
#pragma once
#include "primitives.h"

// Seven 64-KiB L1 slots retain native FP32 operands across one chunk's adjoints.
// Products finish at the caller's cross-core phase boundary. Loads to disjoint
// slots can follow product() without a full barrier and overlap M/FIX work.
struct ResidentCube {
  template<int Slot>
  AICORE void load(__gm__ float* input) {
#if defined(__DAV_CUBE__)
    Tile<TileType::Mat,float,128,128,BLayout::ColMajor,128,128,SLayout::RowMajor,512> value;
    TASSIGN(value,Slot*65536);
    GlobalTensor<float,Shape<1,1,1,128,128>,Stride<1,1,1,128,1>> gm(input);
    TLOAD(value,gm);
#endif
  }
  template<int A,int B,bool TA=false,bool TB=false>
  AICORE void product(__gm__ float* output) {
#if defined(__DAV_CUBE__)
    using Mat=Tile<TileType::Mat,float,128,128,BLayout::ColMajor,128,128,SLayout::RowMajor,512>;
    using Trans=Tile<TileType::Mat,float,128,128,BLayout::RowMajor,128,128,SLayout::ColMajor,512>;
    Mat a,b; Trans at,bt; TileLeft<float,128,128> left;
    TileRight<float,128,128> right; TileAcc<float,128,128> acc;
    TASSIGN(a,A*65536); TASSIGN(b,B*65536);
    TASSIGN(left,0); TASSIGN(right,0); TASSIGN(acc,0);
    set_flag(PIPE_MTE2,PIPE_MTE1,EVENT_ID0); wait_flag(PIPE_MTE2,PIPE_MTE1,EVENT_ID0);
    if constexpr(TA) { TRESHAPE(at,a); TEXTRACT(left,at,0,0); }
    else TEXTRACT(left,a,0,0);
    if constexpr(TB) { TRESHAPE(bt,b); TEXTRACT(right,bt,0,0); }
    else TEXTRACT(right,b,0,0);
    set_flag(PIPE_MTE1,PIPE_M,EVENT_ID0); wait_flag(PIPE_MTE1,PIPE_M,EVENT_ID0);
    TMATMUL(acc,left,right);
    set_flag(PIPE_M,PIPE_FIX,EVENT_ID0); wait_flag(PIPE_M,PIPE_FIX,EVENT_ID0);
    GlobalTensor<float,Shape<1,1,1,128,128>,Stride<1,1,1,128,1>> gm(output);
    TSTORE(gm,acc);
#endif
  }
};
