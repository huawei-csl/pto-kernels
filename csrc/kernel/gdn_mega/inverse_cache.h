#pragma once

// One retained cache. FP32 may alias the working exchange; 16-bit caches are
// exported once from completed, double-buffered per-core scratch.
struct MegaInverseCache {
  __gm__ uint8_t* data;
  int dtype; // 0=FP16, 1=BF16, 2=FP32
  __gm__ float* second = nullptr; // Optional separate lower half of working scratch.
  AICORE MegaInverseCache at(int64_t elements) const {
    return {data ? data + elements * (dtype == 2 ? 4 : 2) : nullptr, dtype};
  }
  AICORE __gm__ float* native() const {
    return dtype == 2 ? reinterpret_cast<__gm__ float*>(data) : nullptr;
  }
  AICORE __gm__ float* native_at(int offset) const {
    return second && offset >= 8192 ? second + offset - 8192 : native() + offset;
  }
  template<typename T>
  AICORE void export_as(MegaInverseCache working) const {
#if defined(__DAV_VEC__)
    constexpr int R=64,C=128;
    int offset=get_subblockid()*R*C;
    Tile<TileType::Vec,float,R,C> value; TASSIGN(value,96*1024);
    Tile<TileType::Vec,T,R,C> rounded; TASSIGN(rounded,0);
    GlobalTensor<float,Shape<1,1,1,R,C>,pto::Stride<1,1,1,C,1>> source(working.native_at(offset));
    GlobalTensor<T,Shape<1,1,1,R,C>,pto::Stride<1,1,1,C,1>> target(
        reinterpret_cast<__gm__ T*>(data)+offset);
    TLOAD(value,source); pipe_barrier(PIPE_ALL);
    TCVT(rounded,value,RoundMode::CAST_RINT); pipe_barrier(PIPE_ALL);
    TSTORE(target,rounded); pipe_barrier(PIPE_ALL);
#endif
  }
  AICORE void export_from(MegaInverseCache working) const {
    if(dtype==0) export_as<half>(working);
    else export_as<bfloat16_t>(working);
  }
};
