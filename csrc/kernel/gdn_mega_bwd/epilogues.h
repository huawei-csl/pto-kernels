// Copyright (c) 2026 Huawei Technologies Co., Ltd. See repository LICENSE.
#pragma once
#include "primitives.h"

#ifdef __CCE_AICORE__

// The upper UB holds the shared-key sums across value heads. All helpers
// below stay below 120 KiB; 184..192 KiB is reserved by PTO.
struct GradientOutput {
  __gm__ float *dq=nullptr,*dk=nullptr,*dv=nullptr;
  int length=0,key_stride=0,value_stride=0;
  float scale=1; bool first=false,last=false;
};
template<int Address>
AICORE inline void scatter_local(__gm__ float* out,int length,int stride) {
#if defined(__DAV_VEC__)
  int row=get_subblockid()*64,n=max(0,min(64,length-row));
  if(!n) return;
  Tile<TileType::Vec,float,64,128,BLayout::RowMajor,DYNAMIC,128> part(n);
  TASSIGN(part,Address);
  GlobalTensor<float,Shape<1,1,1,DYNAMIC,128>,Stride<1,1,1,DYNAMIC,1>> gm(
    out+row*stride,{1,1,1,n,128},{1,1,1,stride,1});
  TSTORE(gm,part); sync();
#endif
}
template<int Which>
AICORE inline void accumulate_gradient(Half& value,const GradientOutput& output) {
#if defined(__DAV_VEC__)
  constexpr int Address=(120+Which*32)*1024;
  Half acc; TASSIGN(acc,Address);
  if constexpr(Which==0) { TMULS(value,value,output.scale); sync(); }
  if(output.first) TMOV(acc,value);
  else TADD(acc,acc,value);
  sync();
  if(output.last) scatter_local<Address>(Which==0 ? output.dq : output.dk,output.length,output.key_stride);
#endif
}

// V is used only by vector epilogues. Load and widen the original input in UB
// instead of retaining an FP32 copy in GM for the entire backward.
template<class T,int Address>
AICORE inline void load_value(const Input& input,int first,int head,int length,Half& out) {
#if defined(__DAV_VEC__)
  using TileT=Tile<TileType::Vec,T,64,128,BLayout::RowMajor,64,128,SLayout::NoneBox,512,PadValue::Zero>;
  using Part=Tile<TileType::Vec,T,64,128,BLayout::RowMajor,DYNAMIC,128,SLayout::NoneBox,512,PadValue::Zero>;
  int row=get_subblockid()*64,n=max(0,min(64,length-row));
  TileT value; TASSIGN(value,Address);
  if(n) {
    Part part(n); TASSIGN(part,Address);
    GlobalTensor<T,Shape<1,1,1,DYNAMIC,128>,Stride<1,1,1,DYNAMIC,1>> gm(
      (__gm__ T*)input.data+(first+row)*input.stride[1]+head*input.stride[2],
      {1,1,1,n,128},{1,1,1,input.stride[1],1});
    TLOAD(part,gm); sync(); TFILLPAD_INPLACE(value,part); sync();
    TCVT(out,value,RoundMode::CAST_NONE); sync();
  } else { TEXPANDS(out,0.0f); sync(); }
#endif
}
template<int Address>
AICORE inline void load_value(const Input& input,int first,int head,int length,
                               __gm__ float* packed,Half& out) {
#if defined(__DAV_VEC__)
  if(input.stride[3]!=1) load<64,128>(out,packed+get_subblockid()*8192);
  else if(input.dtype==0) load_value<half,Address>(input,first,head,length,out);
  else load_value<bfloat16_t,Address>(input,first,head,length,out);
#endif
}

// Keep elementwise chains and reductions in UB. Each subcore owns 64 rows.
template<bool Emit=false>
AICORE inline void value_epilogue(const Input& input,int first,int head,int length,__gm__ float* dvb,__gm__ float* v,__gm__ float* beta,
    __gm__ float* dv,__gm__ float* db,__gm__ float* energy,const GradientOutput& output={}) {
#if defined(__DAV_VEC__)
  Half a,b,product; Tile<TileType::Vec,float,64,64> tmp; Row weights; Col col,sum; Short result;
  TASSIGN(a,0); TASSIGN(b,32768); TASSIGN(product,65536); TASSIGN(tmp,98304);
  TASSIGN(sum,114688); TASSIGN(result,114688); TASSIGN(weights,115200);
  int stripe=get_subblockid(),offset=stripe*8192;
  TASSIGN(col,115200+stripe*256);
  sync(); load_async<64,128>(a,dvb+offset); sync();
  load_value<65536>(input,first,head,length,v,b);
  TMUL(product,a,b); sync(); TROWSUM(sum,product,tmp); sync();
  store<1,64>(result,db+stripe*64);
  load<1,128>(weights,beta); TROWEXPANDMUL(a,a,col); sync();
  if constexpr(Emit) scatter_local<0>(output.dv,output.length,output.value_stride);
  else store<64,128>(a,dv+offset);
  TMUL(product,a,b); sync(); TROWSUM(sum,product,tmp); sync();
  store<1,64>(result,energy+stripe*64);
#endif
}
template<bool Emit=false>
AICORE inline void query_epilogue(__gm__ float* state,__gm__ float* local,
    __gm__ float* q,__gm__ float* exp,__gm__ float* out,__gm__ float* energy,const GradientOutput& output={}) {
#if defined(__DAV_VEC__)
  Half a,b,tmp; Row weights; Col col,sum; Short result;
  TASSIGN(a,0); TASSIGN(b,32768); TASSIGN(tmp,65536);
  TASSIGN(weights,98304); TASSIGN(sum,98816); TASSIGN(result,98816);
  int stripe=get_subblockid(),offset=stripe*8192; TASSIGN(col,98304+stripe*256);
  sync(); load_async<64,128>(a,state+offset); load_async<64,128>(b,local+offset); load_async<1,128>(weights,exp); sync();
  TROWEXPANDMUL(a,a,col); sync(); TADD(b,a,b); sync();
  if constexpr(!Emit) store<64,128>(b,out+offset);
  load<64,128>(a,q+offset); TMUL(a,b,a); sync(); TROWSUM(sum,a,tmp); sync();
  store<1,64>(result,energy+stripe*64);
  if constexpr(Emit) accumulate_gradient<0>(b,output);
#endif
}
// One UB lifetime for the three local key-gradient contributions.
AICORE inline void fused_key_epilogue(__gm__ float* dkr,__gm__ float* local,
    __gm__ float* dkb,__gm__ float* k,__gm__ float* ratio,__gm__ float* exp,
    __gm__ float* beta,__gm__ float* dk,__gm__ float* energy,__gm__ float* db) {
#if defined(__DAV_VEC__)
  Half a,b,c; Tile<TileType::Vec,float,64,64> tmp;
  Row r,e,be; Col rc,ec,bc,sum; Short result;
  TASSIGN(a,0); TASSIGN(b,32768); TASSIGN(c,65536); TASSIGN(tmp,98304);
  TASSIGN(r,114688); TASSIGN(e,115200); TASSIGN(be,115712);
  TASSIGN(sum,116224); TASSIGN(result,116224);
  int stripe=get_subblockid(),offset=stripe*8192;
  TASSIGN(rc,114688+stripe*256); TASSIGN(ec,115200+stripe*256);
  TASSIGN(bc,115712+stripe*256);
  sync(); load_async<64,128>(a,dkr+offset); load_async<64,128>(b,k+offset);
  load_async<1,128>(r,ratio); load_async<1,128>(e,exp); load_async<1,128>(be,beta); sync();
  TROWEXPANDMUL(c,b,rc); sync(); TMUL(c,a,c); sync();
  TROWSUM(sum,c,tmp); sync(); store<1,64>(result,energy+stripe*64);
  TROWEXPANDMUL(a,a,rc); sync(); load<64,128>(c,local+offset);
  TADD(a,a,c); sync(); load<64,128>(c,dkb+offset); TMULS(c,c,-1.0f); sync();
  TROWEXPANDMUL(b,b,ec); sync(); TMUL(b,c,b); sync();
  TROWSUM(sum,b,tmp); sync(); store<1,64>(result,db+stripe*64);
  TMUL(be,be,e); sync(); TROWEXPANDMUL(c,c,bc); sync();
  TADD(a,a,c); sync(); store<64,128>(a,dk+offset);
#endif
}
AICORE inline void key_output(__gm__ float* local,__gm__ float* product,const GradientOutput& output) {
#if defined(__DAV_VEC__)
  Half a,b; TASSIGN(a,0); TASSIGN(b,32768); int offset=get_subblockid()*8192;
  sync(); load_async<64,128>(a,local+offset); load_async<64,128>(b,product+offset); sync();
  TADD(a,a,b); sync(); accumulate_gradient<1>(a,output);
#endif
}
#endif
