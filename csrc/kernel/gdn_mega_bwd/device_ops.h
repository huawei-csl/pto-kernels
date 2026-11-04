// Copyright (c) 2026 Huawei Technologies Co., Ltd. See repository LICENSE.
#pragma once
#include "layout.h"
#include "primitives.h"

#ifdef __CCE_AICORE__
// Both vector subblocks participate in the handshake, and both
// participate in arithmetic where rows are independent. Each block owns its
// scratch, with no cross-block waits.
AICORE inline void join() {
  sync();
#if defined(__DAV_CUBE__)
  // Release both vectors only after both have completed the previous work.
  // Signalling first would allow one vector to read the other's unfinished rows.
  wait_flag_dev(1); kernel_utils::SetCrossFlag<PIPE_FIX>(0);
#elif defined(__DAV_VEC__)
  kernel_utils::SetCrossFlag<PIPE_MTE3>(1); wait_flag_dev(0);
#endif
  sync();
}
struct Buffer { __gm__ float* data=nullptr; bool vector=false; };
struct Ops {
  __gm__ float* arena; bool accurate; int used=0;
  template<bool TA,bool TB>
  AICORE void product(Buffer a,Buffer b,Buffer out) {
    if(accurate) {
      auto parts=arena+ScratchMatrices*Matrix;
      matmul<TA,TB,ProductParts>(a.data,b.data,parts,1);
      join(); sum_products(parts,out.data);
    } else matmul<TA,TB>(a.data,b.data,out.data,1);
  }
  AICORE Buffer empty(bool vector=false) {
    Buffer x{arena+used,vector}; used+=vector ? 128 : Matrix; return x;
  }
  AICORE Buffer call(int kind,int op,Buffer a,Buffer b={},Buffer c={},Buffer d={},
                     float alpha=1,Buffer out={}) {
    if(!out.data) out=empty(kind==2 || kind==3 || kind==10);
    if(kind==0) {
      if(op==0) product<false,false>(a,b,out);
      if(op==1) product<true,false>(a,b,out);
      if(op==2) product<false,true>(a,b,out);
      if(op==3) product<true,true>(a,b,out);
    }
    if(kind==1) matrix_op(a.data,b.data,out.data,1,op,alpha);
    if(kind==2) vector_op(a.data,b.data,c.data,d.data,out.data,1,op,alpha);
    if(kind==3) reduce(a.data,out.data);
    if(kind==5) inverse(a.data,out.data,1);
    if(kind==9) weighted_lower(a.data,b.data,c.data,out.data);
    if(kind==10) dot_rows(a.data,b.data,c.data,out.data);
    join(); return out;
  }
  AICORE Buffer mm(Buffer a,Buffer b,bool ta=false,bool tb=false,Buffer out={}) {
    return call(0,ta+2*tb,a,b,{},{},1,out);
  }
  // Accumulate two products in L0C, avoiding two GM intermediates and a
  // vector addition. FP32 inputs retain compensated, separately rounded dots.
  template<bool TA,bool TB,bool TC,bool TD>
  AICORE Buffer mmadd(Buffer a,Buffer b,Buffer c,Buffer d) {
    if(accurate) return add(mm(a,b,TA,TB),mm(c,d,TC,TD));
    auto out=empty();
    matmul<TA,TB,1,true,TC,TD>(a.data,b.data,out.data,1,c.data,d.data);
    join(); return out;
  }
  AICORE Buffer add(Buffer a,Buffer b,float scale=1,Buffer out={}) {
    return call(a.vector ? 2 : 1,0,a,b,{},{},scale,out);
  }
  AICORE Buffer mul(Buffer a,Buffer b) { return call(a.vector ? 2 : 1,1,a,b); }
  AICORE Buffer row(Buffer a,Buffer b) { return call(1,2,a,b); }
  AICORE Buffer last(Buffer a,Buffer b) { return call(1,3,a,b); }
  AICORE Buffer scale(Buffer a,float s,Buffer out={}) { return call(1,4,a,{},{},{},s,out); }
  AICORE Buffer lower(Buffer a) { return call(1,5,a); }
  AICORE Buffer rows(Buffer a,Buffer out={}) { return call(3,0,a,{},{},{},1,out); }
  AICORE Buffer dotrows(Buffer a,Buffer b,Buffer out={},Buffer c={}) {
    return call(10,0,a,b,c,{},1,out);
  }
  AICORE void copy(Buffer a,Buffer out) { scale(a,1,out); }
  AICORE Buffer pack(const Input& x,int first,int head,int length,Buffer out,
                     bool state=false) {
    int64_t st=x.stride[state ? 2 : 1],sh=x.stride[state ? 1 : 2];
    int64_t offset=first*x.stride[state ? 0 : 1]+head*sh;
    auto src=x.data ? (__gm__ uint8_t*)x.data+offset*(x.dtype==2 ? 4 : 2) : nullptr;
    int rows=out.vector ? 1 : 128;
    if(x.dtype==0) pack_input((__gm__ half*)src,out.data,1,rows,length,1,st,0,x.stride[3]);
    else if(x.dtype==1) pack_input((__gm__ bfloat16_t*)src,out.data,1,rows,length,1,st,0,x.stride[3]);
    else pack_input((__gm__ float*)src,out.data,1,rows,length,1,st,0,x.stride[3]);
    join(); return out;
  }
};
struct Chunk {
  __gm__ float* base;
  AICORE Buffer m(int slot) const { return {base+slot*Matrix}; }
  AICORE Buffer v(int slot) const { return {base+MatrixSlots*Matrix+slot*128,true}; }
};
AICORE inline Chunk chunk(const Plan& p,int c,int h) {
  return {(__gm__ float*)p.workspace+(int64_t(c)*p.heads+h)*RecordSize};
}
AICORE inline int sequence(const Plan& p,int c) {
  int s=0; while(c>=p.offset[s+1]) ++s; return s;
}
AICORE inline __gm__ float* scratch(const Plan& p) {
  return (__gm__ float*)p.workspace+int64_t(p.chunks)*p.heads*RecordSize+
         get_block_idx()*ScratchSize;
}
#endif
