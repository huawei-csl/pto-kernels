// Copyright (c) 2026 Huawei Technologies Co., Ltd. See repository LICENSE.
#pragma once
#include "device_ops.h"

#ifdef __CCE_AICORE__
// The output transpose places eight heads in [token,head] at UB 4096.
// Native FP16/BF16 inputs use an FP32 tree scan with aligned rows; FP32
// inputs retain the compensated scalar scan for their tighter error budget.
AICORE inline void reverse_gate_scan() {
#if defined(__DAV_VEC__)
  using Part=Tile<TileType::Vec,float,128,8,BLayout::RowMajor,DYNAMIC,8>;
  int current=0;
  for(int offset=1;offset<128;offset*=2) {
    int src=current ? 12288 : 4096,dst=current ? 4096 : 12288,n=128-offset;
    Part a(n),b(n),sum(n),tail(offset),dest(offset);
    TASSIGN(a,src); TASSIGN(b,src+offset*32); TASSIGN(sum,dst);
    TASSIGN(tail,src+n*32); TASSIGN(dest,dst+n*32);
    TADD(sum,a,b); TMOV(dest,tail); sync(); current=1-current;
  }
  Tile<TileType::Vec,float,128,8> result,output;
  TASSIGN(result,12288); TASSIGN(output,4096);
  TMOV(output,result); sync();
#endif
}

// Forward: Z=U-WH, then weight Z before K^T Z. Reverse: weight K*dH
// after the product. This avoids materializing the weighted K matrix twice.
AICORE inline void state_update(__gm__ float* u,__gm__ float* product,
    __gm__ float* ratio,__gm__ float* out,__gm__ float* weighted,bool reverse) {
#if defined(__DAV_VEC__)
  Half a,b; Row weights; Col col;
  TASSIGN(a,0); TASSIGN(b,32768); TASSIGN(weights,65536);
  int stripe=get_subblockid(),offset=stripe*8192;
  TASSIGN(col,65536+stripe*256);
  load<64,128>(a,u+offset); load<64,128>(b,product+offset);
  load<1,128>(weights,ratio);
  if(reverse) { TROWEXPANDMUL(b,b,col); sync(); TADD(a,a,b); }
  else TSUB(a,a,b);
  sync(); store<64,128>(a,out+offset);
  if(weighted) {
    TROWEXPANDMUL(a,a,col); sync(); store<64,128>(a,weighted+offset);
  }
#endif
}

AICORE inline void pipelined_states(const Plan& p,int seq,int head,Ops& o) {
  int begin=p.offset[seq],end=p.offset[seq+1];
  auto product=o.empty(),base=o.empty(),weighted=o.empty();
  if(begin<end) o.pack(p.input[6],p.sequence_base+seq,head,128,chunk(p,begin,head).m(H),true);
  for(int c=begin;c<end;++c) {
    auto x=chunk(p,c,head);
    o.begin();
    o.mm(x.m(W_DW),x.m(H),false,false,product);
    if(c+1<end) o.last(x.m(H),x.v(EXP),base);
    o.end(true);
    state_update(x.m(U_Z).data,product.data,x.v(RATIO).data,x.m(U_Z).data,
                 c+1<end ? weighted.data : nullptr,false);
    o.end(true);
    if(c+1<end) {
      o.begin(); o.mm(x.m(K),weighted,true,false,product); o.end(true);
      o.begin(); o.add(base,product,1,chunk(p,c+1,head).m(H)); o.end(true);
    }
  }
  Buffer dh{scratch(p)},prev{dh.data+Matrix};
  o.pack(p.input[7],p.sequence_base+seq,head,128,dh,true);
  for(int c=end-1;c>=begin;--c) {
    auto x=chunk(p,c,head);
    o.begin();
    o.mm(x.m(K),dh,false,false,product);
    o.last(dh,x.v(EXP),base); o.add(base,x.m(DH_DK),1,base);
    o.copy(dh,x.m(DH_DK));
    o.end(true);
    state_update(x.m(DU).data,product.data,x.v(RATIO).data,x.m(DU).data,nullptr,true);
    o.end(true);
    o.begin(); o.mm(x.m(W_DW),x.m(DU),true,false,product); o.end(true);
    o.begin(); o.add(base,product,-1,prev); o.end(true);
    auto tmp=dh; dh=prev; prev=tmp;
  }
  o.copy(dh,{(__gm__ float*)p.output[5]+((p.sequence_base+seq)*p.heads+head)*Matrix});
}

// Cached preparation: Gram products run while vectors prepare gates; the
// inverse completion runs alongside RHS weighting, then QK alongside transpose.
AICORE inline void pipelined_prepare(const Plan& p,int c,int h,Ops& o) {
  auto x=chunk(p,c,h); int seq=sequence(p,c);
  int first=p.start[seq]+(c-p.offset[seq])*128;
  int n=min(128,p.start[seq]+p.length[seq]-first),kh=h/(p.heads/p.key_heads);
  auto factor=o.empty(),score=o.empty(),qe=o.empty(),kb=o.empty(),vb=o.empty();
  auto hi=o.empty(true),lo=o.empty(true),be=o.empty(true);
  o.begin();
  o.pack(p.input[0],first,kh,n,x.m(Q_DQ));
  o.pack(p.input[1],first,kh,n,x.m(K));
  o.pack(p.input[2],first,h,n,x.m(V));
  o.pack(p.input[5],first,h,n,x.m(DO_DV));
  int64_t offset=int64_t(h)*p.tokens+first;
  cached_row((__gm__ float*)p.gate_cache[0]+offset,hi.data,n);
  cached_row((__gm__ float*)p.gate_cache[1]+offset,lo.data,n);
  cached_row((__gm__ half*)p.gate_cache[2]+offset,x.v(BETA).data,n);
  o.end();

  o.begin();
  o.mm(x.m(K),x.m(K),false,true,x.m(KK));
  o.scale(x.m(Q_DQ),p.scale,x.m(Q_DQ));
  gates(hi.data,x.v(EXP).data,x.v(RATIO).data,x.m(DECAY).data,1,false,lo.data,n);
  o.copy({(__gm__ float*)p.inverse_cache+(int64_t(c)*p.heads+h)*Matrix},factor);
  o.end(); // EXP is produced on vector0 and consumed by both subcores.

  o.begin();
  complete_cached_inverse(factor.data);
  o.mul(x.v(BETA),x.v(EXP),be);
  o.row(x.m(V),x.v(BETA),vb);
  o.row(x.m(Q_DQ),x.v(EXP),qe);
  o.end(); // beta*EXP crosses vector stripes in the next phase.

  o.begin();
  o.mm(x.m(Q_DQ),x.m(K),false,true,score);
  transpose_inverse(factor.data,x.m(AT).data);
  o.row(x.m(K),be,kb);
  o.end(true);

  o.begin();
  o.mm(x.m(AT),kb,true,false,x.m(W_DW));
  o.mm(x.m(AT),vb,true,false,x.m(U_Z));
  o.mul(score,x.m(DECAY),score);
  o.end(true);

  o.begin();
  o.mm(score,x.m(DO_DV),true,false,x.m(DU));
  o.mm(qe,x.m(DO_DV),true,false,x.m(DH_DK));
  o.end();
}

// Each phase pairs independent Cube and vector work. Only the phase boundary
// exchanges results across engines; vector chains own disjoint row stripes.
// Native FP32 Cube arithmetic/rounding is unchanged. FP32 inputs use the
// separately compensated path in kernel.cpp.
AICORE inline void pipelined_gradients(const Plan& p,int c,int h,Ops& o) {
  auto x=chunk(p,c,h);
  auto score=o.empty(),state_q=o.empty(),dq=o.empty(),dkr=o.empty();
  auto local_k=o.empty(),dk=o.empty(),expk=o.empty(),dkb=o.empty(),dvb=o.empty();
  auto raw=o.empty(),dkk=o.empty(),sum_k=o.empty();
  auto be=o.empty(true),dbk=o.empty(true),dbv=o.empty(true);

  // Z=A(beta*V-beta*exp(G)*K*H), dW=-dZ*H^T. Therefore
  // dL=-(A^T*dZ)*Z^T and d(beta*exp(G)*K)=-(A^T*dZ)*H^T.
  // Differentiate the corrected values directly instead of building dA.
  o.begin();
  o.mm(x.m(AT),x.m(DU),false,false,dvb);
  o.mul(x.v(BETA),x.v(EXP),be);
  o.row(x.m(K),x.v(EXP),expk);
  o.dotrows(x.m(H),x.m(DH_DK),x.v(END_STATE));
  o.end();

  o.begin();
  o.mm(x.m(DO_DV),x.m(U_Z),false,true,score);
  // dOut is still needed for dq below; retain dV in a scratch slot for now.
  o.row(dvb,x.v(BETA),sum_k);
  o.dotrows(dvb,x.m(V),dbv);
  o.end(true);

  o.begin();
  o.mm(x.m(DO_DV),x.m(H),false,true,state_q);
  o.mul(score,x.m(DECAY),score);
  o.end(true);

  o.begin();
  o.mm(score,x.m(K),false,false,dq);
  o.row(state_q,x.v(EXP),state_q);
  o.copy(sum_k,x.m(DO_DV));
  o.end(true);

  o.begin();
  o.mm(x.m(U_Z),x.m(DH_DK),false,true,dkr);
  o.add(state_q,dq,1,dq);
  o.dotrows(dq,x.m(Q_DQ),x.v(Q_ENERGY));
  o.end(true);

  o.begin();
  o.mm(score,x.m(Q_DQ),true,false,local_k);
  o.row(dkr,x.v(RATIO),dk);
  o.row(x.m(K),x.v(RATIO),state_q);
  o.dotrows(dkr,state_q,x.v(END_KEYS));
  o.end(true);

  o.begin();
  o.mm(dvb,x.m(H),false,true,dkb);
  o.add(dk,local_k,1,dk);
  o.copy(dq,x.m(Q_DQ));
  o.end(true);

  o.begin();
  o.mm(dvb,x.m(U_Z),false,true,raw);
  o.scale(dkb,-1,dkb);
  o.dotrows(dkb,expk,dbk);
  o.row(dkb,be,local_k);
  o.add(dk,local_k,1,dk);
  o.end(true);

  wy_epilogue(raw.data,x.m(DECAY).data,x.m(KK).data,x.v(BETA).data,
              dbk.data,dkk.data,x.v(DB).data,dbv.data); o.end();
  o.begin();
  transpose_inverse(dkk.data,raw.data);
  o.add(dkk,raw,1,raw);
  o.end(true);
  o.begin();
  o.mm(raw,x.m(K),false,false,sum_k);
  o.copy(dk,x.m(DH_DK));
  o.end(true);
  o.add(x.m(DH_DK),sum_k,1,x.m(DH_DK));
}
#endif
