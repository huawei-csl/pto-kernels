// Copyright (c) 2026 Huawei Technologies Co., Ltd. See repository LICENSE.
#pragma once
#include "device_ops.h"
#include "blocked_inverse.h"
#include "epilogues.h"
#include "resident_cube.h"

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
  sync(); load_async<64,128>(a,u+offset); load_async<64,128>(b,product+offset);
  load_async<1,128>(weights,ratio); sync();
  if(reverse) { TROWEXPANDMUL(b,b,col); sync(); TADD(a,a,b); }
  else TSUB(a,a,b);
  sync(); store<64,128>(a,out+offset);
  if(weighted) {
    TROWEXPANDMUL(a,a,col); sync(); store<64,128>(a,weighted+offset);
  }
#endif
}

// The scaled incoming state stays in UB across the two products of a scan step.
AICORE inline void state_base(__gm__ float* state,__gm__ float* exp,
                              __gm__ float* local=nullptr,__gm__ float* saved=nullptr) {
#if defined(__DAV_VEC__)
  Half input,extra,base; Row weights;
  TASSIGN(input,0); TASSIGN(extra,32768); TASSIGN(base,128*1024); TASSIGN(weights,65536);
  int offset=get_subblockid()*8192;
  sync(); load_async<64,128>(input,state+offset); load_async<1,128>(weights,exp); sync();
  float last=weights.GetValue(127); TMULS(base,input,last); sync();
  if(local) { load<64,128>(extra,local+offset); TADD(base,base,extra); sync(); }
  if(saved) store<64,128>(input,saved+offset);
#endif
}
AICORE inline void finish_state(__gm__ float* product,__gm__ float* out,float sign) {
#if defined(__DAV_VEC__)
  Half input,base; TASSIGN(input,0); TASSIGN(base,128*1024);
  int offset=get_subblockid()*8192;
  load<64,128>(input,product+offset); TMULS(input,input,sign); sync();
  TADD(base,base,input); sync(); store<64,128>(base,out+offset);
#endif
}

AICORE inline void pipelined_states(const Plan& p,int seq,int head,Ops& o) {
  int begin=p.offset[seq],end=p.offset[seq+1];
  auto product=o.empty(),weighted=o.empty(); ResidentCube cube;
  if(begin<end) o.pack(p.input[6],p.sequence_base+seq,head,128,chunk(p,begin,head).m(H),true);
  for(int c=begin;c<end;++c) {
    auto x=chunk(p,c,head);
    o.begin();
    cube.load<0>(x.m(W_DW).data); cube.load<1>(x.m(H).data);
    cube.product<0,1>(product.data);
    if(c+1<end) cube.load<2>(x.m(K).data);
    if(c+1<end) state_base(x.m(H).data,x.v(EXP).data);
    o.end(true);
    state_update(x.m(U_Z).data,product.data,x.v(RATIO).data,x.m(U_Z).data,
                 c+1<end ? weighted.data : nullptr,false);
    o.end(true);
    if(c+1<end) {
      o.begin(); cube.load<1>(weighted.data); cube.product<2,1,true>(product.data); o.end(true);
      o.begin(); finish_state(product.data,chunk(p,c+1,head).m(H).data,1); o.end(true);
    }
  }
  Buffer dh{scratch(p)},prev{dh.data+Matrix};
  o.pack(p.input[7],p.sequence_base+seq,head,128,dh,true);
  for(int c=end-1;c>=begin;--c) {
    auto x=chunk(p,c,head);
    o.begin();
    cube.load<0>(x.m(K).data); cube.load<1>(dh.data);
    cube.product<0,1>(product.data); cube.load<2>(x.m(W_DW).data);
    state_base(dh.data,x.v(EXP).data,x.m(DH_DK).data,x.m(DH_DK).data);
    o.end(true);
    state_update(x.m(DU).data,product.data,x.v(RATIO).data,x.m(DU).data,nullptr,true);
    o.end(true);
    o.begin(); cube.load<1>(x.m(DU).data); cube.product<2,1,true>(product.data); o.end(true);
    o.begin(); finish_state(product.data,prev.data,-1); o.end(true);
    auto tmp=dh; dh=prev; prev=tmp;
  }
  o.copy(dh,{(__gm__ float*)p.output[5]+((p.sequence_base+seq)*p.heads+head)*Matrix});
}

// Load a complete aligned head panel, then select this head after transposing.
// This replaces 128 individual GM scalar reads for token-major gates/beta.
template<class T>
AICORE inline void pack_gate(const Input& input,int first,int head,int heads,
                             int valid,__gm__ float* out) {
#if defined(__DAV_VEC__)
  if(get_subblockid()!=0) return;
  constexpr int C=32/sizeof(T);
  using Panel=Tile<TileType::Vec,T,128,C,BLayout::RowMajor,128,C,SLayout::NoneBox,512,PadValue::Zero>;
  using Part=Tile<TileType::Vec,T,128,C,BLayout::RowMajor,DYNAMIC,DYNAMIC,SLayout::NoneBox,512,PadValue::Zero>;
  Panel panel; Part part(valid,min(C,heads-head/C*C));
  Tile<TileType::Vec,T,C,128> transposed,tmp;
  Tile<TileType::Vec,T,1,128> selected; Row value;
  TASSIGN(panel,0); TASSIGN(part,0); TASSIGN(transposed,16384);
  TASSIGN(tmp,32768); TASSIGN(selected,16384+(head%C)*128*sizeof(T)); TASSIGN(value,49152);
  GlobalTensor<T,Shape<1,1,1,DYNAMIC,DYNAMIC>,Stride<1,1,1,DYNAMIC,1>> gm(
    (__gm__ T*)input.data+first*input.stride[1]+head/C*C,
    {1,1,1,valid,min(C,heads-head/C*C)},{1,1,1,input.stride[1],1});
  TLOAD(part,gm); sync(); TFILLPAD_INPLACE(panel,part); sync();
  TTRANS(transposed,panel,tmp); sync();
  if constexpr(std::is_same_v<T,float>) store<1,128>(selected,out);
  else { TCVT(value,selected,RoundMode::CAST_NONE); sync(); store<1,128>(value,out); }
#endif
}
AICORE inline void pack_gate(const Plan& p,int input,int first,int head,int valid,Buffer out,Ops& o) {
  auto x=p.input[input];
  if(x.stride[2]!=1) { o.pack(x,first,head,valid,out); return; }
  if(x.dtype==0) pack_gate<half>(x,first,head,p.heads,valid,out.data);
  if(x.dtype==1) pack_gate<bfloat16_t>(x,first,head,p.heads,valid,out.data);
  if(x.dtype==2) pack_gate<float>(x,first,head,p.heads,valid,out.data);
}

// Cached preparation: Gram products run while vectors prepare gates; the
// inverse completion runs alongside RHS weighting. Complete A^T directly in its
// destination, avoiding the temporary factor copy and its extra GM traffic.
AICORE inline void pipelined_prepare(const Plan& p,int c,int h,int first_head,Ops& o) {
  auto x=chunk(p,c,h),shared=chunk(p,c,first_head); int seq=sequence(p,c);
  bool leader=h==first_head;
  int first=p.start[seq]+(c-p.offset[seq])*128;
  int n=min(128,p.start[seq]+p.length[seq]-first),kh=h/(p.heads/p.key_heads);
  auto factor=o.empty(),score=o.empty(),qe=o.empty(),kb=o.empty(),vb=o.empty();
  auto hi=o.empty(true),lo=o.empty(true),be=o.empty(true);
  o.begin();
  if(first_head==h) {
    o.pack(p.input[0],first,kh,n,x.m(Q_DQ));
    o.pack(p.input[1],first,kh,n,x.m(K));
  } else {
    o.copy(shared.m(Q_DQ),x.m(Q_DQ)); // K/KK alias the leader throughout backward.
  }
  o.pack(p.input[2],first,h,n,x.m(V));
  o.pack(p.input[5],first,h,n,x.m(DO_DV));
  int64_t offset=int64_t(h)*p.tokens+first;
  if(p.gate_cache[0]) {
    cached_row((__gm__ float*)p.gate_cache[0]+offset,hi.data,n);
    cached_row((__gm__ float*)p.gate_cache[1]+offset,lo.data,n);
    cached_row((__gm__ half*)p.gate_cache[2]+offset,x.v(BETA).data,n);
  } else {
    pack_gate(p,3,first,h,n,hi,o);
    pack_gate(p,4,first,h,n,x.v(BETA),o);
  }
  o.end();

  o.begin();
  if(leader) {
    o.mm(x.m(K),x.m(K),false,true,x.m(KK));
    o.scale(x.m(Q_DQ),p.scale,x.m(Q_DQ));
  }
  gates(hi.data,x.v(EXP).data,x.v(RATIO).data,x.m(DECAY).data,1,false,
        p.gate_cache[0] ? lo.data : nullptr,n);
  if(p.inverse_cache)
    transpose_inverse((__gm__ float*)p.inverse_cache+(int64_t(c)*p.heads+h)*Matrix,x.m(AT).data);
  o.end(); // EXP is produced on vector0 and consumed by both subcores.

  o.begin();
  if(p.inverse_cache)
    complete_cached_inverse<true>((__gm__ float*)p.inverse_cache+(int64_t(c)*p.heads+h)*Matrix,x.m(AT).data);
  o.mul(x.v(BETA),x.v(EXP),be);
  o.row(x.m(V),x.v(BETA),vb);
  o.row(x.m(Q_DQ),x.v(EXP),qe);
  o.end(); // beta*EXP crosses vector stripes in the next phase.

  if(!p.inverse_cache) {
    o.call(9,0,x.m(KK),x.m(DECAY),x.v(BETA),{},1,factor);
    blocked_inverse(factor.data,score.data,x.m(AT).data);
  }
  o.begin();
  if(leader) o.mm(x.m(Q_DQ),x.m(K),false,true,shared.m(H));
  o.row(x.m(K),be,kb);
  o.end(true);

  o.begin();
  o.mm(x.m(AT),kb,true,false,x.m(W_DW));
  o.mm(x.m(AT),vb,true,false,x.m(U_Z));
  o.mul(shared.m(H),x.m(DECAY),score);
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
template<bool Emit=false>
AICORE inline void pipelined_gradients(const Plan& p,int c,int h,Ops& o) {
  auto x=chunk(p,c,h); ResidentCube cube; GradientOutput output;
  if constexpr(Emit) {
    int seq=sequence(p,c),first=p.start[seq]+(c-p.offset[seq])*128,group=p.heads/p.key_heads;
    output={(__gm__ float*)p.output[0]+(int64_t(first)*p.key_heads+h/group)*128,
            (__gm__ float*)p.output[1]+(int64_t(first)*p.key_heads+h/group)*128,
            (__gm__ float*)p.output[2]+(int64_t(first)*p.heads+h)*128,
            min(128,p.start[seq]+p.length[seq]-first),p.key_heads*128,p.heads*128,
            p.scale,h%group==0,h%group==group-1};
  }
  auto score=o.empty(),state_q=o.empty(),dq=o.empty(),dkr=o.empty();
  auto local_k=o.empty(),dk=o.empty(),dkb=o.empty(),dvb=o.empty();
  auto raw=o.empty(),dkk=o.empty(),sum_k=o.empty();
  auto dbk=o.empty(true),dbv=o.empty(true),dv_energy=o.empty(true);

  // Z=A(beta*V-beta*exp(G)*K*H), dW=-dZ*H^T. Therefore
  // dL=-(A^T*dZ)*Z^T and d(beta*exp(G)*K)=-(A^T*dZ)*H^T.
  // Differentiate the corrected values directly instead of building dA.
  o.begin();
  cube.load<0>(x.m(AT).data); cube.load<1>(x.m(DU).data);
  cube.product<0,1>(dvb.data);
  cube.load<3>(x.m(U_Z).data); cube.load<6>(x.m(DO_DV).data);
  o.dotrows(x.m(H),x.m(DH_DK),x.v(END_STATE));
  o.end();

  o.begin();
  cube.product<6,3,false,true>(score.data); cube.load<0>(dvb.data); cube.load<2>(x.m(H).data);
  // dOut remains resident in L1. Fused tasks can write dV directly.
  value_epilogue<Emit>(dvb.data,x.m(V).data,x.v(BETA).data,sum_k.data,dbv.data,dv_energy.data,output);
  o.end(true);

  o.begin();
  cube.product<6,2,false,true>(state_q.data);
  // Shared-key heads retain Q/K in their dedicated L1 slots across adjoints.
  if(!Emit || output.first) {
    cube.load<4>(x.m(K).data); cube.load<5>(x.m(Q_DQ).data);
  }
  o.mul(score,x.m(DECAY),score);
  o.end(true);

  o.begin();
  cube.load<1>(score.data); cube.product<1,4>(dq.data); cube.load<6>(x.m(DH_DK).data);
  if constexpr(!Emit) o.copy(sum_k,x.m(DO_DV));
  o.end(true);

  o.begin();
  cube.product<3,6,false,true>(dkr.data);
  query_epilogue<Emit>(state_q.data,dq.data,x.m(Q_DQ).data,x.v(EXP).data,dq.data,x.v(Q_ENERGY).data,output);
  o.end(true);

  o.begin();
  cube.product<1,5,true,false>(local_k.data);
  key_epilogue(dkr.data,x.m(K).data,x.v(RATIO).data,dk.data,x.v(END_KEYS).data);
  o.end(true);

  o.begin();
  cube.product<0,2,false,true>(dkb.data);
  o.add(dk,local_k,1,dk);
  if constexpr(!Emit) o.copy(dq,x.m(Q_DQ));
  // Ratio is dead; copy each subcore's own gradient-energy stripe.
#if defined(__DAV_VEC__)
  Short energy; TASSIGN(energy,0); int stripe=get_subblockid();
  load<1,64>(energy,dv_energy.data+stripe*64); store<1,64>(energy,x.v(RATIO).data+stripe*64);
#endif
  o.end(true);

  o.begin();
  cube.product<0,3,false,true>(raw.data);
  inverse_input_epilogue(dkb.data,x.m(K).data,x.v(EXP).data,x.v(BETA).data,dk.data,dbk.data);
  o.end(true);

  wy_epilogue(raw.data,x.m(DECAY).data,x.m(KK).data,x.v(BETA).data,
              dbk.data,dkk.data,x.v(DB).data,dbv.data); o.end();
  o.begin();
  symmetric_matrix(dkk.data,raw.data);
  o.end(true);
  o.begin();
  cube.load<3>(raw.data); cube.product<3,4>(sum_k.data);
  o.end(true);
  if constexpr(Emit) {
    o.begin();
    key_output(dk.data,sum_k.data,output);
    auto dg=o.add(x.v(Q_ENERGY),x.v(RATIO),-1);
    o.call(2,4,dg,x.v(END_STATE),x.v(END_KEYS),x.v(EXP),1,x.v(Q_ENERGY));
    o.end();
  } else o.add(dk,sum_k,1,x.m(DH_DK));
}
#endif
