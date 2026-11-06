// Copyright (c) 2026 Huawei Technologies Co., Ltd. See repository LICENSE.
#pragma once
#include "device_ops.h"
#include "blocked_inverse.h"
#include "epilogues.h"
#include "resident_cube.h"

#ifdef __CCE_AICORE__
// Stages own fixed scratch addresses and explicit Cube/Vec phase boundaries.
AICORE inline void native_pack(const Input& x,int first,int head,int length,Buffer out,
                               bool state=false,float scale=1) {
  int64_t st=x.stride[state ? 2 : 1],sh=x.stride[state ? 1 : 2];
  int64_t offset=first*x.stride[state ? 0 : 1]+head*sh;
  auto src=x.data ? (__gm__ uint8_t*)x.data+offset*(x.dtype==2 ? 4 : 2) : nullptr;
  int rows=out.vector ? 1 : 128;
  if(x.dtype==0) pack_input((__gm__ half*)src,out.data,1,rows,length,1,st,0,x.stride[3],scale);
  else if(x.dtype==1) pack_input((__gm__ bfloat16_t*)src,out.data,1,rows,length,1,st,0,x.stride[3],scale);
  else pack_input((__gm__ float*)src,out.data,1,rows,length,1,st,0,x.stride[3],scale);
}
AICORE inline void native_copy(Buffer a,Buffer b) { matrix_op(a.data,nullptr,b.data,1,4,1); }

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
template<int Address=128*1024>
AICORE inline void state_base(__gm__ float* state,__gm__ float* exp,
                              __gm__ float* local=nullptr,__gm__ float* saved=nullptr) {
#if defined(__DAV_VEC__)
  Half input,extra,base; Row weights;
  TASSIGN(input,0); TASSIGN(extra,32768); TASSIGN(base,Address); TASSIGN(weights,65536);
  int offset=get_subblockid()*8192;
  sync(); load_async<64,128>(input,state+offset); load_async<1,128>(weights,exp); sync();
  float last=weights.GetValue(127); TMULS(base,input,last); sync();
  if(local) { load<64,128>(extra,local+offset); TADD(base,base,extra); sync(); }
  if(saved) store<64,128>(input,saved+offset);
#endif
}
template<int Address=128*1024>
AICORE inline void finish_state(__gm__ float* product,__gm__ float* out,float sign) {
#if defined(__DAV_VEC__)
  Half input,base; TASSIGN(input,0); TASSIGN(base,Address);
  int offset=get_subblockid()*8192;
  load<64,128>(input,product+offset); TMULS(input,input,sign); sync();
  TADD(base,base,input); sync(); store<64,128>(base,out+offset);
#endif
}

// Replay and reverse propagation are independent scans. Interleave their
// products so each vector update overlaps the opposite direction's Cube work.
AICORE inline void pipelined_states(const Plan& p,int seq,int head) {
  int begin=p.offset[seq],end=p.offset[seq+1],epoch=0;
  auto dh=scratch(p),forward=dh+2*Matrix,weighted=forward+Matrix,reverse=weighted+Matrix;
  if(begin<end) native_pack(p.input[6],p.sequence_base+seq,head,128,chunk(p,begin,head).m(H),true);
  native_pack(p.input[7],p.sequence_base+seq,head,128,{dh},true); join();
  for(int step=0;step<end-begin;++step) {
    int c=begin+step,r=end-1-step; auto f=chunk(p,c,head),b=chunk(p,r,head);
    bool next=c+1<end;
    cube_load<0>(f.m(W_DW).data); cube_load<1>(f.m(H).data);
    cube_product<0,1>(forward); if(next) cube_load<2>(f.m(K).data);
    if(step) finish_state(reverse,dh,-1);
    if(next) state_base<96*1024>(f.m(H).data,f.v(EXP).data);
    state_base(dh,b.v(EXP).data,b.m(DH_DK).data,b.m(DH_DK).data);
    phase_end(epoch,true);

    cube_load<3>(b.m(K).data); cube_load<4>(dh); cube_load<5>(b.m(W_DW).data);
    cube_product<3,4>(reverse);
    state_update(f.m(U_Z).data,forward,f.v(RATIO).data,f.m(U_Z).data,next ? weighted : nullptr,false);
    phase_end(epoch,true);

    if(next) { cube_load<0>(weighted); cube_product<2,0,true>(forward); }
    state_update(b.m(DU).data,reverse,b.v(RATIO).data,b.m(DU).data,nullptr,true);
    phase_end(epoch,true);

    cube_load<4>(b.m(DU).data); cube_product<5,4,true>(reverse);
    if(next) finish_state<96*1024>(forward,chunk(p,c+1,head).m(H).data,1);
    phase_end(epoch,true);
  }
  auto out=(__gm__ float*)p.output[5]+((p.sequence_base+seq)*p.heads+head)*Matrix;
  if(begin<end) finish_state(reverse,out,-1); else native_copy({dh},{out});
  join();
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
AICORE inline void pack_gate(const Plan& p,int input,int first,int head,int valid,Buffer out) {
  auto x=p.input[input];
  if(x.stride[2]!=1) { native_pack(x,first,head,valid,out); return; }
  if(x.dtype==0) pack_gate<half>(x,first,head,p.heads,valid,out.data);
  if(x.dtype==1) pack_gate<bfloat16_t>(x,first,head,p.heads,valid,out.data);
  if(x.dtype==2) pack_gate<float>(x,first,head,p.heads,valid,out.data);
}

AICORE inline void prepare_score(__gm__ float* qk,__gm__ float* decay,
                                  __gm__ float* out,float scale) {
#if defined(__DAV_VEC__)
  Half a,b; TASSIGN(a,0); TASSIGN(b,32768); int offset=get_subblockid()*8192;
  sync(); load_async<64,128>(a,qk+offset); load_async<64,128>(b,decay+offset); sync();
  TMUL(a,a,b); sync(); if(scale!=1) { TMULS(a,a,scale); sync(); }
  store<64,128>(a,out+offset);
#endif
}

// Weight all three right-hand sides in one UB pass; beta*exp(G) stays local.
AICORE inline void prepare_rhs(const Input& input,int first,int head,int length,const Chunk& x,__gm__ float* kb,__gm__ float* vb,__gm__ float* qe) {
#if defined(__DAV_VEC__)
  Half k,v,q; Row b,e,be; Col bc,ec,bec;
  TASSIGN(k,0); TASSIGN(v,32768); TASSIGN(q,65536);
  TASSIGN(b,98304); TASSIGN(e,98816); TASSIGN(be,99328);
  int stripe=get_subblockid(),offset=stripe*8192;
  TASSIGN(bc,98304+stripe*256); TASSIGN(ec,98816+stripe*256);
  TASSIGN(bec,99328+stripe*256);
  sync(); load_async<64,128>(k,x.m(K).data+offset); sync();
  load_value<65536>(input,first,head,length,x.m(V).data,v);
  if(qe) load_async<64,128>(q,x.m(Q_DQ).data+offset);
  load_async<1,128>(b,x.v(BETA).data);
  load_async<1,128>(e,x.v(EXP).data); sync();
  TMUL(be,b,e); sync(); TROWEXPANDMUL(k,k,bec); TROWEXPANDMUL(v,v,bc);
  if(qe) TROWEXPANDMUL(q,q,ec); sync();
  store<64,128>(k,kb+offset); store<64,128>(v,vb+offset);
  if(qe) store<64,128>(q,qe+offset);
#endif
}

// Cached preparation: Gram products run while vectors prepare gates; the
// inverse completion runs alongside RHS weighting. Complete A^T directly in its
// destination, avoiding the temporary factor copy and its extra GM traffic.
AICORE inline void pipelined_prepare(const Plan& p,int c,int h,int first_head) {
  auto x=chunk(p,c,h),shared=chunk(p,c,first_head); int seq=sequence(p,c);
  bool leader=h==first_head;
  int first=p.start[seq]+(c-p.offset[seq])*128;
  int n=min(128,p.start[seq]+p.length[seq]-first),kh=h/(p.heads/p.key_heads);
  int epoch=0; auto arena=scratch(p)+2*Matrix;
  Buffer factor{arena},score{arena+Matrix},qe{arena+2*Matrix},kb{arena+3*Matrix},vb{arena+4*Matrix};
  Buffer hi{arena+5*Matrix,true},lo{hi.data+128,true};
  bool direct=n==128 && p.input[0].dtype==p.input[1].dtype &&
    p.input[0].stride[3]==1 && p.input[1].stride[3]==1;

  if(leader && direct) {
    if(p.input[0].dtype==0) input_grams<half>(p.input[0],p.input[1],first,kh,x.m(KK).data,shared.m(H).data);
    else input_grams<bfloat16_t>(p.input[0],p.input[1],first,kh,x.m(KK).data,shared.m(H).data);
  }
  if(leader) {
    native_pack(p.input[0],first,kh,n,x.m(Q_DQ),false,p.scale);
    native_pack(p.input[1],first,kh,n,x.m(K));
  } else {
    // Fused output tasks also share Q; smaller grids retain private dQ slots.
    if(shared.m(Q_DQ).data!=x.m(Q_DQ).data) native_copy(shared.m(Q_DQ),x.m(Q_DQ));
  }
  if(p.input[2].stride[3]!=1) native_pack(p.input[2],first,h,n,x.m(V));
  native_pack(p.input[5],first,h,n,x.m(DO_DV));
  int64_t offset=int64_t(h)*p.tokens+first;
  if(p.gate_cache[0]) {
    cached_row((__gm__ float*)p.gate_cache[0]+offset,hi.data,n);
    cached_row((__gm__ float*)p.gate_cache[1]+offset,lo.data,n);
    cached_row((__gm__ half*)p.gate_cache[2]+offset,x.v(BETA).data,n);
  } else {
    pack_gate(p,3,first,h,n,hi);
    pack_gate(p,4,first,h,n,x.v(BETA));
  }
  phase_end(epoch);

  if(leader && !direct) {
    cube_load<0>(x.m(K).data); cube_load<1>(x.m(Q_DQ).data);
    cube_product<0,0,false,true>(x.m(KK).data);
    cube_product<1,0,false,true>(shared.m(H).data);
  }
  gates(hi.data,x.v(EXP).data,x.v(RATIO).data,x.m(DECAY).data,1,false,
        p.gate_cache[0] ? lo.data : nullptr,n);
  if(p.inverse_cache) {
    auto saved=inverse_data(p,c,h);
    if(p.inverse_dtype==2) transpose_inverse((__gm__ float*)saved,x.m(AT).data);
    else if(p.inverse_dtype==0) transpose_inverse((__gm__ half*)saved,x.m(AT).data);
    else transpose_inverse((__gm__ bfloat16_t*)saved,x.m(AT).data);
  }
  phase_end(epoch); // EXP is produced on vector0 and consumed by both subcores.

  if(p.inverse_cache) {
    if(p.inverse_dtype==2)
      complete_cached_inverse<true>((__gm__ float*)inverse_data(p,c,h),x.m(AT).data);
    else // Reuse the expanded transpose; no separate FP32 factor copy.
      complete_cached_inverse<true,true>(x.m(AT).data,x.m(AT).data);
  }
  prepare_rhs(p.input[2],first,h,n,x,kb.data,vb.data,qe.data);
  phase_end(epoch,true);

  if(!p.inverse_cache) {
    weighted_lower(x.m(KK).data,x.m(DECAY).data,x.v(BETA).data,factor.data); join();
    blocked_inverse(factor.data,score.data,x.m(AT).data);
  }

  cube_load<0>(x.m(AT).data); cube_load<1>(kb.data); cube_load<2>(vb.data);
  cube_product<0,1,true>(x.m(W_DW).data); cube_product<0,2,true>(x.m(U_Z).data);
  prepare_score(shared.m(H).data,x.m(DECAY).data,score.data,direct ? p.scale : 1);
  phase_end(epoch,true);

  cube_load<0>(score.data); cube_load<1>(x.m(DO_DV).data); cube_load<2>(qe.data);
  cube_product<0,1,true>(x.m(DU).data); cube_product<2,1,true>(x.m(DH_DK).data);
  phase_end(epoch);
}

// Combine the gate-energy terms in UB; only the finished vector reaches GM.
AICORE inline void finish_gate(const Chunk& x) {
#if defined(__DAV_VEC__)
  if(get_subblockid()!=0) return;
  Row q,v,state,keys,e,tmp; Scalar sum;
  TASSIGN(q,0); TASSIGN(v,512); TASSIGN(state,1024); TASSIGN(keys,1536);
  TASSIGN(e,2048); TASSIGN(tmp,2560); TASSIGN(sum,3072);
  sync(); load_async<1,128>(q,x.v(Q_ENERGY).data); load_async<1,128>(v,x.v(RATIO).data);
  load_async<1,128>(state,x.v(END_STATE).data); load_async<1,128>(keys,x.v(END_KEYS).data);
  load_async<1,128>(e,x.v(EXP).data); sync();
  TMULS(v,v,-1.0f); sync(); TADD(q,q,v); sync();
  TMULS(state,state,e.GetValue(127)); sync(); TADD(state,state,keys); sync();
  TROWSUM(sum,state,tmp); sync(); q.SetValue(127,sum.GetValue(0)+q.GetValue(127)); sync();
  store<1,128>(q,x.v(Q_ENERGY).data);
#endif
}

// Each phase pairs independent Cube and vector work. Only the phase boundary
// exchanges results across engines; vector chains own disjoint row stripes.
// FP32 operands and accumulation are retained for the derivatives. FP32 inputs
// use the separately compensated path in kernel.cpp.
template<bool Emit=false>
AICORE inline void pipelined_gradients(const Plan& p,int c,int h) {
  auto x=chunk(p,c,h); GradientOutput output;
  int seq=sequence(p,c),first=p.start[seq]+(c-p.offset[seq])*128;
  int length=min(128,p.start[seq]+p.length[seq]-first);
  if constexpr(Emit) {
    int group=p.heads/p.key_heads;
    output={(__gm__ float*)p.output[0]+(int64_t(first)*p.key_heads+h/group)*128,
            (__gm__ float*)p.output[1]+(int64_t(first)*p.key_heads+h/group)*128,
            (__gm__ float*)p.output[2]+(int64_t(first)*p.heads+h)*128,
            length,p.key_heads*128,p.heads*128,
            p.scale,h%group==0,h%group==group-1};
  }
  int epoch=0; auto arena=scratch(p)+2*Matrix;
  Buffer score{arena},state_q{arena+Matrix},dq{arena+2*Matrix},dkr{arena+3*Matrix};
  Buffer local_k{arena+4*Matrix},dk{arena+5*Matrix},dkb{arena+6*Matrix},dvb{arena+7*Matrix};
  Buffer raw{arena+8*Matrix},dkk{arena+9*Matrix},sum_k{arena+10*Matrix};
  Buffer dbk{arena+11*Matrix,true},dbv{dbk.data+128,true},dv_energy{dbk.data+256,true};

  // Z=A(beta*V-beta*exp(G)*K*H), dW=-dZ*H^T. Therefore
  // dL=-(A^T*dZ)*Z^T and d(beta*exp(G)*K)=-(A^T*dZ)*H^T.
  // Differentiate the corrected values directly instead of building dA.

  cube_load<0>(x.m(AT).data); cube_load<1>(x.m(DU).data);
  cube_product<0,1>(dvb.data);
  cube_load<3>(x.m(U_Z).data); cube_load<6>(x.m(DO_DV).data);
  dot_rows(x.m(H).data,x.m(DH_DK).data,nullptr,x.v(END_STATE).data);
  phase_end(epoch);

  cube_product<6,3,false,true>(score.data); cube_load<0>(dvb.data); cube_load<2>(x.m(H).data);
  cube_product<0,3,false,true>(raw.data);
  // dOut remains resident in L1. Fused tasks can write dV directly.
  value_epilogue<Emit>(p.input[2],first,h,length,dvb.data,x.m(V).data,x.v(BETA).data,sum_k.data,dbv.data,dv_energy.data,output);
  phase_end(epoch,true);

  cube_product<6,2,false,true>(state_q.data);
  // Shared-key heads retain Q/K in their dedicated L1 slots across adjoints.
  if(!Emit || output.first) {
    cube_load<4>(x.m(K).data); cube_load<5>(x.m(Q_DQ).data);
  }
  prepare_score(score.data,x.m(DECAY).data,score.data,1);
  phase_end(epoch,true);

  cube_load<1>(score.data); cube_product<1,4>(dq.data); cube_load<6>(x.m(DH_DK).data);
  if constexpr(!Emit) native_copy(sum_k,x.m(DO_DV));
  phase_end(epoch,true);

  cube_product<0,2,false,true>(dkb.data);
  query_epilogue<Emit>(state_q.data,dq.data,x.m(Q_DQ).data,x.v(EXP).data,dq.data,x.v(Q_ENERGY).data,output);
  if constexpr(!Emit) native_copy(dq,x.m(Q_DQ));
  phase_end(epoch,true);

  // Defer independent key products into the previously Cube-idle WY phases.
  // The beta contributions are combined later, in the original rounding order.
  cube_product<3,6,false,true>(dkr.data);
  wy_epilogue(raw.data,x.m(DECAY).data,x.m(KK).data,x.v(BETA).data,
              nullptr,dkk.data,x.v(DB).data);
  phase_end(epoch);

  cube_product<1,5,true,false>(local_k.data);
  cube_load<3>(dkk.data);
  phase_end(epoch,true);

  cube_product<3,4,false,false,true>(sum_k.data);
  fused_key_epilogue(dkr.data,local_k.data,dkb.data,x.m(K).data,x.v(RATIO).data,
                    x.v(EXP).data,x.v(BETA).data,dk.data,x.v(END_KEYS).data,dbk.data);
#if defined(__DAV_VEC__)
  Short a,b,term; TASSIGN(a,0); TASSIGN(b,256); TASSIGN(term,512); int stripe=get_subblockid();
  load<1,64>(a,dbk.data+stripe*64); load<1,64>(b,dbv.data+stripe*64);
  load<1,64>(term,x.v(DB).data+stripe*64);
  TADD(a,a,b); sync(); TADD(a,a,term); sync(); store<1,64>(a,x.v(DB).data+stripe*64);
  load<1,64>(a,dv_energy.data+stripe*64); store<1,64>(a,x.v(RATIO).data+stripe*64);
#endif
  phase_end(epoch);
  if constexpr(Emit) {

    key_output(dk.data,sum_k.data,output);
    finish_gate(x);
    phase_end(epoch);
  } else { matrix_op(dk.data,sum_k.data,x.m(DH_DK).data,1,0,1); join(); }
}
#endif
