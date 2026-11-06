// Copyright (c) 2026 Huawei Technologies Co., Ltd. See repository LICENSE.
// Three sequence-wide stages; small grids and FP32 inputs retain output assembly.
// Chunk traversal and all numerical work execute in PTO-ISA on NPU.
#include "device_ops.h"
#include "pipeline.h"

#ifdef __CCE_AICORE__
AICORE inline void prepare(const Plan& p,int c,int head,Ops& o) {
  auto x=chunk(p,c,head); int seq=sequence(p,c);
  int first=p.start[seq]+(c-p.offset[seq])*128;
  int length=min(128,p.start[seq]+p.length[seq]-first),kh=head/(p.heads/p.key_heads);
  o.pack(p.input[0],first,kh,length,x.m(Q_DQ));
  o.scale(x.m(Q_DQ),p.scale,x.m(Q_DQ));
  o.pack(p.input[1],first,kh,length,x.m(K));
  o.pack(p.input[2],first,head,length,x.m(V));
  o.pack(p.input[5],first,head,length,x.m(DO_DV));
  if(p.gate_cache[0]) {
    auto hi=o.empty(true),lo=o.empty(true); int64_t offset=int64_t(head)*p.tokens+first;
    cached_row((__gm__ float*)p.gate_cache[0]+offset,hi.data,length);
    cached_row((__gm__ float*)p.gate_cache[1]+offset,lo.data,length);
    cached_row((__gm__ half*)p.gate_cache[2]+offset,x.v(BETA).data,length); join();
    gates(hi.data,x.v(EXP).data,x.v(RATIO).data,x.m(DECAY).data,1,o.accurate,lo.data,length); join();
  } else {
    auto g=o.pack(p.input[3],first,head,length,o.empty(true));
    o.pack(p.input[4],first,head,length,x.v(BETA));
    gates(g.data,x.v(EXP).data,x.v(RATIO).data,x.m(DECAY).data,1,o.accurate); join();
  }
  o.mm(x.m(K),x.m(K),false,true,x.m(KK));
  if(p.inverse_cache) {
    auto factor=o.empty();
    Input saved{(uint64_t)inverse_data(p,c,head),{Matrix,128,0,1},p.inverse_dtype};
    o.pack(saved,0,0,128,factor);
    complete_cached_inverse(factor.data); join();
    transpose_inverse(factor.data,x.m(AT).data); join();
  } else {
    auto l=o.call(9,0,x.m(KK),x.m(DECAY),x.v(BETA));
    o.call(5,0,l,{},{},{},1,x.m(AT));
  }
  auto kb=o.row(x.m(K),o.mul(x.v(BETA),x.v(EXP))),vb=o.row(x.m(V),x.v(BETA));
  o.mm(x.m(AT),kb,true,false,x.m(W_DW));
  o.mm(x.m(AT),vb,true,false,x.m(U_Z));
}

AICORE inline void replay(const Plan& p,int seq,int head,Ops& o) {
  int begin=p.offset[seq],end=p.offset[seq+1];
  if(begin==end) return;
  o.pack(p.input[6],p.sequence_base+seq,head,128,chunk(p,begin,head).m(H),true);
  for(int c=begin;c<end;++c) {
    o.used=0; auto x=chunk(p,c,head);
    o.add(x.m(U_Z),o.mm(x.m(W_DW),x.m(H)),-1,x.m(U_Z));
    if(c+1<end)
      o.add(o.last(x.m(H),x.v(EXP)),
            o.mm(o.row(x.m(K),x.v(RATIO)),x.m(U_Z),true),1,chunk(p,c+1,head).m(H));
  }
}

AICORE inline void local_gradient(const Plan& p,int c,int head,Ops& o) {
  auto x=chunk(p,c,head);
  auto attention=o.mul(o.mm(x.m(Q_DQ),x.m(K),false,true),x.m(DECAY));
  o.mm(attention,x.m(DO_DV),true,false,x.m(DU));
  o.mm(o.row(x.m(Q_DQ),x.v(EXP)),x.m(DO_DV),true,false,x.m(DH_DK));
}

AICORE inline void state_gradient(const Plan& p,int seq,int head,Ops& o) {
  Buffer dh{scratch(p)},prev{dh.data+Matrix};
  o.pack(p.input[7],p.sequence_base+seq,head,128,dh,true);
  for(int c=p.offset[seq+1]-1;c>=p.offset[seq];--c) {
    o.used=0; auto x=chunk(p,c,head);
    o.add(x.m(DU),o.mm(o.row(x.m(K),x.v(RATIO)),dh),1,x.m(DU));
    o.add(o.add(o.last(dh,x.v(EXP)),x.m(DH_DK)),o.mm(x.m(W_DW),x.m(DU),true),-1,prev);
    o.copy(dh,x.m(DH_DK));
    auto tmp=dh; dh=prev; prev=tmp;
  }
  o.copy(dh,{(__gm__ float*)p.output[5]+((p.sequence_base+seq)*p.heads+head)*Matrix});
}

AICORE inline Buffer accurate_dq(Chunk x,Ops& o,Buffer dscore) {
  auto dq=o.empty(),local=o.empty(),state=o.empty();
  local_scores(dscore.data,local.data); join();
  o.mm(local,x.m(K),false,false,dq); o.copy(x.m(H),state);
  int keep=o.used;
  for(int start=0;start<128;start+=16) {
    o.used=keep;
    auto d=o.empty(true),k=o.empty(true),factor=o.empty(true);
    block_weights(x.v(EXP).data,x.m(DECAY).data,start,d.data,k.data,factor.data); join();
    o.add(dq,o.mm(o.row(x.m(DO_DV),d),state,false,true),1,dq);
    if(start+16<128)
      o.add(o.last(state,factor),o.mm(o.row(x.m(K),k),x.m(U_Z),true),1,state);
  }
  o.used=keep; return dq;
}

AICORE inline void output_gradient(const Plan& p,int c,int head,Ops& o) {
  auto x=chunk(p,c,head);
  auto dscore=o.mul(o.mm(x.m(DO_DV),x.m(U_Z),false,true),x.m(DECAY));
  auto dq=o.accurate ? accurate_dq(x,o,dscore) :
    o.add(o.row(o.mm(x.m(DO_DV),x.m(H),false,true),x.v(EXP)),o.mm(dscore,x.m(K)));
  auto dkr=o.mm(x.m(U_Z),x.m(DH_DK),false,true);
  auto dk=o.add(o.row(dkr,x.v(RATIO)),o.mm(dscore,x.m(Q_DQ),true));
  o.scale(o.mm(x.m(DU),x.m(H),false,true),-1,x.m(W_DW));
  o.dotrows(x.m(H),x.m(DH_DK),x.v(END_STATE));
  o.dotrows(dkr,o.row(x.m(K),x.v(RATIO)),x.v(END_KEYS));
  o.dotrows(dq,x.m(Q_DQ),x.v(Q_ENERGY));
  o.copy(dq,x.m(Q_DQ)); o.copy(dk,x.m(DH_DK));
}

AICORE inline void wy_gradient(const Plan& p,int c,int head,Ops& o) {
  auto x=chunk(p,c,head);
  auto kb=o.row(x.m(K),o.mul(x.v(BETA),x.v(EXP))),vb=o.row(x.m(V),x.v(BETA));
  auto da=o.mmadd<false,true,false,true>(x.m(W_DW),kb,x.m(DU),vb);
  auto dkb=o.mm(x.m(AT),x.m(W_DW)),dvb=o.mm(x.m(AT),x.m(DU));
  o.call(1,2,dvb,x.v(BETA),{},{},1,x.m(DO_DV));
  auto db=o.add(o.dotrows(dkb,o.row(x.m(K),x.v(EXP))),o.dotrows(dvb,x.m(V)));
  auto dk=o.add(x.m(DH_DK),o.row(dkb,o.mul(x.v(BETA),x.v(EXP))));
  auto raw=o.mm(o.mm(x.m(AT),da),x.m(AT)),dkk=o.empty();
  wy_epilogue(raw.data,x.m(DECAY).data,x.m(KK).data,x.v(BETA).data,
              db.data,dkk.data,x.v(DB).data); join();
  if(o.accurate)
    o.add(o.add(dk,o.mm(dkk,x.m(K))),o.mm(dkk,x.m(K),true),1,x.m(DH_DK));
  else o.add(dk,o.mmadd<false,false,true,false>(dkk,x.m(K),dkk,x.m(K)),1,x.m(DH_DK));
  // Z=beta*residual and dvb is the gradient of the token update. Avoid
  // cancellation between WY paths; retain the general formula at beta=0.
  if(o.accurate)
    o.call(2,3,x.v(DB),o.dotrows(dvb,x.m(U_Z)),x.v(BETA),{},1,x.v(DB));
}

AICORE inline void scatter_matrix(Buffer x,__gm__ float* out,int length,int stride) {
#if defined(__DAV_VEC__)
  Half value; TASSIGN(value,0);
  for(int i=get_subblockid()*64;i<length;i+=128) {
    load<64,128>(value,x.data+i*128);
    int n=min(64,length-i);
    Tile<TileType::Vec,float,64,128,BLayout::RowMajor,DYNAMIC,128> part(n); TASSIGN(part,0);
    GlobalTensor<float,Shape<1,1,1,DYNAMIC,128>,Stride<1,1,1,DYNAMIC,1>> gm(
      out+i*stride,{1,1,1,n,128},{1,1,1,stride,1});
    TSTORE(gm,part); sync();
  }
#endif
}

AICORE inline void scatter_gates(const Plan& p,int c,int first_head,int heads,bool accurate) {
  int seq=sequence(p,c),first=p.start[seq]+(c-p.offset[seq])*128;
  int length=min(128,p.start[seq]+p.length[seq]-first);
#if defined(__DAV_VEC__)
  if(get_subblockid()==0) {
    // Transpose eight complete head rows; no scalar GM gathers. Head groups
    // own whole 32-byte output spans, preventing writes to shared cache lines.
    Tile<TileType::Vec,float,8,128> input,tmp;
    Tile<TileType::Vec,float,128,8> output;
    TASSIGN(input,0); TASSIGN(output,4096); TASSIGN(tmp,8192);
    for(int which=0;which<2;++which) for(int h=first_head;h<first_head+heads;h+=8) {
      int n=min(8,first_head+heads-h);
      TEXPANDS(input,0.0f); sync();
      for(int j=0;j<n;++j) {
        Row row; TASSIGN(row,j*512);
        load<1,128>(row,chunk(p,c,h+j).v(which==0 ? Q_ENERGY : DB).data);
      }
      TTRANS(output,input,tmp); sync();
      if(which==0 && !accurate) reverse_gate_scan();
      Tile<TileType::Vec,float,128,8,BLayout::RowMajor,DYNAMIC,DYNAMIC> part(length,n);
      TASSIGN(part,4096);
      GlobalTensor<float,Shape<1,1,1,DYNAMIC,DYNAMIC>,Stride<1,1,1,DYNAMIC,1>> gm(
        (__gm__ float*)p.output[3+which]+int64_t(first)*p.heads+h,
        {1,1,1,length,n},{1,1,1,p.heads,1});
      TSTORE(gm,part); sync();
    }
  }
#endif
}

AICORE inline void finish(const Plan& p,int c,int first_head,int heads,Ops& o) {
  int seq=sequence(p,c),first=p.start[seq]+(c-p.offset[seq])*128;
  int length=min(128,p.start[seq]+p.length[seq]-first),group=p.heads/p.key_heads;
  if(!o.accurate) o.begin();
  for(int h=first_head;h<first_head+heads;++h) {
    o.used=0; auto x=chunk(p,c,h);
    if(o.accurate) {
      auto dg=o.add(x.v(Q_ENERGY),o.dotrows(x.m(DO_DV),x.m(V)),-1);
      o.call(2,2,dg,x.v(END_STATE),x.v(END_KEYS),x.v(EXP),1,x.v(Q_ENERGY));
    } else finish_gate(x);
    scatter_matrix(x.m(DO_DV),(__gm__ float*)p.output[2]+(int64_t(first)*p.heads+h)*128,length,p.heads*128);
    if(o.accurate) join();
  }
#if defined(__DAV_VEC__)
  // Shared-head sums stay in UB through the final strided output store.
  Half dq,dk,a,b;
  TASSIGN(dq,0); TASSIGN(dk,32768); TASSIGN(a,65536); TASSIGN(b,98304);
  int stripe=get_subblockid()*64,n=max(0,min(64,length-stripe));
  if(n) for(int kh=first_head/group;kh<(first_head+heads)/group;++kh) {
    for(int j=0;j<group;++j) {
      auto x=chunk(p,c,kh*group+j);
      load<64,128>(a,x.m(Q_DQ).data+stripe*128);
      load<64,128>(b,x.m(DH_DK).data+stripe*128);
      TMULS(a,a,p.scale); sync(); // Preserve scale-before-reduction rounding.
      if(j==0) { TMOV(dq,a); TMOV(dk,b); }
      else { TADD(dq,dq,a); TADD(dk,dk,b); }
      sync();
    }
    for(int which=0;which<2;++which) {
      Tile<TileType::Vec,float,64,128,BLayout::RowMajor,DYNAMIC,128> part(n);
      TASSIGN(part,which*32768);
      GlobalTensor<float,Shape<1,1,1,DYNAMIC,128>,Stride<1,1,1,DYNAMIC,1>> gm(
        (__gm__ float*)p.output[which]+(int64_t(first+stripe)*p.key_heads+kh)*128,
        {1,1,1,n,128},{1,1,1,p.key_heads*128,1});
      TSTORE(gm,part); sync();
    }
  }
#endif
  if(o.accurate) join();
  scatter_gates(p,c,first_head,heads,o.accurate);
  if(o.accurate) join(); else sync();
}
#endif

template<int Stage>
__global__ AICORE void gdn_bwd_stage(Plan p) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if defined(__DAV_VEC__)
  set_mask_norm(); set_vector_mask(-1,-1);
#endif
#ifdef __CCE_AICORE__
  if constexpr(Stage==0)
    if(!p.inverse_cache && p.input[0].dtype!=2 && p.input[1].dtype!=2 && p.input[2].dtype!=2)
      inverse_indices();
  // Keep each shared-key group together for dQ/dK reduction. Split value
  // heads only at cache-line boundaries for the interleaved scalar outputs.
  int group=p.heads/p.key_heads,fold=output_head_group(p);
  int groups=p.heads/fold;
  bool emit=fused_output(p);
  bool share=Stage==0 && shared_grams(p);
  int tasks=share ? p.chunks*p.key_heads : Stage==1 ? p.sequences*p.heads : (Stage==3 || (Stage==2 && emit)) ? p.chunks*groups : p.chunks*p.heads;
  for(int task=get_block_idx();task<tasks;task+=get_block_num()) {
    Ops o{scratch(p)+2*Matrix,p.input[0].dtype==2 || p.input[1].dtype==2 || p.input[2].dtype==2};
    int head=task%p.heads,index=task/p.heads;
    if constexpr(Stage==0) {
      if(!o.accurate) {
        int count=share ? group : 1;
        head=share ? (task%p.key_heads)*group : head;
        index=share ? task/p.key_heads : index;
        for(int j=0;j<count;++j) { pipelined_prepare(p,index,head+j,head); }
      }
      else { prepare(p,index,head,o); o.used=0; local_gradient(p,index,head,o); }
    }
    if constexpr(Stage==1) {
      if(o.accurate) { replay(p,index,head,o); o.used=0; state_gradient(p,index,head,o); }
      else pipelined_states(p,index,head);
    }
    if constexpr(Stage==2) {
      if(o.accurate) {
        output_gradient(p,index,head,o); o.used=0;
        wy_gradient(p,index,head,o);
      } else if(emit) {
        int c=task/groups,first_head=(task%groups)*fold;
        for(int h=first_head;h<first_head+fold;++h) {
          pipelined_gradients<true>(p,c,h);
        }
        scatter_gates(p,c,first_head,fold,false); join();
      } else pipelined_gradients(p,index,head);
    }
    if constexpr(Stage==3) finish(p,task/groups,(task%groups)*fold,fold,o);
  }
#endif
}

extern "C" void launch_stages(Plan p,void* stream) {
  gdn_bwd_stage<0><<<BlockCount,nullptr,stream>>>(p);
  gdn_bwd_stage<1><<<BlockCount,nullptr,stream>>>(p);
  gdn_bwd_stage<2><<<BlockCount,nullptr,stream>>>(p);
  if(!fused_output(p)) gdn_bwd_stage<3><<<BlockCount,nullptr,stream>>>(p);
}
