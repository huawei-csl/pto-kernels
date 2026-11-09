// Copyright (c) 2026 Huawei Technologies Co., Ltd. See repository LICENSE.
// Three sequence-wide stages; small grids and FP32 inputs retain output assembly.
// Chunk traversal and all numerical work execute in PTO-ISA on NPU.
#include "device_ops.h"
#include "pipeline.h"

#ifdef __CCE_AICORE__
// FP32 inputs retain K=32 partial products and compensated reduction. Add
// applies the existing rounded product directly to its consumer, avoiding a
// full GM temporary and a separate vector-add phase.
template<bool TA=false,bool TB=false,bool Add=false>
AICORE inline void compensated_mm(const Plan& p,__gm__ float* a,__gm__ float* b,
    __gm__ float* out,__gm__ float* base=nullptr,float sign=1) {
  auto parts=scratch(p)+(2+ScratchMatrices)*Matrix;
  partial_products<TA,TB>(a,b,parts); join();
  sum_products<Add>(parts,out,base,sign); join();
}

AICORE inline void accurate_prepare(const Plan& p,int c,int head) {
  auto x=chunk(p,c,head); int seq=sequence(p,c);
  int first=p.start[seq]+(c-p.offset[seq])*128;
  int length=min(128,p.start[seq]+p.length[seq]-first),kh=head/(p.heads/p.key_heads);
  auto a=scratch(p)+2*Matrix,kb=a+Matrix,vb=a+2*Matrix,qe=a+3*Matrix;
  auto hi=a+4*Matrix,lo=hi+128;
  native_pack(p.input[0],first,kh,length,x.m(Q_DQ),false,p.scale);
  native_pack(p.input[1],first,kh,length,x.m(K));
  native_pack(p.input[2],first,head,length,x.m(V));
  native_pack(p.input[5],first,head,length,x.m(DO_DV));
  if(p.gate_cache[0]) {
    int64_t offset=int64_t(head)*p.tokens+first;
    cached_row((__gm__ float*)p.gate_cache[0]+offset,hi,length);
    cached_row((__gm__ float*)p.gate_cache[1]+offset,lo,length);
    cached_row((__gm__ half*)p.gate_cache[2]+offset,x.v(BETA).data,length);
  } else {
    pack_gate(p,3,first,head,length,{hi,true});
    pack_gate(p,4,first,head,length,x.v(BETA));
  }
  join();
  gates(hi,x.v(EXP).data,x.v(RATIO).data,x.m(DECAY).data,1,true,
        p.gate_cache[0] ? lo : nullptr,length); join();
  compensated_mm<false,true>(p,x.m(K).data,x.m(K).data,x.m(KK).data);
  if(p.inverse_cache) {
    Input saved{(uint64_t)inverse_data(p,c,head),{Matrix,128,0,1},p.inverse_dtype};
    native_pack(saved,0,0,128,{a}); join();
    complete_cached_inverse(a); join(); transpose_inverse(a,x.m(AT).data); join();
  } else {
    weighted_lower(x.m(KK).data,x.m(DECAY).data,x.v(BETA).data,a); join();
    blocked_inverse(a,kb,x.m(AT).data);
  }
  prepare_rhs(p.input[2],first,head,length,x,kb,vb,qe); join();
  compensated_mm<true>(p,x.m(AT).data,kb,x.m(W_DW).data);
  compensated_mm<true>(p,x.m(AT).data,vb,x.m(U_Z).data);
  compensated_mm<false,true>(p,x.m(Q_DQ).data,x.m(K).data,a);
  prepare_score(a,x.m(DECAY).data,a,1); join();
  compensated_mm<true>(p,a,x.m(DO_DV).data,x.m(DU).data);
  compensated_mm<true>(p,qe,x.m(DO_DV).data,x.m(DH_DK).data);
}

AICORE inline void accurate_states(const Plan& p,int seq,int head) {
  auto weighted=scratch(p)+2*Matrix,base=weighted+Matrix;
  int begin=p.offset[seq],end=p.offset[seq+1];
  if(begin<end) {
    native_pack(p.input[6],p.sequence_base+seq,head,128,chunk(p,begin,head).m(H),true); join();
  }
  for(int c=begin;c<end;++c) {
    auto x=chunk(p,c,head);
    compensated_mm<false,false,true>(p,x.m(W_DW).data,x.m(H).data,x.m(U_Z).data,x.m(U_Z).data,-1);
    if(c+1<end) {
      matrix_op(x.m(K).data,x.v(RATIO).data,weighted,1,2,1);
      matrix_op(x.m(H).data,x.v(EXP).data,base,1,3,1); join();
      compensated_mm<true,false,true>(p,weighted,x.m(U_Z).data,chunk(p,c+1,head).m(H).data,base);
    }
  }
  auto dh=scratch(p),prev=dh+Matrix;
  native_pack(p.input[7],p.sequence_base+seq,head,128,{dh},true); join();
  for(int c=end-1;c>=begin;--c) {
    auto x=chunk(p,c,head);
    matrix_op(x.m(K).data,x.v(RATIO).data,weighted,1,2,1);
    matrix_op(dh,x.v(EXP).data,base,1,3,1);
    matrix_op(base,x.m(DH_DK).data,base,1,0,1);
    native_copy({dh},x.m(DH_DK)); join();
    compensated_mm<false,false,true>(p,weighted,dh,x.m(DU).data,x.m(DU).data);
    compensated_mm<true,false,true>(p,x.m(W_DW).data,x.m(DU).data,prev,base,-1);
    auto tmp=dh; dh=prev; prev=tmp;
  }
  native_copy({dh},{(__gm__ float*)p.output[5]+((p.sequence_base+seq)*p.heads+head)*Matrix}); join();
}

// Eight local temporal windows avoid the dQ cancellation of a whole-chunk dot.
AICORE inline void accurate_dq(const Plan& p,Chunk x,__gm__ float* score,__gm__ float* dq) {
  auto local=score+Matrix,state=score+2*Matrix,weighted=score+3*Matrix,base=score+4*Matrix;
  auto d=score+10*Matrix,k=d+128,factor=k+128;
  local_scores(score,local); native_copy(x.m(H),{state}); join();
  compensated_mm(p,local,x.m(K).data,dq);
  for(int start=0;start<128;start+=16) {
    block_weights(x.v(EXP).data,x.m(DECAY).data,start,d,k,factor); join();
    matrix_op(x.m(DO_DV).data,d,weighted,1,2,1); join();
    compensated_mm<false,true,true>(p,weighted,state,dq,dq);
    if(start+16<128) {
      matrix_op(x.m(K).data,k,weighted,1,2,1);
      matrix_op(state,factor,base,1,3,1); join();
      compensated_mm<true,false,true>(p,weighted,x.m(U_Z).data,state,base);
    }
  }
}

AICORE inline void accurate_output_gradient(const Plan& p,int c,int head) {
  auto x=chunk(p,c,head); auto s=scratch(p)+2*Matrix;
  auto dkr=s+6*Matrix,dq=s+8*Matrix,dk=s+9*Matrix,weighted=s+4*Matrix;
  compensated_mm<false,true>(p,x.m(DO_DV).data,x.m(U_Z).data,s);
  prepare_score(s,x.m(DECAY).data,s,1); join();
  accurate_dq(p,x,s,dq);
  compensated_mm<false,true>(p,x.m(U_Z).data,x.m(DH_DK).data,dkr);
  matrix_op(dkr,x.v(RATIO).data,dk,1,2,1); join();
  compensated_mm<true,false,true>(p,s,x.m(Q_DQ).data,dk,dk);
  compensated_mm<false,true>(p,x.m(DU).data,x.m(H).data,x.m(W_DW).data,nullptr,-1);
  dot_rows(x.m(H).data,x.m(DH_DK).data,nullptr,x.v(END_STATE).data);
  matrix_op(x.m(K).data,x.v(RATIO).data,weighted,1,2,1);
  dot_rows(dkr,weighted,nullptr,x.v(END_KEYS).data);
  dot_rows(dq,x.m(Q_DQ).data,nullptr,x.v(Q_ENERGY).data);
  native_copy({dq},x.m(Q_DQ)); native_copy({dk},x.m(DH_DK)); join();
}

AICORE inline void accurate_wy_gradient(const Plan& p,int c,int head) {
  auto x=chunk(p,c,head); int seq=sequence(p,c),first=p.start[seq]+(c-p.offset[seq])*128;
  int length=min(128,p.start[seq]+p.length[seq]-first);
  auto kb=scratch(p)+2*Matrix,vb=kb+Matrix,da=kb+2*Matrix,dkb=kb+3*Matrix,dvb=kb+4*Matrix;
  auto dk=kb+5*Matrix,raw=kb+6*Matrix,dkk=kb+7*Matrix,tmp=kb+8*Matrix;
  auto db=kb+10*Matrix,dbv=db+128,be=dbv+128;
  prepare_rhs(p.input[2],first,head,length,x,kb,vb,nullptr); join();
  compensated_mm<false,true>(p,x.m(W_DW).data,kb,da);
  compensated_mm<false,true,true>(p,x.m(DU).data,vb,da,da);
  compensated_mm(p,x.m(AT).data,x.m(W_DW).data,dkb);
  compensated_mm(p,x.m(AT).data,x.m(DU).data,dvb);
  matrix_op(dvb,x.v(BETA).data,x.m(DO_DV).data,1,2,1);
  matrix_op(x.m(K).data,x.v(EXP).data,tmp,1,2,1);
  dot_rows(dkb,tmp,nullptr,db); dot_rows(dvb,x.m(V).data,nullptr,dbv); join();
  vector_op(db,dbv,nullptr,nullptr,db,1,0,1);
  vector_op(x.v(BETA).data,x.v(EXP).data,nullptr,nullptr,be,1,1,1); join();
  matrix_op(dkb,be,dk,1,2,1);
  matrix_op(x.m(DH_DK).data,dk,dk,1,0,1); join();
  compensated_mm(p,x.m(AT).data,da,tmp);
  compensated_mm(p,tmp,x.m(AT).data,raw);
  wy_epilogue(raw,x.m(DECAY).data,x.m(KK).data,x.v(BETA).data,db,dkk,x.v(DB).data); join();
  compensated_mm<false,false,true>(p,dkk,x.m(K).data,dk,dk);
  compensated_mm<true,false,true>(p,dkk,x.m(K).data,x.m(DH_DK).data,dk);
  dot_rows(dvb,x.m(U_Z).data,nullptr,db); join();
  vector_op(x.v(DB).data,db,x.v(BETA).data,nullptr,x.v(DB).data,1,3,1); join();
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

AICORE inline void finish(const Plan& p,int c,int first_head,int heads,bool accurate) {
  int seq=sequence(p,c),first=p.start[seq]+(c-p.offset[seq])*128;
  int length=min(128,p.start[seq]+p.length[seq]-first),group=p.heads/p.key_heads;
  for(int h=first_head;h<first_head+heads;++h) {
    auto x=chunk(p,c,h);
    if(accurate) {
      auto energy=scratch(p)+2*Matrix,dg=energy+128;
      dot_rows(x.m(DO_DV).data,x.m(V).data,nullptr,energy); join();
      vector_op(x.v(Q_ENERGY).data,energy,nullptr,nullptr,dg,1,0,-1); join();
      vector_op(dg,x.v(END_STATE).data,x.v(END_KEYS).data,x.v(EXP).data,x.v(Q_ENERGY).data,1,2,1); join();
    } else finish_gate(x);
    scatter_matrix(x.m(DO_DV),(__gm__ float*)p.output[2]+(int64_t(first)*p.heads+h)*128,length,p.heads*128);
    if(accurate) join();
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
  if(accurate) join();
  scatter_gates(p,c,first_head,heads,accurate);
  if(accurate) join(); else sync();
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
    if(!p.inverse_cache)
      inverse_indices();
  // Keep each shared-key group together for dQ/dK reduction. Split value
  // heads only at cache-line boundaries for the interleaved scalar outputs.
  int group=p.heads/p.key_heads,fold=output_head_group(p);
  int groups=p.heads/fold;
  bool accurate=p.input[0].dtype==2 || p.input[1].dtype==2 || p.input[2].dtype==2;
  bool emit=fused_output(p);
  bool share=Stage==0 && shared_grams(p);
  int tasks=share ? p.chunks*p.key_heads : Stage==1 ? p.sequences*p.heads : (Stage==3 || (Stage==2 && emit)) ? p.chunks*groups : p.chunks*p.heads;
  for(int task=get_block_idx();task<tasks;task+=get_block_num()) {
    int head=task%p.heads,index=task/p.heads;
    if constexpr(Stage==0) {
      if(!accurate) {
        int count=share ? group : 1;
        head=share ? (task%p.key_heads)*group : head;
        index=share ? task/p.key_heads : index;
        for(int j=0;j<count;++j) { pipelined_prepare(p,index,head+j,head); }
      }
      else accurate_prepare(p,index,head);
    }
    if constexpr(Stage==1) {
      if(accurate) accurate_states(p,index,head);
      else pipelined_states(p,index,head);
    }
    if constexpr(Stage==2) {
      if(accurate) {
        accurate_output_gradient(p,index,head);
        accurate_wy_gradient(p,index,head);
      } else if(emit) {
        int c=task/groups,first_head=(task%groups)*fold;
        for(int h=first_head;h<first_head+fold;++h) {
          pipelined_gradients<true>(p,c,h);
        }
        scatter_gates(p,c,first_head,fold,false); join();
      } else pipelined_gradients(p,index,head);
    }
    if constexpr(Stage==3) finish(p,task/groups,(task%groups)*fold,fold,accurate);
  }
#endif
}

extern "C" void pto_launch_gdn_mega_bwd(void* stream,Plan p) {
  gdn_bwd_stage<0><<<BlockCount,nullptr,stream>>>(p);
  gdn_bwd_stage<1><<<BlockCount,nullptr,stream>>>(p);
  gdn_bwd_stage<2><<<BlockCount,nullptr,stream>>>(p);
  if(!fused_output(p)) gdn_bwd_stage<3><<<BlockCount,nullptr,stream>>>(p);
}
