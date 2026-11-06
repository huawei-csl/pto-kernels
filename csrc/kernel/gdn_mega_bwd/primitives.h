// Copyright (c) 2026 Huawei Technologies Co., Ltd. See repository LICENSE.
// FLA-style GDN primitives: padded 128x128 tiles, native FP32 throughout.
#pragma once
#include "../kernel_utils.h"
using namespace pto;

#ifdef __CCE_AICORE__
using Half = Tile<TileType::Vec, float, 64, 128>;
using Full = Tile<TileType::Vec, float, 128, 128>;
using Row = Tile<TileType::Vec, float, 1, 128>;
using Col = Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor>;
using Short = Tile<TileType::Vec, float, 1, 64>;
using Scalar = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, 1, 1>;
AICORE inline void sync() { pipe_barrier(PIPE_ALL); }
// Batch disjoint DMA loads, then fence once before vector consumers.
template<int R, int C, class TileT>
AICORE inline void load_async(TileT& tile, __gm__ float* ptr) {
  GlobalTensor<float, Shape<1,1,1,R,C>, Stride<1,1,1,C,1>> gm(ptr);
  TLOAD(tile, gm);
}
template<int R, int C, class TileT>
AICORE inline void load(TileT& tile, __gm__ float* ptr) {
  sync(); load_async<R,C>(tile,ptr); sync();
}
template<int R, int C, class TileT>
AICORE inline void store(TileT& tile, __gm__ float* ptr) {
  sync();
  GlobalTensor<float, Shape<1,1,1,R,C>, Stride<1,1,1,C,1>> gm(ptr);
  TSTORE(gm, tile); sync();
}
#endif

// Generic square products keep the first port short; every tensor is padded
// to C=D=128. Transposition is a view of the L1 fractal, not a GM copy.
template<bool TA, bool TB, int Parts=1, bool Sum=false, bool TC=false, bool TD=false>
AICORE inline void matmul(__gm__ float* a, __gm__ float* b,
                             __gm__ float* c, int batch, __gm__ float* d=nullptr,__gm__ float* e=nullptr) {
#if defined(__DAV_CUBE__)
  using Mat = Tile<TileType::Mat,float,128,128,BLayout::ColMajor,128,128,SLayout::RowMajor,512>;
  using Trans = Tile<TileType::Mat,float,128,128,BLayout::RowMajor,128,128,SLayout::ColMajor,512>;
  Mat am,bm; Trans at,bt;
  constexpr int K=128/Parts;
  TileLeft<float,128,K> left;
  TileRight<float,K,128> right;
  TileAcc<float,128,128> acc;
  TASSIGN(am,0); TASSIGN(bm,65536);
  TASSIGN(left,0); TASSIGN(right,0); TASSIGN(acc,0);
  if constexpr(Parts==1) {
    for(int h=0;h<batch;++h) {
      GlobalTensor<float,Shape<1,1,1,128,128>,Stride<1,1,1,128,1>> ga(a+h*16384),gb(b+h*16384),gc(c+h*16384);
      TLOAD(am,ga); TLOAD(bm,gb);
      set_flag(PIPE_MTE2,PIPE_MTE1,EVENT_ID0); wait_flag(PIPE_MTE2,PIPE_MTE1,EVENT_ID0);
      if constexpr(TA) { TRESHAPE(at,am); TEXTRACT(left,at,0,0); }
      else TEXTRACT(left,am,0,0);
      if constexpr(TB) { TRESHAPE(bt,bm); TEXTRACT(right,bt,0,0); }
      else TEXTRACT(right,bm,0,0);
      set_flag(PIPE_MTE1,PIPE_M,EVENT_ID0); wait_flag(PIPE_MTE1,PIPE_M,EVENT_ID0);
      TMATMUL(acc,left,right);
      if constexpr(Sum) {
        // Prefetch the second pair while the first product executes. Its L1
        // slots are disjoint; L0 operands are reused only after M completes.
        Mat cm,dm; TASSIGN(cm,131072); TASSIGN(dm,196608);
        GlobalTensor<float,Shape<1,1,1,128,128>,Stride<1,1,1,128,1>> gd(d+h*16384),ge(e+h*16384);
        TLOAD(cm,gd); TLOAD(dm,ge);
        set_flag(PIPE_MTE2,PIPE_MTE1,EVENT_ID0); wait_flag(PIPE_MTE2,PIPE_MTE1,EVENT_ID0);
        set_flag(PIPE_M,PIPE_MTE1,EVENT_ID0); wait_flag(PIPE_M,PIPE_MTE1,EVENT_ID0);
        if constexpr(TC) { TRESHAPE(at,cm); TEXTRACT(left,at,0,0); }
        else TEXTRACT(left,cm,0,0);
        if constexpr(TD) { TRESHAPE(bt,dm); TEXTRACT(right,bt,0,0); }
        else TEXTRACT(right,dm,0,0);
        set_flag(PIPE_MTE1,PIPE_M,EVENT_ID0); wait_flag(PIPE_MTE1,PIPE_M,EVENT_ID0);
        TMATMUL_ACC(acc,acc,left,right);
      }
      set_flag(PIPE_M,PIPE_FIX,EVENT_ID0); wait_flag(PIPE_M,PIPE_FIX,EVENT_ID0);
      TSTORE(gc,acc); sync();
    }
    return;
  }
  for (int h=0; h<batch; ++h) {
    load<128,128>(am,a+h*16384); load<128,128>(bm,b+h*16384);
    for(int part=0;part<Parts;++part) {
      if constexpr(TA) { TRESHAPE(at,am); TEXTRACT(left,at,0,part*K); }
      else TEXTRACT(left,am,0,part*K);
      if constexpr(TB) { TRESHAPE(bt,bm); TEXTRACT(right,bt,part*K,0); }
      else TEXTRACT(right,bm,part*K,0);
      sync(); TMATMUL(acc,left,right); sync();
      if constexpr(Sum) {
        load<128,128>(am,d+h*16384); load<128,128>(bm,e+h*16384);
        if constexpr(TC) { TRESHAPE(at,am); TEXTRACT(left,at,0,0); }
        else TEXTRACT(left,am,0,0);
        if constexpr(TD) { TRESHAPE(bt,bm); TEXTRACT(right,bt,0,0); }
        else TEXTRACT(right,bm,0,0);
        sync(); TMATMUL_ACC(acc,acc,left,right); sync();
      }
      store<128,128>(acc,c+(h*Parts+part)*16384);
    }
  }
#endif
}

// The forward saves [D0,0; P,D1], P=-D1*L10. Complete A10=P*D0 on Cube.
template<bool Transposed=false,bool InputTransposed=false>
AICORE inline void complete_cached_inverse(__gm__ float* factor,__gm__ float* output=nullptr) {
#if defined(__DAV_CUBE__)
  Tile<TileType::Mat,float,128,128,BLayout::ColMajor,128,128,SLayout::RowMajor,512> f;
  TileLeft<float,64,64> left; TileRight<float,64,64> right;
  TileAcc<float,64,64> acc;
  TASSIGN(f,0); TASSIGN(left,0); TASSIGN(right,0); TASSIGN(acc,0);
  load<128,128>(f,factor);
  if constexpr(InputTransposed) {
    TEXTRACT(left,f,0,0); TEXTRACT(right,f,0,64);
  } else if constexpr(Transposed) {
    Tile<TileType::Mat,float,128,128,BLayout::RowMajor,128,128,SLayout::ColMajor,512> trans;
    TRESHAPE(trans,f); TEXTRACT(left,trans,0,0); TEXTRACT(right,trans,0,64);
  } else { TEXTRACT(left,f,64,0); TEXTRACT(right,f,0,0); }
  sync();
  TMATMUL(acc,left,right); sync();
  GlobalTensor<float,Shape<1,1,1,64,64>,Stride<1,1,1,128,1>> gm(Transposed ? output+64 : factor+64*128);
  TSTORE(gm,acc); sync();
#endif
}
template<typename T>
AICORE inline void transpose_inverse(__gm__ T* x,__gm__ float* y) {
#if defined(__DAV_VEC__)
  Tile<TileType::Vec,float,128,64> input;
  Half output,tmp;
  TASSIGN(input,0); TASSIGN(output,32768); TASSIGN(tmp,65536);
  int stripe=get_subblockid();
  GlobalTensor<T,Shape<1,1,1,128,64>,Stride<1,1,1,128,1>> gm(x+stripe*64);
  if constexpr(std::is_same_v<T,float>) TLOAD(input,gm);
  else {
    Tile<TileType::Vec,T,128,64> packed; TASSIGN(packed,98304);
    TLOAD(packed,gm); sync(); TCVT(input,packed,RoundMode::CAST_NONE); sync();
  }
  sync(); TTRANS(output,input,tmp); sync();
  store<64,128>(output,y+stripe*8192);
#endif
}

// Matrix products execute on Cube. Vector cores only add the partial results
// with compensation, shortening FP32 accumulation chains without vector dots.
AICORE inline void sum_products(__gm__ float* parts,__gm__ float* out) {
#if defined(__DAV_VEC__)
  using Part=Tile<TileType::Vec,float,32,128>;
  Part a,b,low,sum,bb,err;
  TASSIGN(a,0); TASSIGN(b,16384); TASSIGN(low,32768);
  TASSIGN(sum,49152); TASSIGN(bb,65536); TASSIGN(err,81920);
  for(int row=get_subblockid()*32;row<128;row+=64) {
    int offset=row*128;
    load<32,128>(a,parts+offset); TEXPANDS(low,0.0f); sync();
    for(int i=1;i<ProductParts;++i) {
      load<32,128>(b,parts+i*16384+offset);
      TADD(sum,a,b); sync(); TSUB(bb,sum,a); sync();
      TSUB(err,sum,bb); sync(); TSUB(err,a,err); sync();
      TSUB(bb,b,bb); sync(); TADD(err,err,bb); sync();
      TADD(low,low,err); sync(); TADDS(a,sum,0.0f); sync();
    }
    TADD(a,a,low); sync(); store<32,128>(a,out+offset);
  }
#endif
}

// Keep dQ's temporal cancellation local for FP32 inputs. Matrix products still
// execute on Cube; these helpers only mask scores and construct decay weights.
AICORE inline void local_scores(__gm__ float* x,__gm__ float* out) {
#if defined(__DAV_VEC__)
  Half a; TASSIGN(a,0); int first=get_subblockid()*64;
  load<64,128>(a,x+first*128);
  for(int i=0;i<64;++i) for(int j=0;j<128;++j)
    if((first+i)/16!=j/16) a.SetValue(i*128+j,0.0f);
  sync(); store<64,128>(a,out+first*128);
#endif
}
AICORE inline void block_weights(__gm__ float* e,__gm__ float* decay,int start,
                                 __gm__ float* d,__gm__ float* k,__gm__ float* factor) {
#if defined(__DAV_VEC__)
  if(get_subblockid()!=0) return;
  Row a,b,c; TASSIGN(a,0); TASSIGN(b,512); TASSIGN(c,1024);
  for(int i=0;i<128;++i) {
    bool inside=i>=start && i<start+16;
    a.SetValue(i,inside ? (start ? decay[i*128+start-1] : e[i]) : 0.0f);
    b.SetValue(i,inside ? decay[(start+15)*128+i] : 0.0f);
  }
  TEXPANDS(c,0.0f); sync();
  c.SetValue(127,start ? decay[(start+15)*128+start-1] : e[start+15]); sync();
  store<1,128>(a,d); store<1,128>(b,k); store<1,128>(c,factor);
#endif
}

// Build a causal mask with broadcasts/clamps instead of 8K scalar stores.
template<class TileT>
AICORE inline void causal_mask(TileT& mask,int stripe,bool strict,int scratch) {
#if defined(__DAV_VEC__)
  Row columns; Short rows; Col col;
  TASSIGN(columns,scratch); TASSIGN(rows,scratch+512); TASSIGN(col,scratch+512);
  Tile<TileType::Vec,int32_t,1,128> ids;
  Tile<TileType::Vec,int32_t,1,64> row_ids;
  Tile<TileType::Vec,float,1,256> tmp;
  TASSIGN(ids,scratch+768); TASSIGN(row_ids,scratch+768); TASSIGN(tmp,scratch+1280);
  TCI<decltype(ids),decltype(tmp),int32_t,1>(ids,0,tmp); sync();
  TCVT(columns,ids,RoundMode::CAST_NONE); sync();
  TCI<decltype(row_ids),decltype(tmp),int32_t,0>(row_ids,stripe*64+(strict ? 0 : 1),tmp); sync();
  TCVT(rows,row_ids,RoundMode::CAST_NONE); sync();
  set_mask_norm(); set_vector_mask(-1,-1);
  TCOLEXPAND(mask,columns); sync(); TROWEXPANDADD(mask,mask,col); sync();
  TMAXS(mask,mask,0.0f); sync(); TMINS(mask,mask,1.0f); sync();
#endif
}

// op: 0 add(alpha*y), 1 multiply, 2 row scale, 3 last-vector-element scale,
// 4 scalar scale, 5 strict lower triangle.
AICORE inline void matrix_op(__gm__ float* x, __gm__ float* y,
                                __gm__ float* out, int batch, int op, float alpha) {
#if defined(__DAV_VEC__)
  Half a,b; Row row; Col col;
  TASSIGN(a,0); TASSIGN(b,32768); TASSIGN(row,65536);
  for(int task=get_subblockid(); task<batch*2; task+=2) {
    int h=task/2, stripe=task%2; int64_t p=(int64_t)task*8192;
    load<64,128>(a,x+p);
    if(op==0 || op==1) {
      load<64,128>(b,y+p);
      if(op==0) { TMULS(b,b,alpha); sync(); TADD(a,a,b); }
      else TMUL(a,a,b);
    } else if(op==2 || op==3) {
      load<1,128>(row,y+h*128);
      if(op==2) { TASSIGN(col,65536+stripe*64*4); TROWEXPANDMUL(a,a,col); }
      else { float last=row.GetValue(127); TMULS(a,a,last); }
    } else if(op==5) {
      causal_mask(b,stripe,true,66048); TMUL(a,a,b);
    } else TMULS(a,a,alpha);
    sync(); store<64,128>(a,out+p);
  }
#endif
}

// Row reduction, with disjoint stripes on the two vector subblocks.
AICORE inline void reduce(__gm__ float* x,__gm__ float* out) {
#if defined(__DAV_VEC__)
  Half a,tmp; Col col; Short result;
  TASSIGN(a,0); TASSIGN(tmp,32768); TASSIGN(col,65536); TASSIGN(result,65536);
  int stripe=get_subblockid();
  load<64,128>(a,x+stripe*8192);
  TROWSUM(col,a,tmp); sync(); store<1,64>(result,out+stripe*64);
#endif
}

// Fuse multiply and reduction so the product never goes to GM. Optional z
// preserves the baseline's (x*y)*z rounding order for the WY beta derivative.
AICORE inline void dot_rows(__gm__ float* x,__gm__ float* y,
                            __gm__ float* z,__gm__ float* out) {
#if defined(__DAV_VEC__)
  Half a,b,tmp; Col col; Short result;
  TASSIGN(a,0); TASSIGN(b,32768); TASSIGN(tmp,65536);
  TASSIGN(col,98304); TASSIGN(result,98304);
  int stripe=get_subblockid(),offset=stripe*8192;
  sync(); load_async<64,128>(a,x+offset); load_async<64,128>(b,y+offset); sync();
  TMUL(a,a,b); sync();
  if(z) { load<64,128>(b,z+offset); TMUL(a,a,b); sync(); }
  TROWSUM(col,a,tmp); sync(); store<1,64>(result,out+stripe*64);
#endif
}

AICORE inline void weighted_lower(__gm__ float* kk,__gm__ float* decay,
                                 __gm__ float* beta,__gm__ float* out) {
#if defined(__DAV_VEC__)
  Half a,b; Row weights; Col col;
  TASSIGN(a,0); TASSIGN(b,32768); TASSIGN(weights,65536);
  int stripe=get_subblockid(); TASSIGN(col,65536+stripe*256);
  load<64,128>(a,kk+stripe*8192); load<64,128>(b,decay+stripe*8192);
  load<1,128>(weights,beta);
  TMUL(a,a,b); sync(); TROWEXPANDMUL(a,a,col); sync();
  causal_mask(b,stripe,true,66048); TMUL(a,a,b); sync();
  store<64,128>(a,out+stripe*8192);
#endif
}

// Fuse inverse-adjoint masking, decay/beta weights and the beta reduction.
// Neither dL nor its elementwise products are written to GM.
AICORE inline void wy_epilogue(__gm__ float* raw,__gm__ float* decay,
    __gm__ float* kk,__gm__ float* beta,__gm__ float* base_db,
    __gm__ float* dkk,__gm__ float* db,__gm__ float* base_db2=nullptr) {
#if defined(__DAV_VEC__)
  Half a,b,c; Tile<TileType::Vec,float,64,64> tmp; Row weights; Col col,sum; Short result,base;
  TASSIGN(a,0); TASSIGN(b,32768); TASSIGN(c,65536); TASSIGN(tmp,98304);
  TASSIGN(weights,114688); TASSIGN(sum,115200); TASSIGN(result,115200);
  TASSIGN(base,115456);
  int stripe=get_subblockid(),offset=stripe*8192;
  load<64,128>(a,raw+offset); TMULS(a,a,-1.0f); sync();
  causal_mask(c,stripe,true,115712); TMUL(a,a,c); sync();
  load<64,128>(b,decay+offset); TMUL(c,a,b); sync();
  load<1,128>(weights,beta); TASSIGN(col,114688+stripe*256);
  TROWEXPANDMUL(c,c,col); sync(); store<64,128>(c,dkk+offset);
  load<64,128>(c,kk+offset); TMUL(a,a,c); sync(); TMUL(a,a,b); sync();
  TROWSUM(sum,a,tmp); sync();
  if(base_db) {
    load<1,64>(base,base_db+stripe*64);
    if(base_db2) {
      Short second; TASSIGN(second,114688);
      load<1,64>(second,base_db2+stripe*64); TADD(base,base,second); sync();
    }
    TADD(result,base,result); sync();
  }
  store<1,64>(result,db+stripe*64);
#endif
}

// Vector algebra; op 2 adds the end-state gate contribution then reverse scans.
AICORE inline void vector_op(__gm__ float* x,__gm__ float* y,__gm__ float* z,
    __gm__ float* e,__gm__ float* out,int batch,int op,float alpha) {
#if defined(__DAV_VEC__)
  if(get_subblockid()!=0) return;
  Row a,b,c,d,tmp; Scalar scalar;
  TASSIGN(a,0); TASSIGN(b,512); TASSIGN(c,1024); TASSIGN(d,1536);
  TASSIGN(tmp,2048); TASSIGN(scalar,2560);
  for(int h=0;h<batch;++h) {
    load<1,128>(a,x+h*128); load<1,128>(b,y+h*128);
    if(op==0) { TMULS(b,b,alpha); sync(); TADD(a,a,b); }
    else if(op==1) TMUL(a,a,b);
    else if(op==3) {
      load<1,128>(c,z+h*128);
      for(int i=0;i<128;++i) {
        float beta=c.GetValue(i);
        if(beta!=0.0f) a.SetValue(i,b.GetValue(i)/beta);
      }
    }
    else {
      load<1,128>(c,z+h*128); load<1,128>(d,e+h*128);
      float last=d.GetValue(127);
      TMULS(b,b,last); sync(); TADD(b,b,c); sync();
      TROWSUM(scalar,b,tmp); sync();
      float total=scalar.GetValue(0), correction=0.0f;
      if(op==4) a.SetValue(127,total+a.GetValue(127));
      else for(int i=127;i>=0;--i) {
        float delta=a.GetValue(i)-correction, next=total+delta;
        correction=(next-total)-delta; total=next; a.SetValue(i,total);
      }
    }
    sync(); store<1,128>(a,out+h*128);
  }
#endif
}

// Cached head-major vectors load contiguously, including short packed tails.
template<class T>
AICORE inline void cached_row(__gm__ T* src,__gm__ float* dst,int valid) {
#if defined(__DAV_VEC__)
  if(get_subblockid()!=0) return;
  Tile<TileType::Vec,T,1,128> input;
  Tile<TileType::Vec,T,1,128,BLayout::RowMajor,1,DYNAMIC> part(valid);
  Row output; TASSIGN(input,0); TASSIGN(part,0); TASSIGN(output,512);
  TEXPANDS(input,static_cast<T>(0)); sync();
  GlobalTensor<T,Shape<1,1,1,1,DYNAMIC>,Stride<1,1,1,128,1>> gm(src,{1,1,1,1,valid});
  TLOAD(part,gm); sync();
  if constexpr(std::is_same_v<T,float>) store<1,128>(input,dst);
  else { TCVT(output,input,RoundMode::CAST_NONE); sync(); store<1,128>(output,dst); }
#endif
}

// Aligned vector tree for native-input gate prefixes; FP32 inputs retain Kahan.
template<class TileT>
AICORE inline void native_prefix(TileT& prefix) {
#if defined(__DAV_VEC__)
  Tile<TileType::Vec,float,128,8> a,b;
  Tile<TileType::Vec,float,128,1,BLayout::ColMajor> col;
  Tile<TileType::Vec,float,8,128> transposed,tmp;
  using Part=Tile<TileType::Vec,float,128,8,BLayout::RowMajor,DYNAMIC,8>;
  TASSIGN(a,98304); TASSIGN(b,102400); TASSIGN(col,0);
  TASSIGN(transposed,106496); TASSIGN(tmp,110592);
  TROWEXPAND(a,col); sync();
  int current=0;
  for(int offset=1;offset<128;offset*=2) {
    int src=current ? 102400 : 98304,dst=current ? 98304 : 102400,n=128-offset;
    Part left(n),right(n),sum(n),head(offset),dest(offset);
    TASSIGN(left,src); TASSIGN(right,src+offset*32); TASSIGN(sum,dst+offset*32);
    TASSIGN(head,src); TASSIGN(dest,dst);
    TADD(sum,left,right); TMOV(dest,head); sync(); current=1-current;
  }
  TTRANS(transposed,b,tmp); sync();
  Row first; TASSIGN(first,106496); TMOV(prefix,first); sync();
#endif
}

// Natural-log gate prefix, exp(G), exp(G_end-G), and causal exp(G_i-G_j).
AICORE inline void gates(__gm__ float* g,__gm__ float* e,
    __gm__ float* r,__gm__ float* decay,int batch,bool accurate,
    __gm__ float* low=nullptr,int valid=128) {
#if defined(__DAV_VEC__)
  if(accurate && get_subblockid()!=0) return;
  Row prefix,expv,ratio,raw; Col col; Half a,b;
  TASSIGN(prefix,0); TASSIGN(expv,512); TASSIGN(ratio,1024);
  TASSIGN(raw,1536);
  TASSIGN(a,2048); TASSIGN(b,34816);
  for(int h=0;h<batch;++h) {
    load<1,128>(prefix,g+h*128);
    float sum=0.0f,correction=0.0f;
    if(low) {
      load<1,128>(raw,low+h*128);
      sum=prefix.GetValue(valid-1); float last_low=raw.GetValue(valid-1);
      for(int i=valid;i<128;++i) { prefix.SetValue(i,sum); raw.SetValue(i,last_low); }
    } else if(!accurate) { native_prefix(prefix); sum=prefix.GetValue(127); }
    else {
      TADDS(raw,prefix,0.0f); sync();
      for(int i=0;i<128;++i) {
        float delta=prefix.GetValue(i)-correction,next=sum+delta;
        correction=(next-sum)-delta; sum=next; prefix.SetValue(i,sum);
      }
    }
    sync(); TEXP(expv,prefix); sync();
    if(low) { TEXP(ratio,raw); sync(); TMUL(expv,expv,ratio); sync(); }
    if(get_subblockid()==0) store<1,128>(expv,e+h*128);
    if(low) {
      TMULS(ratio,prefix,-1.0f); sync(); TADDS(ratio,ratio,sum); sync();
      TMULS(expv,raw,-1.0f); sync(); TADDS(expv,expv,raw.GetValue(127)); sync();
      TADD(ratio,ratio,expv); sync();
    } else if(accurate) {
      // Sum intervals directly: subtracting rounded large prefixes loses bits.
      float tail=0.0f,low=0.0f;
      for(int i=127;i>=0;--i) {
        ratio.SetValue(i,tail);
        float delta=raw.GetValue(i)-low,next=tail+delta;
        low=(next-tail)-delta; tail=next;
      }
      sync();
    } else { TMULS(ratio,prefix,-1.0f); sync(); TADDS(ratio,ratio,sum); sync(); }
    TEXP(ratio,ratio); sync();
    if(get_subblockid()==0) store<1,128>(ratio,r+h*128);
    for(int stripe=accurate ? 0 : get_subblockid();stripe<2;stripe+=accurate ? 1 : 2) {
      TASSIGN(col,stripe*64*4);
      if(accurate && !low) {
        TEXPANDS(a,0.0f); sync();
        for(int i=0;i<64;++i) {
          float interval=0.0f,low=0.0f;
          for(int j=stripe*64+i;j>=0;--j) {
            a.SetValue(i*128+j,interval);
            float delta=raw.GetValue(j)-low,next=interval+delta;
            low=(next-interval)-delta; interval=next;
          }
        }
        sync();
      } else {
        TCOLEXPAND(a,prefix); sync(); TMULS(a,a,-1.0f); sync();
        TROWEXPANDADD(a,a,col); sync();
        if(low) {
          TCOLEXPAND(b,raw); sync(); TMULS(b,b,-1.0f); sync();
          TASSIGN(col,1536+stripe*64*4);
          TROWEXPANDADD(b,b,col); sync(); TADD(a,a,b); sync();
        }
      }
      // Clamp upper-triangular exponents before exp, preventing 0*inf.
      causal_mask(b,stripe,false,67584);
      TMUL(a,a,b); sync();
      TEXP(a,a); sync(); TMUL(a,a,b); sync();
      store<64,128>(a,decay+h*16384+stripe*8192);
    }
  }
#endif
}

// (I+L)^-T by rank-one back substitution, adapted from the repository's
// triangular-inverse kernel. Saves transposed A to avoid a UB transpose.
AICORE inline void inverse(__gm__ float* l,__gm__ float* at,int batch) {
#if defined(__DAV_VEC__)
  if(get_subblockid()!=0) return;
  Full input,y;
  TASSIGN(input,0); TASSIGN(y,65536);
  for(int h=0;h<batch;++h) {
    load<128,128>(input,l+h*16384);
    TEXPANDS(y,0.0f); sync();
    for(int i=0;i<128;++i) y.SetValue(i*128+i,1.0f);
    sync();
    for(int m=127;m>=1;--m) {
      Row pivot; TASSIGN(pivot,65536+m*512);
      for(int start=0;start<m;start+=64) {
        int rows=min(64,m-start);
        Tile<TileType::Vec,float,64,128,BLayout::RowMajor,DYNAMIC,128> tmp(rows),dst(rows);
        Tile<TileType::Vec,float,128,1,BLayout::ColMajor,DYNAMIC,1> coeff(rows);
        TASSIGN(tmp,131072); TASSIGN(dst,65536+start*512);
        TASSIGN(coeff,(m*128+start)*4);
        TCOLEXPAND(tmp,pivot); sync(); TROWEXPANDMUL(tmp,tmp,coeff); sync();
        TSUB(dst,dst,tmp); sync();
      }
    }
    store<128,128>(y,at+h*16384);
  }
#endif
}

// Pack strided inputs, cast, expand shared heads and zero-pad on device.
// Scalars use one row with token positions as columns; matrices use 128 rows.
template<class T>
AICORE inline void pack_input(__gm__ T* x,__gm__ float* y,int heads,
    int rows,int length,int group,int64_t st,int64_t sh,int64_t sd,float alpha=1) {
#if defined(__DAV_VEC__)
  if(rows==128 && (sd==1 || !x)) {
    using InputTile=Tile<TileType::Vec,T,64,128,BLayout::RowMajor,64,128,SLayout::NoneBox,512,PadValue::Zero>;
    using Part=Tile<TileType::Vec,T,64,128,BLayout::RowMajor,DYNAMIC,128,SLayout::NoneBox,512,PadValue::Zero>;
    InputTile input; Half output;
    TASSIGN(input,0); TASSIGN(output,32768);
    for(int task=get_subblockid();task<heads*2;task+=2) {
      int h=task/2,first=(task%2)*64,n=max(0,min(64,length-first));
      if(x && n) {
        Part part(n); TASSIGN(part,0);
        GlobalTensor<T,Shape<1,1,1,DYNAMIC,128>,Stride<1,1,1,DYNAMIC,1>> gm(
          x+(h/group)*sh+first*st,{1,1,1,n,128},{1,1,1,st,1});
        TLOAD(part,gm); sync(); TFILLPAD_INPLACE(input,part); sync();
      } else { TEXPANDS(input,static_cast<T>(0)); sync(); }
      if constexpr(std::is_same_v<T,float>) {
        if(alpha!=1) { TMULS(input,input,alpha); sync(); }
        store<64,128>(input,y+task*8192);
      } else {
        TCVT(output,input,RoundMode::CAST_NONE); sync();
        if(alpha!=1) { TMULS(output,output,alpha); sync(); }
        store<64,128>(output,y+task*8192);
      }
    }
    return;
  }
  Tile<TileType::Vec,T,1,128> input;
  Row output; TASSIGN(input,0); TASSIGN(output,512);
  for(int task=get_subblockid();task<heads*rows;task+=2) {
    int h=task/rows,i=task%rows;
    TEXPANDS(input,static_cast<T>(0)); sync();
    if(x && rows==128 && i<length && sd==1) {
      GlobalTensor<T,Shape<1,1,1,1,128>,Stride<1,1,1,128,1>> gm(x+(h/group)*sh+i*st);
      TLOAD(input,gm); sync();
    } else if(x) {
      for(int j=0;j<128;++j) {
        int t=rows==1 ? j : i;
        if(t<length) input.SetValue(j,x[(h/group)*sh+t*st+(rows==1 ? 0 : j*sd)]);
      }
      sync();
    }
    if constexpr(std::is_same_v<T,float>) { if(alpha!=1) { TMULS(input,input,alpha); sync(); } store<1,128>(input,y+task*128); }
    else { TCVT(output,input,RoundMode::CAST_NONE); sync(); if(alpha!=1) { TMULS(output,output,alpha); sync(); } store<1,128>(output,y+task*128); }
  }
#endif
}
