// Copyright (c) 2026 Huawei Technologies Co., Ltd. See repository LICENSE.
// FLA-style GDN primitives: padded 128x128 tiles, native FP32 throughout.
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
template<int R, int C, class TileT>
AICORE inline void load(TileT& tile, __gm__ float* ptr) {
  sync();
  GlobalTensor<float, Shape<1,1,1,R,C>, Stride<1,1,1,C,1>> gm(ptr);
  TLOAD(tile, gm); sync();
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
template<bool TA, bool TB>
__global__ AICORE void matmul(__gm__ float* a, __gm__ float* b,
                             __gm__ float* c, int batch) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
#if defined(__DAV_CUBE__)
  using Mat = Tile<TileType::Mat,float,128,128,BLayout::ColMajor,128,128,SLayout::RowMajor,512>;
  using Trans = Tile<TileType::Mat,float,128,128,BLayout::RowMajor,128,128,SLayout::ColMajor,512>;
  Mat am,bm; Trans at,bt;
  TileLeft<float,128,128> left;
  TileRight<float,128,128> right;
  TileAcc<float,128,128> acc;
  TASSIGN(am,0); TASSIGN(bm,65536);
  TASSIGN(left,0); TASSIGN(right,0); TASSIGN(acc,0);
  for (int h=get_block_idx(); h<batch; h+=get_block_num()) {
    load<128,128>(am,a+h*16384); load<128,128>(bm,b+h*16384);
    if constexpr(TA) { TRESHAPE(at,am); TEXTRACT(left,at,0,0); }
    else TEXTRACT(left,am,0,0);
    if constexpr(TB) { TRESHAPE(bt,bm); TEXTRACT(right,bt,0,0); }
    else TEXTRACT(right,bm,0,0);
    sync(); TMATMUL(acc,left,right); sync(); store<128,128>(acc,c+h*16384);
  }
#endif
}

// op: 0 add(alpha*y), 1 multiply, 2 row scale, 3 last-vector-element scale,
// 4 scalar scale, 5 strict lower triangle.
__global__ AICORE void matrix_op(__gm__ float* x, __gm__ float* y,
                                __gm__ float* out, int batch, int op, float alpha) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if defined(__DAV_VEC__)
  if(get_subblockid()!=0) return;
  Half a,b; Row row; Col col;
  TASSIGN(a,0); TASSIGN(b,32768); TASSIGN(row,65536);
  for(int task=get_block_idx(); task<batch*2; task+=get_block_num()) {
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
      for(int i=0;i<64;++i)
        for(int j=stripe*64+i;j<128;++j) a.SetValue(i*128+j,0.0f);
    } else TMULS(a,a,alpha);
    sync(); store<64,128>(a,out+p);
  }
#endif
}

// Row/column sums, with two 64-row stripes fitting comfortably in UB.
__global__ AICORE void reduce(__gm__ float* x,__gm__ float* out,int batch,int cols) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if defined(__DAV_VEC__)
  if(get_subblockid()!=0) return;
  Half a,tmp; Row sum,part; Col col; Short shortrow;
  TASSIGN(a,0); TASSIGN(tmp,32768); TASSIGN(sum,65536); TASSIGN(part,66048);
  TASSIGN(col,66048); TASSIGN(shortrow,66048);
  for(int h=get_block_idx();h<batch;h+=get_block_num()) {
    TEXPANDS(sum,0.0f); sync();
    for(int stripe=0;stripe<2;++stripe) {
      load<64,128>(a,x+h*16384+stripe*8192);
      if(cols) { TCOLSUM(part,a,tmp,true); sync(); TADD(sum,sum,part); sync(); }
      else { TROWSUM(col,a,tmp); sync(); store<1,64>(shortrow,out+h*128+stripe*64); }
    }
    if(cols) store<1,128>(sum,out+h*128);
  }
#endif
}

// Vector algebra; op 2 adds the end-state gate contribution then reverse scans.
__global__ AICORE void vector_op(__gm__ float* x,__gm__ float* y,__gm__ float* z,
    __gm__ float* e,__gm__ float* out,int batch,int op,float alpha) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if defined(__DAV_VEC__)
  if(get_subblockid()!=0) return;
  Row a,b,c,d,tmp; Scalar scalar;
  TASSIGN(a,0); TASSIGN(b,512); TASSIGN(c,1024); TASSIGN(d,1536);
  TASSIGN(tmp,2048); TASSIGN(scalar,2560);
  for(int h=get_block_idx();h<batch;h+=get_block_num()) {
    load<1,128>(a,x+h*128); load<1,128>(b,y+h*128);
    if(op==0) { TMULS(b,b,alpha); sync(); TADD(a,a,b); }
    else if(op==1) TMUL(a,a,b);
    else {
      load<1,128>(c,z+h*128); load<1,128>(d,e+h*128);
      float last=d.GetValue(127);
      TMULS(b,b,last); sync(); TADD(b,b,c); sync();
      TROWSUM(scalar,b,tmp); sync();
      float total=scalar.GetValue(0), correction=0.0f;
      for(int i=127;i>=0;--i) {
        float delta=a.GetValue(i)-correction, next=total+delta;
        correction=(next-total)-delta; total=next; a.SetValue(i,total);
      }
    }
    sync(); store<1,128>(a,out+h*128);
  }
#endif
}

// Natural-log gate prefix, exp(G), exp(G_end-G), and causal exp(G_i-G_j).
__global__ AICORE void gates(__gm__ float* g,__gm__ float* e,
    __gm__ float* r,__gm__ float* decay,int batch) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if defined(__DAV_VEC__)
  if(get_subblockid()!=0) return;
  Row prefix,expv,ratio; Col col; Half a,b;
  TASSIGN(prefix,0); TASSIGN(expv,512); TASSIGN(ratio,1024);
  TASSIGN(a,2048); TASSIGN(b,34816);
  for(int h=get_block_idx();h<batch;h+=get_block_num()) {
    load<1,128>(prefix,g+h*128);
    float sum=0.0f,correction=0.0f;
    for(int i=0;i<128;++i) {
      float delta=prefix.GetValue(i)-correction,next=sum+delta;
      correction=(next-sum)-delta; sum=next; prefix.SetValue(i,sum);
    }
    sync(); TEXP(expv,prefix); sync(); store<1,128>(expv,e+h*128);
    TMULS(ratio,prefix,-1.0f); sync(); TADDS(ratio,ratio,sum); sync();
    TEXP(ratio,ratio); sync(); store<1,128>(ratio,r+h*128);
    for(int stripe=0;stripe<2;++stripe) {
      TASSIGN(col,stripe*64*4);
      TCOLEXPAND(a,prefix); sync(); TMULS(a,a,-1.0f); sync();
      TROWEXPANDADD(a,a,col); sync();
      // Clamp upper-triangular exponents before exp, preventing 0*inf.
      TEXPANDS(b,1.0f); sync();
      for(int i=0;i<64;++i)
        for(int j=stripe*64+i+1;j<128;++j) b.SetValue(i*128+j,0.0f);
      sync(); TMUL(a,a,b); sync();
      TEXP(a,a); sync(); TMUL(a,a,b); sync();
      store<64,128>(a,decay+h*16384+stripe*8192);
    }
  }
#endif
}

// (I+L)^-T by rank-one back substitution, adapted from the repository's
// triangular-inverse kernel. Saves transposed A to avoid a UB transpose.
__global__ AICORE void inverse(__gm__ float* l,__gm__ float* at,int batch) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if defined(__DAV_VEC__)
  if(get_subblockid()!=0) return;
  Full input,y;
  TASSIGN(input,0); TASSIGN(y,65536);
  for(int h=get_block_idx();h<batch;h+=get_block_num()) {
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

__global__ AICORE void group_sum(__gm__ float* x,__gm__ float* out,int heads,int group) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if defined(__DAV_VEC__)
  if(get_subblockid()!=0) return;
  Half a,b; TASSIGN(a,0); TASSIGN(b,32768);
  for(int task=get_block_idx();task<heads*2;task+=get_block_num()) {
    int h=task/2,stripe=task%2;
    TEXPANDS(a,0.0f); sync();
    for(int j=0;j<group;++j) {
      load<64,128>(b,x+(h*group+j)*16384+stripe*8192); TADD(a,a,b); sync();
    }
    store<64,128>(a,out+task*8192);
  }
#endif
}

// Pack strided inputs, cast, expand shared heads and zero-pad on device.
// Scalars use one row with token positions as columns; matrices use 128 rows.
template<class T>
__global__ AICORE void pack_input(__gm__ T* x,__gm__ float* y,int heads,
    int rows,int length,int group,int64_t st,int64_t sh,int64_t sd) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if defined(__DAV_VEC__)
  if(get_subblockid()!=0) return;
  Tile<TileType::Vec,T,1,128> input;
  Row output; TASSIGN(input,0); TASSIGN(output,512);
  for(int task=get_block_idx();task<heads*rows;task+=get_block_num()) {
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
    if constexpr(std::is_same_v<T,float>) store<1,128>(input,y+task*128);
    else { TCVT(output,input,RoundMode::CAST_NONE); sync(); store<1,128>(output,y+task*128); }
  }
#endif
}

// Assemble contiguous [T,H,D] / [T,H] gradients with no host tensor operations.
__global__ AICORE void unpack_output(__gm__ float* x,__gm__ float* y,
    int heads,int rows,int length) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if defined(__DAV_VEC__)
  if(get_subblockid()!=0) return;
  Row value; TASSIGN(value,0);
  int count=length*heads*(rows==1 ? 1 : 128);
  for(int task=get_block_idx();task<(count+127)/128;task+=get_block_num()) {
    int n=min(128,count-task*128);
    if(rows==128) {
      int h=task%heads,t=task/heads;
      load<1,128>(value,x+(h*128+t)*128);
    } else {
      for(int j=0;j<n;++j) {
        int index=task*128+j;
        value.SetValue(j,x[(index%heads)*128+index/heads]);
      }
      sync();
    }
    Tile<TileType::Vec,float,1,128,BLayout::RowMajor,1,DYNAMIC> part(n);
    TASSIGN(part,0);
    GlobalTensor<float,Shape<1,1,1,1,DYNAMIC>,Stride<1,1,1,128,1>> gm(y+task*128,{1,1,1,1,n});
    TSTORE(gm,part); sync();
  }
#endif
}

extern "C" void transfer(void* x,void* y,int dtype,int heads,int rows,int length,
    int group,int64_t st,int64_t sh,int64_t sd,int unpack,void* stream) {
  if(unpack) unpack_output<<<24,nullptr,stream>>>((float*)x,(float*)y,heads,rows,length);
  else if(dtype==0) pack_input<half><<<24,nullptr,stream>>>((half*)x,(float*)y,heads,rows,length,group,st,sh,sd);
  else if(dtype==1) pack_input<bfloat16_t><<<24,nullptr,stream>>>((bfloat16_t*)x,(float*)y,heads,rows,length,group,st,sh,sd);
  else pack_input<float><<<24,nullptr,stream>>>((float*)x,(float*)y,heads,rows,length,group,st,sh,sd);
}

__global__ AICORE void transpose_matrix(__gm__ float* x,__gm__ float* y,int batch) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if defined(__DAV_VEC__)
  if(get_subblockid()!=0) return;
  Tile<TileType::Vec,float,128,64> input;
  Half output,tmp;
  TASSIGN(input,0); TASSIGN(output,32768); TASSIGN(tmp,65536);
  for(int task=get_block_idx();task<batch*2;task+=get_block_num()) {
    GlobalTensor<float,Shape<1,1,1,128,64>,Stride<1,1,1,128,1>> gm(x+(task/2)*16384+(task%2)*64);
    TLOAD(input,gm); sync(); TTRANS(output,input,tmp); sync();
    store<64,128>(output,y+task*8192);
  }
#endif
}

// Tree-reduced dots avoid long cube accumulations before the cancellation-
// sensitive temporal sum in dQ. Both operands contain rows of feature vectors.
__global__ AICORE void score_product(__gm__ float* x,__gm__ float* y,
                                     __gm__ float* out,int batch) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if defined(__DAV_VEC__)
  if(get_subblockid()!=0) return;
  Row a; Half b,product,tmp; Short result;
  TASSIGN(a,0); TASSIGN(b,512); TASSIGN(product,33280);
  TASSIGN(tmp,66048);
  TASSIGN(result,512);
  for(int task=get_block_idx();task<batch*128;task+=get_block_num()) {
    int h=task/128;
    load<1,128>(a,x+task*128);
    for(int stripe=0;stripe<2;++stripe) {
      load<64,128>(b,y+h*16384+stripe*8192);
      TCOLEXPAND(product,a); sync(); TMUL(product,product,b); sync();
      // Error-free TwoSum tree over features. Carry low parts until the final
      // rounding; ordinary FP32 dot errors otherwise accumulate across time.
      Tile<TileType::Vec,float,128,64> high,low;
      TASSIGN(high,512); TASSIGN(low,33280);
      TTRANS(high,product,tmp); sync(); TEXPANDS(low,0.0f); sync();
      for(int n=64;n>=1;n/=2) {
        using Part=Tile<TileType::Vec,float,128,64,BLayout::RowMajor,DYNAMIC,64>;
        Part left(n),right(n),ll(n),rl(n),s(n),bb(n),err(n);
        TASSIGN(left,512); TASSIGN(right,512+n*256);
        TASSIGN(ll,33280); TASSIGN(rl,33280+n*256);
        TASSIGN(s,66048); TASSIGN(bb,98816); TASSIGN(err,131584);
        TADD(s,left,right); sync(); TSUB(bb,s,left); sync();
        TSUB(err,s,bb); sync(); TSUB(err,left,err); sync();
        TSUB(bb,right,bb); sync(); TADD(err,err,bb); sync();
        TADD(ll,ll,rl); sync(); TADD(ll,ll,err); sync();
        TADDS(left,s,0.0f); sync();
      }
      Short tail; TASSIGN(tail,33280);
      TADD(result,result,tail); sync();
      store<1,64>(result,out+task*128+stripe*64);
    }
  }
#endif
}

extern "C" void launch(int kind,int op,void** p,int batch,float alpha,void* stream) {
#define P(i) ((float*)p[i])
  if(kind==0) {
    if(op==0) matmul<false,false><<<24,nullptr,stream>>>(P(0),P(1),P(2),batch);
    if(op==1) matmul<true,false><<<24,nullptr,stream>>>(P(0),P(1),P(2),batch);
    if(op==2) matmul<false,true><<<24,nullptr,stream>>>(P(0),P(1),P(2),batch);
    if(op==3) matmul<true,true><<<24,nullptr,stream>>>(P(0),P(1),P(2),batch);
  }
  if(kind==1) matrix_op<<<24,nullptr,stream>>>(P(0),P(1),P(2),batch,op,alpha);
  if(kind==2) vector_op<<<24,nullptr,stream>>>(P(0),P(1),P(2),P(3),P(4),batch,op,alpha);
  if(kind==3) reduce<<<24,nullptr,stream>>>(P(0),P(1),batch,op);
  if(kind==4) gates<<<24,nullptr,stream>>>(P(0),P(1),P(2),P(3),batch);
  if(kind==5) inverse<<<24,nullptr,stream>>>(P(0),P(1),batch);
  if(kind==6) group_sum<<<24,nullptr,stream>>>(P(0),P(1),batch,op);
  if(kind==7) score_product<<<24,nullptr,stream>>>(P(0),P(1),P(2),batch);
  if(kind==8) transpose_matrix<<<24,nullptr,stream>>>(P(0),P(1),batch);
#undef P
}
