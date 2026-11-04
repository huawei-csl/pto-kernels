// Copyright (c) 2026 Huawei Technologies Co., Ltd. See repository LICENSE.
// Native staged backward. All tensor arithmetic is in the PTO device kernels.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <vector>
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

extern "C" void launch(int,int,void**,int,float,void*);
extern "C" void transfer(void*,void*,int,int,int,int,int,int64_t,int64_t,int64_t,int,void*);
struct Input { void* data; int64_t stride[4]; int64_t dtype; };
struct Buffer { float* data=nullptr; bool vector=false; };
constexpr int Matrix=128*128, ScratchMatrices=96;

// One arena is reused in stream order, with no allocations or host waits in
// the chunk loops. Boundary states and two state adjoints live outside it.
struct Ops {
  float* arena; int heads; void* stream; int64_t used=0;
  Buffer empty(bool vector=false) {
    Buffer x{arena+used,vector}; used+=int64_t(heads)*(vector ? 128 : Matrix);
    assert(used<=int64_t(ScratchMatrices)*heads*Matrix);
    return x;
  }
  Buffer call(int kind,int op,Buffer a,Buffer b={},Buffer c={},Buffer d={},
              float alpha=1,Buffer out={}) {
    if(!out.data) out=empty(kind==2 || kind==3);
    void* p[]={a.data,b.data,c.data,d.data,out.data};
    if(kind==2 || kind==4) {} // Four operands, then output.
    else if(kind==3 || kind==5 || kind==6 || kind==8) p[1]=out.data;
    else p[2]=out.data;
    launch(kind,op,p,heads,alpha,stream); return out;
  }
  Buffer mm(Buffer a,Buffer b,bool ta=false,bool tb=false) { return call(0,ta+2*tb,a,b); }
  Buffer dots(Buffer a,Buffer b,bool ta=false,bool tb=true) {
    if(ta) a=call(8,0,a);
    if(!tb) b=call(8,0,b);
    return call(7,0,a,b);
  }
  Buffer add(Buffer a,Buffer b,float scale=1,Buffer out={}) {
    return call(a.vector ? 2 : 1,0,a,b,{},{},scale,out);
  }
  Buffer mul(Buffer a,Buffer b) { return call(a.vector ? 2 : 1,1,a,b); }
  Buffer row(Buffer a,Buffer b) { return call(1,2,a,b); }
  Buffer last(Buffer a,Buffer b) { return call(1,3,a,b); }
  Buffer scale(Buffer a,float s) { return call(1,4,a,{},{},{},s); }
  Buffer lower(Buffer a) { return call(1,5,a); }
  Buffer rows(Buffer a) { return call(3,0,a); }
  Buffer pack(const Input& x,int first,int length,int group=1,bool vector=false,
              bool state=false,Buffer out={}) {
    if(!out.data) out=empty(vector);
    auto st=x.stride[state ? 2 : 1],sh=x.stride[state ? 1 : 2];
    int64_t offset=first*x.stride[state ? 0 : 1];
    void* src=x.data ? (char*)x.data+offset*(x.dtype==2 ? 4 : 2) : nullptr;
    transfer(src,out.data,x.dtype,heads,vector ? 1 : 128,length,group,
             st,sh,vector ? 0 : x.stride[3],0,stream);
    return out;
  }
  void unpack(Buffer x,float* out,int first,int length,int output_heads) {
    transfer(x.data,out+int64_t(first)*output_heads*(x.vector ? 1 : 128),2,
             output_heads,x.vector ? 1 : 128,length,1,0,0,0,1,stream);
  }
};

struct WY { Buffer at,e,r,decay,kk,kb,vb,w,u; };
static WY prepare_wy(Ops& o,Buffer k,Buffer v,Buffer g,Buffer beta) {
  auto e=o.empty(true),r=o.empty(true),decay=o.empty();
  void* p[]={g.data,e.data,r.data,decay.data};
  launch(4,0,p,o.heads,1,o.stream);
  auto kk=o.mm(k,k,false,true),l=o.lower(o.row(o.mul(kk,decay),beta));
  auto at=o.call(5,0,l),kb=o.row(k,o.mul(beta,e)),vb=o.row(v,beta);
  return {at,e,r,decay,kk,kb,vb,o.mm(at,kb,true),o.mm(at,vb,true)};
}

// Shared intermediates feed all six gradients; no one-kernel-per-gradient split.
static std::array<Buffer,5> backward_chunk(Ops& o,Buffer q,Buffer k,Buffer v,
    Buffer g,Buffer beta,Buffer dout,Buffer h,Buffer dh,Buffer dh_prev) {
  auto [at,e,r,decay,kk,kb,vb,w,u]=prepare_wy(o,k,v,g,beta);
  auto z=o.add(u,o.mm(w,h),-1),qe=o.row(q,e),kr=o.row(k,r);
  auto attention=o.mul(o.mm(q,k,false,true),decay);
  auto du=o.add(o.dots(attention,dout,true,false),o.mm(kr,dh));
  o.add(o.add(o.last(dh,e),o.dots(qe,dout,true,false)),o.mm(w,du,true),-1,dh_prev);
  auto dscore=o.mul(o.dots(dout,z),decay);
  auto dq=o.add(o.row(o.mm(dout,h,false,true),e),o.dots(dscore,k,false,false));
  auto dkr=o.dots(z,dh);
  auto dk=o.add(o.row(dkr,r),o.dots(dscore,q,true,false));
  auto dw=o.scale(o.dots(du,h),-1);
  auto end_state=o.rows(o.mul(h,dh)),end_keys=o.rows(o.mul(dkr,kr));
  auto da=o.add(o.mm(dw,kb,false,true),o.mm(du,vb,false,true));
  auto dkb=o.mm(at,dw),dvb=o.mm(at,du),dv=o.row(dvb,beta);
  auto db=o.add(o.rows(o.mul(dkb,o.row(k,e))),o.rows(o.mul(dvb,v)));
  dk=o.add(dk,o.row(dkb,o.mul(beta,e)));
  auto dl=o.lower(o.scale(o.mm(o.mm(at,da),at),-1));
  auto dkk=o.row(o.mul(dl,decay),beta);
  dk=o.add(o.add(dk,o.mm(dkk,k)),o.mm(dkk,k,true));
  db=o.add(db,o.rows(o.mul(o.mul(dl,kk),decay)));
  // State-energy identity, avoiding cancellation between WY gate derivatives.
  auto dg=o.add(o.rows(o.mul(dq,q)),o.rows(o.mul(dv,v)),-1);
  dg=o.call(2,2,dg,end_state,end_keys,e);
  return {dq,dk,dv,dg,db};
}

extern "C" int64_t workspace_elements(const int64_t* bounds,int sequences,int heads) {
  int64_t chunks=1;
  for(int s=0;s<sequences;++s) chunks=std::max(chunks,(bounds[s+1]-bounds[s]+127)/128);
  return (chunks+2+ScratchMatrices)*heads*Matrix;
}

extern "C" void gdn_backward(const Input* inputs,const int64_t* bounds,int sequences,
    int heads,int key_heads,float** outputs,float* workspace,float scale) {
  std::array<Input,8> in; std::copy_n(inputs,8,in.begin());
  std::array<float*,6> out; std::copy_n(outputs,6,out.begin());
  std::vector<int64_t> cu(bounds,bounds+sequences+1);
  int64_t matrix=int64_t(heads)*Matrix;
  int64_t checkpoints=workspace_elements(bounds,sequences,heads)/matrix-ScratchMatrices-2;
  void* stream=c10_npu::getCurrentNPUStream().stream(false);
  // Use Torch's host task queue, exactly as the forward launcher does. This
  // preserves ordering and allocator lifetimes without Python/device fences.
  at_npu::native::OpCommand::RunOpApi("gdn_mega_bwd",[=]() -> int {
    Ops o{workspace+(checkpoints+2)*matrix,heads,stream};
    for(int s=0;s<sequences;++s) {
      int start=cu[s],end=cu[s+1],chunks=(end-start+127)/128;
      Buffer dh{workspace+checkpoints*matrix},next{dh.data+matrix};
      o.pack(in[6],s,128,1,false,true,{workspace});
      o.pack(in[7],s,128,1,false,true,dh);
      // Replay only the states required as inputs to later chunks.
      for(int c=0;c+1<chunks;++c) {
        o.used=0; int first=start+c*128;
        auto k=o.pack(in[1],first,128,heads/key_heads),v=o.pack(in[2],first,128);
        auto g=o.pack(in[3],first,128,1,true),beta=o.pack(in[4],first,128,1,true);
        auto wy=prepare_wy(o,k,v,g,beta);
        Buffer h{workspace+c*matrix},hn{h.data+matrix};
        auto z=o.add(wy.u,o.mm(wy.w,h),-1);
        o.add(o.last(h,wy.e),o.mm(o.row(k,wy.r),z,true),1,hn);
      }
      for(int c=chunks-1;c>=0;--c) {
        o.used=0; int first=start+c*128,length=std::min(128,end-first);
        auto q=o.scale(o.pack(in[0],first,length,heads/key_heads),scale);
        auto k=o.pack(in[1],first,length,heads/key_heads),v=o.pack(in[2],first,length);
        auto g=o.pack(in[3],first,length,1,true),beta=o.pack(in[4],first,length,1,true);
        auto dout=o.pack(in[5],first,length);
        auto grads=backward_chunk(o,q,k,v,g,beta,dout,{workspace+c*matrix},dh,next);
        grads[0]=o.scale(grads[0],scale);
        for(int j=0;j<5;++j) {
          int nheads=j<2 ? key_heads : heads;
          if(j<2) {
            auto grouped=o.empty(); void* p[]={grads[j].data,grouped.data};
            launch(6,heads/key_heads,p,key_heads,1,stream); grads[j]=grouped;
          }
          o.unpack(grads[j],out[j],first,length,nheads);
        }
        std::swap(dh,next);
      }
      // State outputs are already head-major, unlike the token gradients.
      o.call(1,4,dh,{},{},{},1,{out[5]+s*matrix});
    }
    return 0;
  });
}
