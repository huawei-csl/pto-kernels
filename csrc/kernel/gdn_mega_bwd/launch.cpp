// Copyright (c) 2026 Huawei Technologies Co., Ltd. See repository LICENSE.
// Host metadata and one queued operation; no host loop over token chunks.
#include <algorithm>
#include <array>
#include <vector>
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "layout.h"
using StageLauncher=void (*)(Plan,void*);

extern "C" int64_t workspace_elements(const int64_t* bounds,int sequences,int heads) {
  int64_t largest=0;
  for(int first=0;first<sequences;first+=MaxSequences) {
    int64_t chunks=0;
    for(int s=first;s<std::min(first+MaxSequences,sequences);++s)
      chunks+=(bounds[s+1]-bounds[s]+127)/128;
    largest=std::max(largest,chunks);
  }
  return largest*heads*RecordSize+BlockCount*ScratchSize;
}

extern "C" void gdn_backward(const Input* inputs,const int64_t* bounds,int sequences,
    int heads,int key_heads,float** outputs,float* workspace,float scale,StageLauncher launch_stages) {
  std::vector<Plan> plans;
  for(int first=0;first<sequences;first+=MaxSequences) {
    Plan p{};
    std::copy_n(inputs,8,p.input);
    for(int i=0;i<6;++i) p.output[i]=reinterpret_cast<uint64_t>(outputs[i]);
    p.workspace=reinterpret_cast<uint64_t>(workspace); p.heads=heads; p.key_heads=key_heads; p.scale=scale;
    p.sequence_base=first; p.sequences=std::min(MaxSequences,sequences-first);
    for(int s=0;s<p.sequences;++s) {
      p.start[s]=bounds[first+s]; p.length[s]=bounds[first+s+1]-bounds[first+s];
      p.offset[s+1]=p.offset[s]+(p.length[s]+127)/128;
    }
    p.chunks=p.offset[p.sequences]; plans.push_back(p);
  }
  void* stream=c10_npu::getCurrentNPUStream().stream(false);
  at_npu::native::OpCommand::RunOpApi("gdn_mega_bwd",[=]() -> int {
    for(const auto& p:plans) launch_stages(p,stream);
    return 0;
  });
}
