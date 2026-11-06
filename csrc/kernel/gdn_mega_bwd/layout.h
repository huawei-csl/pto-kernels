// Copyright (c) 2026 Huawei Technologies Co., Ltd. See repository LICENSE.
#pragma once
#include <cstdint>

// Inline packed-sequence metadata keeps the launch independent of host/device
// metadata copies. Larger batches are submitted in groups of 64 sequences.
constexpr int MaxSequences=64, BlockCount=24, Matrix=128*128;
// WY backward peaks at 19 matrices plus vectors. Two additional slots hold
// the running state adjoints; other scratch is reused between chunk tasks.
constexpr int ScratchMatrices=24, ProductParts=4;
constexpr int ScratchSize=(ScratchMatrices+2+ProductParts)*Matrix;
struct Input { uint64_t data; int64_t stride[4]; int64_t dtype; };
struct Plan {
  Input input[8]; uint64_t output[6],workspace,inverse_cache,gate_cache[3];
  int start[MaxSequences],length[MaxSequences],offset[MaxSequences+1];
  int sequences,sequence_base,heads,key_heads,chunks,tokens;
  float scale; int inverse_dtype;
};
// Several slots change meaning only after their previous consumers finish.
enum MatrixSlot { Q_DQ,K,V,DO_DV,AT,KK,DECAY,W_DW,U_Z,H,DU,DH_DK,MatrixSlots };
enum VectorSlot { BETA,EXP,RATIO,Q_ENERGY,END_STATE,END_KEYS,DB,VectorSlots };
constexpr int RecordSize=MatrixSlots*Matrix+VectorSlots*128;

// Host and device share the dispatch decision; small grids retain more tasks.
#ifdef __CCE_AICORE__
__host__ __aicore__
#endif
inline int output_head_group(const Plan& p) {
  int group=p.heads/p.key_heads,fold=8;
  if(p.heads%8) return p.heads;
  while(fold%group) fold+=8;
  return fold;
}
#ifdef __CCE_AICORE__
__host__ __aicore__
#endif
inline bool fused_output(const Plan& p) {
  return p.input[0].dtype!=2 && p.input[1].dtype!=2 && p.input[2].dtype!=2 &&
    p.chunks*(p.heads/output_head_group(p))>=4*BlockCount;
}
