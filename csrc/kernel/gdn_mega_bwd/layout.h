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
  float scale;
};
// Several slots change meaning only after their previous consumers finish.
enum MatrixSlot { Q_DQ,K,V,DO_DV,AT,KK,DECAY,W_DW,U_Z,H,DU,DH_DK,MatrixSlots };
enum VectorSlot { BETA,EXP,RATIO,Q_ENERGY,END_STATE,END_KEYS,DB,VectorSlots };
constexpr int RecordSize=MatrixSlots*Matrix+VectorSlots*128;
