// Copyright (c) 2026 Huawei Technologies Co., Ltd. See repository LICENSE.
#pragma once
#include <cstdint>

// Inline packed-sequence metadata keeps the launch independent of host/device
// metadata copies. Larger batches are submitted in groups of 64 sequences.
constexpr int MaxSequences = 64, BlockCount = 24, Matrix = 128 * 128;
// WY backward peaks at 19 matrices plus vectors. Two additional slots hold
// the running state adjoints; other scratch is reused between chunk tasks.
constexpr int ScratchMatrices = 24, ProductParts = 4;
constexpr int ScratchSize = (ScratchMatrices + 2 + ProductParts) * Matrix;
/// Device tensor descriptor.
struct Input {
  uint64_t data;      ///< Device address; zero for an omitted input.
  int64_t stride[4];  ///< Element strides in tensor axis order.
  int64_t dtype;      ///< 0=FP16, 1=BF16, 2=FP32.
};
/// Backward launch buffers and packed-sequence metadata.
struct Plan {
  Input input[8];                ///< q, k, v, g, beta, dout, h0, dht.
  uint64_t output[6];            ///< FP32 dq, dk, dv, dg, dbeta, dh0 addresses.
  uint64_t workspace;            ///< Reusable device workspace address.
  uint64_t inverse_cache;        ///< Optional saved inverse factors.
  uint64_t gate_cache[3];        ///< Optional g_t, g_sum, beta_t addresses.
  int start[MaxSequences];       ///< Global token start of each sequence.
  int length[MaxSequences];      ///< Token count of each sequence.
  int offset[MaxSequences + 1];  ///< Chunk prefix sums within this launch.
  int sequences;                 ///< Number of sequences in this launch.
  int sequence_base;  ///< First sequence's index in the packed batch.
  int heads;          ///< Value head count.
  int key_heads;      ///< Shared query/key head count.
  int chunks;         ///< Total 128-token chunks in this launch.
  int tokens;         ///< Total tokens in the packed input.
  float scale;        ///< Query/output attention scale.
  int inverse_dtype;  ///< Inverse storage: 0=FP16, 1=BF16, 2=FP32.
};
// Several slots change meaning only after their previous consumers finish.
enum MatrixSlot {
  Q_DQ,
  K,
  V,
  DO_DV,
  AT,
  KK,
  DECAY,
  W_DW,
  U_Z,
  H,
  DU,
  DH_DK,
  MatrixSlots
};
enum VectorSlot {
  BETA,
  EXP,
  RATIO,
  Q_ENERGY,
  END_STATE,
  END_KEYS,
  DB,
  VectorSlots
};
constexpr int RecordSize = MatrixSlots * Matrix + VectorSlots * 128;

// Host and device share the dispatch decision; small grids retain more tasks.
#ifdef __CCE_AICORE__
__host__ __aicore__
#endif
    inline int
    output_head_group(const Plan& p) {
  int group = p.heads / p.key_heads, fold = 8;
  if (p.heads % 8) return p.heads;
  while (fold % group) fold += 8;
  return fold;
}
#ifdef __CCE_AICORE__
__host__ __aicore__
#endif
    inline bool
    fused_output(const Plan& p) {
  return p.input[0].dtype != 2 && p.input[1].dtype != 2 &&
         p.input[2].dtype != 2 &&
         p.chunks * (p.heads / output_head_group(p)) >= 4 * BlockCount;
}
