// Copyright (c) 2026 Huawei Technologies Co., Ltd. See repository LICENSE.
#pragma once
#include <cstdint>

// Packed sequences are submitted in batches of at most 64.
constexpr int MaxSequences = 64, BlockCount = 24, Matrix = 128 * 128;
constexpr int MetadataWords = 200, ChunkBaseWord = 3 * MaxSequences + 1;
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
/// One batch's packed-sequence indexing, stored in device workspace.
struct SequenceMetadata {
  int start[MaxSequences];       ///< Global token starts.
  int length[MaxSequences];      ///< Token counts.
  int offset[MaxSequences + 1];  ///< Local chunk prefix sums.
  int chunk_base;                ///< Global inverse-cache chunk offset.
};
/// Immutable buffers and dimensions shared by all backward stages.
struct KernelArgs {
  Input input[8];          ///< q, k, v, g, beta, dout, h0, dht.
  uint64_t output[6];      ///< FP32 dq, dk, dv, dg, dbeta, dh0 addresses.
  uint64_t workspace;      ///< Reusable device workspace address.
  uint64_t inverse_cache;  ///< Optional saved inverse factors.
  uint64_t cu_seqlens;     ///< Optional device sequence boundaries.
  uint64_t metadata;       ///< Packed-sequence indexing in workspace.
  int64_t cu_stride;       ///< Boundary stride in elements.
  int cu_bytes;            ///< Boundary element size: 4 or 8 bytes.
  int total_sequences;     ///< Sequence count across all launch batches.
  int inverse_chunks;      ///< Saved inverse capacity, or zero without a cache.
  int capacity;  ///< Reserved chunk records; also fixes the stage schedule.
  uint64_t gate_cache[3];  ///< Optional g_t, g_sum, beta_t addresses.
  int heads;               ///< Value head count.
  int key_heads;           ///< Shared query/key head count.
  int tokens;              ///< Total tokens in the packed input.
  float scale;             ///< Query/output attention scale.
  int head_group;          ///< Value heads per query/key head.
  int output_group;        ///< Value heads per output task.
  bool accurate;           ///< FP32 input schedule.
  bool fused_output;       ///< Emit outputs in the gradient stage.
  int inverse_dtype;       ///< Inverse storage: 0=FP16, 1=BF16, 2=FP32.
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
