/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/
#pragma once

#include <cstddef>
#include <cstdint>

// Initial upstream configuration: Ascend 910B, fp16 BNSD inputs, head size 128,
// and a 512-column streaming tile. Keep workspace sizing in this shared header
// so the PyTorch wrapper and kernel launch shim cannot drift apart.
constexpr int kFaHeadSize = 128;
constexpr int kFaCubeS0 = 128;
constexpr int kFaCubeS1 = 128;
constexpr int kFaTileS1 = 512;
constexpr int kFaCvFifoSize = 8;
constexpr int kFaCvFifoConsSyncPeriod = kFaCvFifoSize / 2;
constexpr int kFaQkPreload = 4;
constexpr int kFaMinQkPreload = 2;
constexpr int kFaMaxCores = 24;
constexpr int VEC_CORES = 2;
constexpr std::size_t kFaWorkspaceAlignment = 512;

/// Device-side scratch FIFO pointers carved out of the FA workspace.
struct FaScratch {
  void* p_tile_fifo;    ///< fp16 softmax-probability (P) tile FIFO
  void* exp_max_ififo;  ///< fp32 per-row running exp-max FIFO
  void* qk_tile_fifo;   ///< fp32 QK^T score tile FIFO
  void* pv_tile_fifo;   ///< fp32 PV partial-output tile FIFO
};

constexpr std::size_t FaAlignUp(std::size_t value) {
  return (value + kFaWorkspaceAlignment - 1) & ~(kFaWorkspaceAlignment - 1);
}

constexpr std::size_t FaNumCores(std::size_t s0, std::size_t batch,
                                 std::size_t num_q_heads) {
  const std::size_t row_blocks = s0 / kFaCubeS0;
  const std::size_t total = batch * num_q_heads * row_blocks;
  return total < static_cast<std::size_t>(kFaMaxCores)
             ? total
             : static_cast<std::size_t>(kFaMaxCores);
}

constexpr std::size_t FaWorkspaceSize(std::size_t s0, std::size_t batch,
                                      std::size_t num_q_heads) {
  const std::size_t cores = FaNumCores(s0, batch, num_q_heads);
  constexpr std::size_t p_elements = kFaCvFifoSize * kFaCubeS0 * kFaTileS1;
  constexpr std::size_t exp_max_elements = kFaCvFifoSize * kFaCubeS0;
  constexpr std::size_t qk_elements = kFaCvFifoSize * kFaCubeS0 * kFaTileS1;
  constexpr std::size_t pv_elements = kFaCvFifoSize * kFaCubeS0 * kFaHeadSize;

  const std::size_t sizes[] = {
      p_elements * cores * sizeof(uint16_t),
      exp_max_elements * cores * sizeof(float),
      qk_elements * cores * sizeof(float),
      pv_elements * cores * sizeof(float),
  };

  std::size_t offset = 0;
  for (std::size_t size : sizes) {
    offset = FaAlignUp(offset);
    offset += size;
  }
  return offset;
}

inline FaScratch FaCarveWorkspace(void* workspace, std::size_t s0,
                                  std::size_t batch, std::size_t num_q_heads) {
  const std::size_t cores = FaNumCores(s0, batch, num_q_heads);
  constexpr std::size_t p_elements = kFaCvFifoSize * kFaCubeS0 * kFaTileS1;
  constexpr std::size_t exp_max_elements = kFaCvFifoSize * kFaCubeS0;
  constexpr std::size_t qk_elements = kFaCvFifoSize * kFaCubeS0 * kFaTileS1;
  constexpr std::size_t pv_elements = kFaCvFifoSize * kFaCubeS0 * kFaHeadSize;
  const std::size_t sizes[] = {
      p_elements * cores * sizeof(uint16_t),
      exp_max_elements * cores * sizeof(float),
      qk_elements * cores * sizeof(float),
      pv_elements * cores * sizeof(float),
  };

  void* pointers[4]{};
  const auto base = reinterpret_cast<std::uintptr_t>(workspace);
  std::size_t offset = 0;
  for (std::size_t i = 0; i < 4; ++i) {
    offset = FaAlignUp(offset);
    pointers[i] = reinterpret_cast<void*>(base + offset);
    offset += sizes[i];
  }
  return {pointers[0], pointers[1], pointers[2], pointers[3]};
}
