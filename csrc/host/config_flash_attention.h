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

/// Round @p value up to the next multiple of kFaWorkspaceAlignment.
constexpr std::size_t FaAlignUp(std::size_t value) {
  return (value + kFaWorkspaceAlignment - 1) & ~(kFaWorkspaceAlignment - 1);
}

/**
 * @brief Number of AI cores the launch grid uses for a given problem shape.
 *
 * One core processes one kFaCubeS0-row block of one (batch, q-head) pair,
 * capped at kFaMaxCores. Shared by the launch block_dim and the per-core
 * workspace striding so the host and kernel agree on the core count.
 *
 * @param s0          Query sequence length (a multiple of kFaCubeS0).
 * @param batch       Batch size B.
 * @param num_q_heads Number of query heads Nq.
 * @return Core count in [1, kFaMaxCores].
 */
constexpr std::size_t FaNumCores(std::size_t s0, std::size_t batch,
                                 std::size_t num_q_heads) {
  const std::size_t row_blocks = s0 / kFaCubeS0;
  const std::size_t total = batch * num_q_heads * row_blocks;
  return total < static_cast<std::size_t>(kFaMaxCores)
             ? total
             : static_cast<std::size_t>(kFaMaxCores);
}

/**
 * @brief Total device workspace, in bytes, for the cross-core FA FIFOs.
 *
 * On A2/A3 the cube and vector cores cannot hand tiles to each other
 * directly, so every producer -> consumer step of the pipeline exchanges its
 * tiles through a ring buffer (FIFO) in GM. This workspace is that scratch GM.
 * The pipeline is QK -> P -> PV -> GU, backed by four FIFOs (see FaScratch):
 *   - qk_tile_fifo  (fp32): QK (cube) -> P (vec); one [kFaCubeS0 x kFaTileS1]
 *                           QK^T score tile per slot.
 *   - p_tile_fifo   (fp16): P (vec) -> PV (cube); one [kFaCubeS0 x kFaTileS1]
 *                           softmax-probability tile per slot.
 *   - pv_tile_fifo  (fp32): PV (cube) -> GU (vec); one [kFaCubeS0 x
 * kFaHeadSize] partial-output tile per slot.
 *   - exp_max_ififo (fp32): P's per-row running exp-max correction
 *                           ([kFaCubeS0 x 1] per slot) that GU uses to rescale
 *                           the running output.
 *
 * Each FIFO is a ring of kFaCvFifoSize slots (so a producer can run ahead of
 * its consumer) and is replicated per core, hence the FaNumCores() factor. The
 * four regions are laid out back-to-back, each padded up to
 * kFaWorkspaceAlignment; the returned size is the final end offset.
 * FaCarveWorkspace() partitions a buffer of exactly this size into the same
 * four regions.
 *
 * @param s0          Query sequence length (a multiple of kFaCubeS0).
 * @param batch       Batch size B.
 * @param num_q_heads Number of query heads Nq.
 * @return Workspace size in bytes.
 */
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

/**
 * @brief Partition a raw workspace buffer into the four FA scratch FIFOs.
 *
 * Carves @p workspace into the P, exp-max, QK, and PV FIFOs using the same
 * per-core sizes and kFaWorkspaceAlignment padding as FaWorkspaceSize(), so a
 * buffer allocated from that size is consumed exactly. Returns the device FIFO
 * pointers for the kernel launch.
 *
 * @param workspace   Base pointer of a buffer at least FaWorkspaceSize() bytes.
 * @param s0          Query sequence length (a multiple of kFaCubeS0).
 * @param batch       Batch size B.
 * @param num_q_heads Number of query heads Nq.
 * @return FaScratch holding the four device FIFO pointers.
 */
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
