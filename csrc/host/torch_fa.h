/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/
#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

#include <cstdint>
#include <limits>

#include "../kernel/fa_config.h"
#include "utils.h"

// rtGetC2cCtrlAddr is exported by the CANN runtime, but its declaration lives
// in version-specific internal header locations. Keep the stable C ABI here
// instead of adding a CANN-layout-dependent include path to the host target.
extern "C" int32_t rtGetC2cCtrlAddr(uint64_t* addr, uint32_t* len);

extern "C" void pto_launch_fa_fp16(
    uint32_t block_dim, void* stream, void* ffts_addr, void* q, void* k,
    void* v, void* p_tile_fifo, void* exp_max_ififo, void* o_out,
    void* qk_tile_fifo, void* pv_tile_fifo, uint32_t s0, uint32_t s1,
    uint32_t qk_preload, uint32_t batch, uint32_t num_q_heads,
    uint32_t num_kv_heads, int64_t q_batch_stride, int64_t q_head_stride,
    int64_t q_seq_stride, int64_t kv_batch_stride, int64_t kv_head_stride,
    int64_t kv_seq_stride, bool causal);

namespace pto_isa_ops {

/**
 * @brief Flash attention for contiguous fp16 BNSD tensors on Ascend 910B.
 *
 * Q is [B, Nq, S0, 128], while K and V are [B, Nkv, S1, 128].
 * Nkv must divide Nq. S0 must be a multiple of 128 and S1 a multiple
 * of 512. The returned tensor is fp32 with the same shape as Q.
 *
 * @param q Query tensor, fp16 BNSD [B, Nq, S0, 128].
 * @param k Key tensor, fp16 BNSD [B, Nkv, S1, 128].
 * @param v Value tensor, fp16 BNSD [B, Nkv, S1, 128].
 * @param causal If true, apply the causal (lower-triangular) attention mask.
 * @param qk_preload QK pipeline warmup depth; must be in [kFaMinQkPreload,
 *                   kFaCvFifoSize].
 * @return fp32 attention output with the same shape as @p q.
 */
at::Tensor run_fa(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                  bool causal = false, int64_t qk_preload = kFaQkPreload) {
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "fa: Q, K, and V must be 4D BNSD tensors");
  TORCH_CHECK(q.device().type() == DEVICE_TYPE,
              "fa: tensors must be on NPU, got ", q.device());
  TORCH_CHECK(k.device() == q.device() && v.device() == q.device(),
              "fa: Q, K, and V must be on the same NPU device");
  TORCH_CHECK(q.scalar_type() == at::kHalf && k.scalar_type() == at::kHalf &&
                  v.scalar_type() == at::kHalf,
              "fa: Q, K, and V must have dtype fp16");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
              "fa: Q, K, and V must be contiguous");

  const int64_t batch = q.size(0);
  const int64_t num_q_heads = q.size(1);
  const int64_t num_kv_heads = k.size(1);
  const int64_t s0 = q.size(2);
  const int64_t s1 = k.size(2);

  TORCH_CHECK(k.sizes() == v.sizes(), "fa: K and V must have the same shape");
  TORCH_CHECK(k.size(0) == batch,
              "fa: Q, K, and V batch dimensions must match");
  TORCH_CHECK(q.size(3) == kFaHeadSize && k.size(3) == kFaHeadSize,
              "fa: head dimension must be ", kFaHeadSize);
  TORCH_CHECK(
      batch > 0 && num_q_heads > 0 && num_kv_heads > 0 && s0 > 0 && s1 > 0,
      "fa: all BNSD dimensions must be positive");
  TORCH_CHECK(s0 % kFaCubeS0 == 0, "fa: S0 must be a multiple of ", kFaCubeS0,
              ", got ", s0);
  TORCH_CHECK(s1 % kFaTileS1 == 0, "fa: S1 must be a multiple of ", kFaTileS1,
              ", got ", s1);
  TORCH_CHECK(num_q_heads % num_kv_heads == 0,
              "fa: the number of KV heads must divide the number of Q heads");
  TORCH_CHECK(qk_preload >= kFaMinQkPreload && qk_preload <= kFaCvFifoSize,
              "fa: qk_preload must be in [", kFaMinQkPreload, ", ",
              kFaCvFifoSize, "], got ", qk_preload);

  constexpr int64_t kUint32Max =
      static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
  TORCH_CHECK(batch <= kUint32Max && num_q_heads <= kUint32Max &&
                  num_kv_heads <= kUint32Max && s0 <= kUint32Max &&
                  s1 <= kUint32Max,
              "fa: dimensions exceed the uint32 kernel argument range");

  const uint32_t batch_u32 = static_cast<uint32_t>(batch);
  const uint32_t nq_u32 = static_cast<uint32_t>(num_q_heads);
  const uint32_t nkv_u32 = static_cast<uint32_t>(num_kv_heads);
  const uint32_t s0_u32 = static_cast<uint32_t>(s0);
  const uint32_t s1_u32 = static_cast<uint32_t>(s1);
  const uint32_t preload_u32 = static_cast<uint32_t>(qk_preload);
  const uint32_t block_dim =
      static_cast<uint32_t>(FaNumCores(s0_u32, batch_u32, nq_u32));

  const int64_t q_seq_stride = kFaHeadSize;
  const int64_t q_head_stride = s0 * kFaHeadSize;
  const int64_t q_batch_stride = num_q_heads * q_head_stride;
  const int64_t kv_seq_stride = kFaHeadSize;
  const int64_t kv_head_stride = s1 * kFaHeadSize;
  const int64_t kv_batch_stride = num_kv_heads * kv_head_stride;

  at::Tensor output = at::empty(q.sizes(), q.options().dtype(at::kFloat));
  const auto workspace_size =
      static_cast<int64_t>(FaWorkspaceSize(s0_u32, batch_u32, nq_u32));
  at::Tensor workspace =
      at::empty({workspace_size}, q.options().dtype(at::kByte));
  FaScratch scratch =
      FaCarveWorkspace(ConvertType(workspace), s0_u32, batch_u32, nq_u32);

  uint64_t ffts = 0;
  uint32_t ffts_len = 0;
  const auto rt_status = rtGetC2cCtrlAddr(&ffts, &ffts_len);
  TORCH_CHECK(rt_status == 0 && ffts != 0,
              "fa: failed to get the FFTS control address, error ", rt_status);
  void* ffts_addr = reinterpret_cast<void*>(ffts);

  EXEC_KERNEL_CMD(fa_fp16, block_dim, ffts_addr, q, k, v, scratch.p_tile_fifo,
                  scratch.exp_max_ififo, output, scratch.qk_tile_fifo,
                  scratch.pv_tile_fifo, s0_u32, s1_u32, preload_u32, batch_u32,
                  nq_u32, nkv_u32, q_batch_stride, q_head_stride, q_seq_stride,
                  kv_batch_stride, kv_head_stride, kv_seq_stride, causal);

  return output;
}

}  // namespace pto_isa_ops
