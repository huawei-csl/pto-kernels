// Copyright (c) 2026 Huawei Technologies Co., Ltd. See repository LICENSE.
#pragma once

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "../kernel/gdn_mega_bwd/layout.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "utils.h"

extern "C" void pto_launch_gdn_mega_bwd(void* stream, KernelArgs args,
                                        const int64_t* cpu_bounds);

namespace pto_isa_ops {

/// Launch packed GDN backward; all gradient arithmetic runs in PTO.
/// @param q Queries [1,T,Hk,128].
/// @param k Keys [1,T,Hk,128].
/// @param v Values [1,T,Hv,128].
/// @param g Log retention gates [1,T,Hv].
/// @param beta Update weights [1,T,Hv].
/// @param dout Output adjoints, shaped like v.
/// @param initial_state Optional h0 [N,Hv,128,128]; defaults to zero.
/// @param dfinal_state Optional final-state adjoints; defaults to zero.
/// @param cu_seqlens Optional packed sequence boundaries [N+1].
/// @param scale Query/output attention scale.
/// @param inverse_cache Optional forward inverse factors [chunks,Hv,128,128].
/// @param gate_cache Optional forward buffers (g_t, g_sum, beta_t).
/// @return FP32 gradients (dq, dk, dv, dg, dbeta, dh0).
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor>
run_gdn_mega_bwd(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                 const at::Tensor& g, const at::Tensor& beta,
                 const at::Tensor& dout,
                 const std::optional<at::Tensor>& initial_state,
                 const std::optional<at::Tensor>& dfinal_state,
                 const std::optional<at::Tensor>& cu_seqlens, double scale,
                 const std::optional<at::Tensor>& inverse_cache,
                 const std::optional<std::vector<at::Tensor>>& gate_cache) {
  auto check = [](bool condition, const char* message) {
    TORCH_CHECK(condition, "gdn_mega_bwd: ", message);
  };
  auto dtype = [](const at::Tensor& x) {
    if (x.scalar_type() == at::kHalf) return 0;
    if (x.scalar_type() == at::kBFloat16) return 1;
    return 2;
  };
  check(q.device().type() == DEVICE_TYPE, "inputs must be on NPU");
  check(q.dim() == 4 && q.size(0) == 1 && q.size(3) == 128 &&
            k.sizes() == q.sizes(),
        "q/k must have shape [1,T,Hk,128]");
  const int64_t tokens = q.size(1), key_heads = q.size(2);
  check(
      v.dim() == 4 && v.size(0) == 1 && v.size(1) == tokens && v.size(3) == 128,
      "v must have shape [1,T,Hv,128]");
  const int64_t heads = v.size(2);
  check(heads > 0 && key_heads > 0 && heads % key_heads == 0 &&
            g.sizes() == at::IntArrayRef({1, tokens, heads}) &&
            beta.sizes() == g.sizes() && dout.sizes() == v.sizes(),
        "invalid heads, gates or dout shapes");
  constexpr int64_t max_index = std::numeric_limits<int>::max();
  check(tokens <= max_index && heads <= max_index && key_heads <= max_index,
        "tokens and heads must fit the kernel's int32 indexing");
  const c10::DeviceGuard device_guard(q.device());
  const at::NoGradGuard no_grad;

  // CPU boundaries are read directly; device boundaries stay on device.
  const bool device_bounds = cu_seqlens && !cu_seqlens->is_cpu();
  std::vector<int64_t> bounds;
  int sequences = 1;
  if (cu_seqlens) {
    const auto& cu = *cu_seqlens;
    check(cu.dim() == 1 && cu.numel() >= 2 && cu.numel() - 1 <= max_index &&
              (cu.scalar_type() == at::kInt || cu.scalar_type() == at::kLong),
          "cu_seqlens must be a 1D integer tensor");
    check(cu.is_cpu() || cu.device() == q.device(),
          "cu_seqlens must be on CPU or the input NPU");
    sequences = cu.numel() - 1;
    if (!device_bounds) {
      bounds.resize(cu.numel());
      for (int64_t i = 0; i < cu.numel(); ++i)
        bounds[i] = cu.scalar_type() == at::kInt
                        ? cu.data_ptr<int32_t>()[i * cu.stride(0)]
                        : cu.data_ptr<int64_t>()[i * cu.stride(0)];
    }
  }
  if (!bounds.empty())
    check(bounds.front() == 0 && bounds.back() == tokens &&
              std::is_sorted(bounds.begin(), bounds.end()),
          "cu_seqlens must span [0,T] in nondecreasing order");
  std::array<at::Tensor, 8> inputs{q,
                                   k,
                                   v,
                                   g,
                                   beta,
                                   dout,
                                   initial_state.value_or(at::Tensor()),
                                   dfinal_state.value_or(at::Tensor())};
  for (const auto& x : inputs) {
    if (!x.defined()) continue;
    check(x.device() == q.device() && (x.scalar_type() == at::kHalf ||
                                       x.scalar_type() == at::kBFloat16 ||
                                       x.scalar_type() == at::kFloat),
          "all data tensors must share the NPU and use FP16/BF16/FP32");
  }
  for (int i : {6, 7}) {
    check(
        !inputs[i].defined() ||
            inputs[i].sizes() == at::IntArrayRef({sequences, heads, 128, 128}),
        "state must have shape [N,Hv,128,128]");
  }
  int64_t chunks = (tokens + 127) / 128;
  // Device lengths are unknown on the host; reserve a shape-derived upper
  // bound.
  int64_t capacity = std::min(
      tokens,
      (tokens + 127 * std::min<int64_t>(sequences, MaxSequences)) / 128);
  if (!bounds.empty()) {
    chunks = capacity = 0;
    int64_t batch_chunks = 0;
    for (int i = 0; i < sequences; ++i) {
      int64_t n = (bounds[i + 1] - bounds[i] + 127) / 128;
      chunks += n;
      batch_chunks += n;
      capacity = std::max(capacity, batch_chunks);
      if ((i + 1) % MaxSequences == 0) batch_chunks = 0;
    }
  }
  if (inverse_cache) {
    const auto& cache = *inverse_cache;
    check(
        cache.device() == q.device() && cache.is_contiguous() &&
            (cache.scalar_type() == at::kFloat ||
             cache.scalar_type() == at::kHalf ||
             cache.scalar_type() == at::kBFloat16) &&
            cache.dim() == 4 && cache.size(0) >= chunks &&
            cache.size(0) <= max_index && cache.size(1) == heads &&
            cache.size(2) == 128 && cache.size(3) == 128,
        "inverse_cache must be contiguous FP32/FP16/BF16 [capacity,Hv,128,128] "
        "on NPU");
  }
  if (gate_cache) {
    check(gate_cache->size() == 3,
          "gate_cache must contain (g_t, g_sum, beta_t)");
    for (int i = 0; i < 3; ++i) {
      const auto& x = (*gate_cache)[i];
      check(x.device() == q.device() && x.is_contiguous() &&
                x.numel() == heads * tokens &&
                x.scalar_type() == (i == 2 ? at::kHalf : at::kFloat),
            "gate_cache must be forward's contiguous (g_t, g_sum, beta_t) on "
            "NPU");
    }
  }

  KernelArgs args{};
  args.heads = heads;
  args.key_heads = key_heads;
  args.tokens = tokens;
  args.scale = scale;
  args.total_sequences = sequences;
  if (device_bounds) {
    args.cu_seqlens = reinterpret_cast<uint64_t>(cu_seqlens->data_ptr());
    args.cu_stride = cu_seqlens->stride(0);
    args.cu_bytes = cu_seqlens->element_size();
  }
  for (int i = 0; i < 8; ++i) {
    const auto& x = inputs[i];
    args.input[i].dtype = x.defined() ? dtype(x) : 2;
    if (!x.defined()) continue;
    args.input[i].data = reinterpret_cast<uint64_t>(x.data_ptr());
    std::copy(x.strides().begin(), x.strides().end(), args.input[i].stride);
  }
  args.inverse_dtype = inverse_cache ? dtype(*inverse_cache) : 2;
  if (inverse_cache) {
    args.inverse_cache = reinterpret_cast<uint64_t>(inverse_cache->data_ptr());
    args.inverse_chunks = inverse_cache->size(0);
  }
  if (gate_cache)
    for (int i = 0; i < 3; ++i)
      args.gate_cache[i] =
          reinterpret_cast<uint64_t>((*gate_cache)[i].data_ptr());

  if (inverse_cache)
    capacity = std::min(capacity, int64_t(args.inverse_chunks));
  check(capacity * heads <= max_index &&
            int64_t(std::min(sequences, MaxSequences)) * heads <= max_index,
        "chunk/head task count exceeds int32 indexing");
  args.capacity = capacity;
  const auto options = q.options().dtype(at::kFloat).requires_grad(false);
  std::array<at::Tensor, 6> outputs;
  for (int i = 0; i < 5; ++i)
    outputs[i] = at::empty(inputs[i].sizes(), options);
  outputs[5] = at::empty({sequences, heads, 128, 128}, options);
  const int64_t metadata_offset =
      capacity * heads * RecordSize + BlockCount * ScratchSize;
  const int64_t batches =
      (int64_t(sequences) + MaxSequences - 1) / MaxSequences;
  const bool needs_metadata = device_bounds || sequences > 1;
  auto workspace = at::empty(
      {metadata_offset + (needs_metadata ? batches * MetadataWords : 0)},
      options);
  args.workspace = reinterpret_cast<uint64_t>(workspace.data_ptr());
  if (needs_metadata)
    args.metadata = args.workspace + metadata_offset * sizeof(float);
  for (int i = 0; i < 6; ++i)
    args.output[i] = reinterpret_cast<uint64_t>(outputs[i].data_ptr());
  // Retain tensors until the queued callback submits every stage, and record
  // the consuming stream so storage from other streams cannot be reused early.
  std::vector<at::Tensor> retained;
  retained.reserve(20);
  retained.insert(retained.end(), inputs.begin(), inputs.end());
  retained.insert(retained.end(), outputs.begin(), outputs.end());
  retained.push_back(workspace);
  if (device_bounds) retained.push_back(*cu_seqlens);
  if (inverse_cache) retained.push_back(*inverse_cache);
  if (gate_cache)
    retained.insert(retained.end(), gate_cache->begin(), gate_cache->end());
  auto npu_stream = c10_npu::getCurrentNPUStream();
  for (const auto& x : retained)
    if (x.defined())
      c10_npu::NPUCachingAllocator::recordStream(x.storage().data_ptr(),
                                                 npu_stream);
  void* stream = npu_stream.stream(false);
  at_npu::native::OpCommand::RunOpApi(
      "gdn_mega_bwd",
      [args, bounds = std::move(bounds), retained = std::move(retained),
       stream]() -> int {
        pto_launch_gdn_mega_bwd(stream, args,
                                bounds.empty() ? nullptr : bounds.data());
        return 0;
      });
  return {outputs[0], outputs[1], outputs[2],
          outputs[3], outputs[4], outputs[5]};
}

}  // namespace pto_isa_ops
