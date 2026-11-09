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

extern "C" void pto_launch_gdn_mega_bwd(void* stream, Plan plan);

namespace pto_isa_ops {

// Metadata and allocations only. All gradient arithmetic runs in PTO.
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

  // Sequence boundaries are scheduling metadata, never model data. Preserve
  // their dtype/stride on CPU instead of introducing a tensor cast or scan.
  std::vector<int64_t> bounds{0, tokens};
  if (cu_seqlens) {
    const auto& cu = *cu_seqlens;
    check(cu.dim() == 1 && cu.numel() >= 2 && cu.numel() - 1 <= max_index &&
              (cu.scalar_type() == at::kInt || cu.scalar_type() == at::kLong),
          "cu_seqlens must be a 1D integer tensor");
    auto cpu = cu.cpu();
    bounds.resize(cpu.numel());
    for (int64_t i = 0; i < cpu.numel(); ++i) {
      bounds[i] = cpu.scalar_type() == at::kInt
                      ? cpu.data_ptr<int32_t>()[i * cpu.stride(0)]
                      : cpu.data_ptr<int64_t>()[i * cpu.stride(0)];
    }
  }
  check(bounds.front() == 0 && bounds.back() == tokens &&
            std::is_sorted(bounds.begin(), bounds.end()),
        "cu_seqlens must span [0,T] in nondecreasing order");
  const int sequences = bounds.size() - 1;
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
  int64_t chunks = 0;
  for (int i = 0; i < sequences; ++i)
    chunks += (bounds[i + 1] - bounds[i] + 127) / 128;
  if (inverse_cache) {
    const auto& cache = *inverse_cache;
    check(cache.device() == q.device() && cache.is_contiguous() &&
              (cache.scalar_type() == at::kFloat ||
               cache.scalar_type() == at::kHalf ||
               cache.scalar_type() == at::kBFloat16) &&
              cache.sizes() == at::IntArrayRef({chunks, heads, 128, 128}),
          "inverse_cache must be contiguous FP32/FP16/BF16 [chunks,Hv,128,128] "
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

  std::vector<Plan> plans;
  int64_t chunk_base = 0, largest = 0;
  for (int64_t first = 0; first < sequences; first += MaxSequences) {
    Plan p{};
    p.heads = heads;
    p.key_heads = key_heads;
    p.tokens = tokens;
    p.scale = scale;
    p.sequence_base = first;
    p.sequences = std::min<int64_t>(MaxSequences, sequences - first);
    for (int s = 0; s < p.sequences; ++s) {
      p.start[s] = bounds[first + s];
      p.length[s] = bounds[first + s + 1] - bounds[first + s];
      // Widen before padding the length so INT_MAX cannot wrap.
      p.offset[s + 1] = p.offset[s] + (int64_t(p.length[s]) + 127) / 128;
    }
    p.chunks = p.offset[p.sequences];
    check(int64_t(p.chunks) * heads <= max_index &&
              int64_t(p.sequences) * heads <= max_index,
          "chunk/head task count exceeds int32 indexing");
    largest = std::max(largest, int64_t(p.chunks));
    for (int i = 0; i < 8; ++i) {
      const auto& x = inputs[i];
      p.input[i].dtype = x.defined() ? dtype(x) : 2;
      if (!x.defined()) continue;
      p.input[i].data = reinterpret_cast<uint64_t>(x.data_ptr());
      std::copy(x.strides().begin(), x.strides().end(), p.input[i].stride);
    }
    p.inverse_dtype = inverse_cache ? dtype(*inverse_cache) : 2;
    if (inverse_cache) {
      p.inverse_cache =
          reinterpret_cast<uint64_t>(inverse_cache->data_ptr()) +
          chunk_base * heads * Matrix * (p.inverse_dtype == 2 ? 4 : 2);
    }
    if (gate_cache)
      for (int i = 0; i < 3; ++i)
        p.gate_cache[i] =
            reinterpret_cast<uint64_t>((*gate_cache)[i].data_ptr());
    chunk_base += p.chunks;
    plans.push_back(p);
  }
  const auto options = q.options().dtype(at::kFloat).requires_grad(false);
  std::array<at::Tensor, 6> outputs;
  for (int i = 0; i < 5; ++i)
    outputs[i] = at::empty(inputs[i].sizes(), options);
  outputs[5] = at::empty({sequences, heads, 128, 128}, options);
  auto workspace = at::empty(
      {largest * heads * RecordSize + BlockCount * ScratchSize}, options);
  for (auto& p : plans) {
    p.workspace = reinterpret_cast<uint64_t>(workspace.data_ptr());
    for (int i = 0; i < 6; ++i)
      p.output[i] = reinterpret_cast<uint64_t>(outputs[i].data_ptr());
  }
  // Retain tensors until the queued callback submits every stage, and record
  // the consuming stream so storage from other streams cannot be reused early.
  std::vector<at::Tensor> retained(inputs.begin(), inputs.end());
  retained.insert(retained.end(), outputs.begin(), outputs.end());
  retained.push_back(workspace);
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
      [plans = std::move(plans), retained = std::move(retained),
       stream]() -> int {
        for (const auto& p : plans) pto_launch_gdn_mega_bwd(stream, p);
        return 0;
      });
  return {outputs[0], outputs[1], outputs[2],
          outputs[3], outputs[4], outputs[5]};
}

}  // namespace pto_isa_ops
