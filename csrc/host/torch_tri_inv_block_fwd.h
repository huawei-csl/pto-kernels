#pragma once
#include <ATen/ATen.h>
#include "utils.h"

extern "C" void pto_launch_tri_inv_block_fwd_fp16(
    uint32_t blockDim, void* stream, void* input, void* output,
    void* workspace, uint32_t tiles, uint32_t heads);

namespace pto_isa_ops {
// Native mega-GDN shape and precision. Each tile represents I + tril(x, -1).
// BSND sequences must contain complete 128-row chunks; no varlen support.
at::Tensor run_tri_inv_block_fwd(const at::Tensor& x,
                                 bool is_bsnd_format = false) {
  TORCH_CHECK(x.device().type() == DEVICE_TYPE,
              "tri_inv_block_fwd: input must be on NPU");
  TORCH_CHECK(x.scalar_type() == at::kHalf && x.is_contiguous(),
              "tri_inv_block_fwd: input must be contiguous fp16");
  TORCH_CHECK(x.dim() >= 2 && x.size(-1) == 128,
              "tri_inv_block_fwd: only 128x128 matrices are supported");
  if (is_bsnd_format) {
    TORCH_CHECK(x.dim() == 4 && x.size(1) % 128 == 0 && x.size(2) > 0,
                "tri_inv_block_fwd: expected [B, S, H, 128], S divisible by 128");
  } else {
    TORCH_CHECK(x.size(-2) == 128,
                "tri_inv_block_fwd: matrices must be square");
  }
  auto output = at::empty_like(x);
  if (x.numel() == 0) return output;
  TORCH_CHECK(x.numel() / (128 * 128) <= UINT32_MAX,
              "tri_inv_block_fwd: too many matrices");
  uint32_t tiles = x.numel() / (128 * 128);
  uint32_t heads = is_bsnd_format ? x.size(2) : 0;
  uint32_t blocks = std::min(tiles, GetNumCubeCores());
  auto workspace = at::empty({blocks, 128 * 128 + 64 * 64},
                              x.options().dtype(at::kFloat));
  EXEC_KERNEL_CMD(tri_inv_block_fwd_fp16, blocks, x, output, workspace,
                  tiles, heads);
  return output;
}
}  // namespace pto_isa_ops
