#include "kernel_operator.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_BYTES = 96 * 1024;
constexpr uint32_t BLOCK_BYTES = 32;
constexpr uint32_t STATIC_BATCH = 4096;
constexpr uint32_t STATIC_N = 4096;
constexpr uint32_t STATIC_TOTAL_BYTES = STATIC_BATCH * STATIC_N * sizeof(half);

__aicore__ inline uint32_t RoundUpBlockBytes(uint32_t value) {
  return ((value + BLOCK_BYTES - 1) / BLOCK_BYTES) * BLOCK_BYTES;
}

namespace {

class KernelRawCceStatic4096Copy {
 public:
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t gm_offset_bytes,
                              uint32_t byte_count) {
    byte_count_ = byte_count;
    if (byte_count_ == 0) {
      return;
    }

    pipe_.InitBuffer(queue_, BUFFER_NUM, TILE_BYTES);
    x_gm_.SetGlobalBuffer((__gm__ uint8_t *)x + gm_offset_bytes, byte_count_);
    y_gm_.SetGlobalBuffer((__gm__ uint8_t *)y + gm_offset_bytes, byte_count_);
  }

  __aicore__ inline void Process() {
    if (byte_count_ == 0) {
      return;
    }

    for (uint32_t done = 0; done < byte_count_; done += TILE_BYTES) {
      const uint32_t current_bytes =
          (byte_count_ - done > TILE_BYTES) ? TILE_BYTES : (byte_count_ - done);

      LocalTensor<uint8_t> local = queue_.AllocTensor<uint8_t>();
      if ((current_bytes % BLOCK_BYTES) == 0) {
        DataCopy(local, x_gm_[done], current_bytes);
      } else {
        DataCopyExtParams copy_params{
            1, current_bytes, 0, 0, 0};
        DataCopyPadExtParams<uint8_t> pad_params{false, 0, 0, 0};
        DataCopyPad(local, x_gm_[done], copy_params, pad_params);
      }
      queue_.EnQue(local);

      local = queue_.DeQue<uint8_t>();
      if ((current_bytes % BLOCK_BYTES) == 0) {
        DataCopy(y_gm_[done], local, current_bytes);
      } else {
        DataCopyExtParams copy_params{1, current_bytes, 0, 0, 0};
        DataCopyPad(y_gm_[done], local, copy_params);
      }
      queue_.FreeTensor(local);
    }
  }

 private:
  TPipe pipe_;
  TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> queue_;
  GlobalTensor<uint8_t> x_gm_;
  GlobalTensor<uint8_t> y_gm_;
  uint32_t byte_count_ = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void raw_cce_copy_static_4096_fp16(
    GM_ADDR x, GM_ADDR y) {
  const uint32_t block_count = GetBlockNum();
  const uint32_t block_idx = GetBlockIdx();
  const uint32_t bytes_per_block = RoundUpBlockBytes(
      (STATIC_TOTAL_BYTES + block_count - 1) / block_count);
  const uint32_t gm_offset = bytes_per_block * block_idx;
  if (gm_offset >= STATIC_TOTAL_BYTES) {
    return;
  }

  uint32_t byte_count = bytes_per_block;
  if (gm_offset + byte_count > STATIC_TOTAL_BYTES) {
    byte_count = STATIC_TOTAL_BYTES - gm_offset;
  }

  KernelRawCceStatic4096Copy kernel;
  kernel.Init(x, y, gm_offset, byte_count);
  kernel.Process();
}

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *x,
                            uint8_t *y, uint32_t batch, uint32_t n) {
  if (batch != STATIC_BATCH || n != STATIC_N) {
    return;
  }
  const uint32_t max_blocks = blockDim * 2;
  const uint32_t tiles =
      (STATIC_TOTAL_BYTES + TILE_BYTES - 1) / TILE_BYTES;
  const uint32_t launch_blocks =
      tiles == 0 ? 1 : (tiles < max_blocks ? tiles : max_blocks);
  raw_cce_copy_static_4096_fp16<<<launch_blocks, nullptr, stream>>>(x, y);
}
