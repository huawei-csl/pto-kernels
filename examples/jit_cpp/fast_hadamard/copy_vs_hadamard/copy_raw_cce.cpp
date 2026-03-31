#include "kernel_operator.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_BYTES = 96 * 1024;
constexpr uint32_t TILE_ELEMENTS = TILE_BYTES / sizeof(half);
constexpr uint32_t BLOCK_BYTES = 32;
constexpr uint32_t BLOCK_ELEMENTS = BLOCK_BYTES / sizeof(half);

__aicore__ inline uint32_t RoundUpBlockElements(uint32_t value) {
  return ((value + BLOCK_ELEMENTS - 1) / BLOCK_ELEMENTS) * BLOCK_ELEMENTS;
}

namespace {

class KernelRawCceCopy {
 public:
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t gm_offset,
                              uint32_t element_count) {
    element_count_ = element_count;
    if (element_count_ == 0) {
      return;
    }

    pipe_.InitBuffer(queue_, BUFFER_NUM, TILE_BYTES);
    x_gm_.SetGlobalBuffer((__gm__ half *)x + gm_offset, element_count_);
    y_gm_.SetGlobalBuffer((__gm__ half *)y + gm_offset, element_count_);
  }

  __aicore__ inline void Process() {
    if (element_count_ == 0) {
      return;
    }

    for (uint32_t done = 0; done < element_count_; done += TILE_ELEMENTS) {
      const uint32_t current =
          (element_count_ - done > TILE_ELEMENTS) ? TILE_ELEMENTS
                                                  : (element_count_ - done);

      LocalTensor<half> local = queue_.AllocTensor<half>();
      if ((current % BLOCK_ELEMENTS) == 0) {
        DataCopy(local, x_gm_[done], current);
      } else {
        DataCopyExtParams copy_params{
            1, static_cast<uint32_t>(current * sizeof(half)), 0, 0, 0};
        DataCopyPadExtParams<half> pad_params{false, 0, 0, 0};
        DataCopyPad(local, x_gm_[done], copy_params, pad_params);
      }
      queue_.EnQue(local);

      local = queue_.DeQue<half>();
      event_t store_ready = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE3));
      SetFlag<HardEvent::V_MTE3>(store_ready);
      WaitFlag<HardEvent::V_MTE3>(store_ready);
      if ((current % BLOCK_ELEMENTS) == 0) {
        DataCopy(y_gm_[done], local, current);
      } else {
        DataCopyExtParams copy_params{
            1, static_cast<uint32_t>(current * sizeof(half)), 0, 0, 0};
        DataCopyPad(y_gm_[done], local, copy_params);
      }
      event_t recycle_ready =
          static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE3_MTE2));
      SetFlag<HardEvent::MTE3_MTE2>(recycle_ready);
      WaitFlag<HardEvent::MTE3_MTE2>(recycle_ready);
      queue_.FreeTensor(local);
    }
  }

 private:
  TPipe pipe_;
  TQue<QuePosition::VECIN, BUFFER_NUM> queue_;
  GlobalTensor<half> x_gm_;
  GlobalTensor<half> y_gm_;
  uint32_t element_count_ = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void raw_cce_copy_fp16(GM_ADDR x, GM_ADDR y,
                                                        uint32_t total_elements) {
  const uint32_t block_count = GetBlockNum();
  const uint32_t block_idx = GetBlockIdx();
  const uint32_t elements_per_block = RoundUpBlockElements(
      (total_elements + block_count - 1) / block_count);
  const uint32_t gm_offset = elements_per_block * block_idx;
  if (gm_offset >= total_elements) {
    return;
  }

  uint32_t element_count = elements_per_block;
  if (gm_offset + element_count > total_elements) {
    element_count = total_elements - gm_offset;
  }

  KernelRawCceCopy kernel;
  kernel.Init(x, y, gm_offset, element_count);
  kernel.Process();
}

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *x,
                            uint8_t *y, uint32_t batch, uint32_t n) {
  raw_cce_copy_fp16<<<blockDim, nullptr, stream>>>(x, y, batch * n);
}
