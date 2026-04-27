#pragma once

#include <pto/pto-inst.hpp>

enum inter_core_event_id : uint16_t {
  IC_EVENT_ID0 = 0,
  IC_EVENT_ID1 = 1,
  IC_EVENT_ID2 = 2,
  IC_EVENT_ID3 = 3,
  IC_EVENT_ID4 = 4,
  IC_EVENT_ID5 = 5,
  IC_EVENT_ID6 = 6,
  IC_EVENT_ID7 = 7
};

AICORE inline void set_inter_flag(const pipe_t producer_pipe,
                                  const inter_core_event_id event_id) {
  //   bits [3:0]   base    = 1
  //   bits [5:4]   mode    = 0: all, 1: AIV within group, 2: AIC and AIV within
  //   group bits [11:8]  flag_id = event_id
  constexpr uint16_t kBase = 1;
  constexpr uint16_t kMode = 2;
  constexpr uint16_t FFTS_MODE_BIT_START = 4;
  constexpr uint16_t FFTS_FLAG_ID_BIT_START = 8;
  const uint64_t kMsg =
      kBase | (kMode << FFTS_MODE_BIT_START) |
      ((static_cast<uint16_t>(event_id) & 0xf) << FFTS_FLAG_ID_BIT_START);
  switch (producer_pipe) {
    case PIPE_FIX:
      ffts_cross_core_sync(PIPE_FIX, kMsg);
      break;
    case PIPE_MTE3:
      ffts_cross_core_sync(PIPE_MTE3, kMsg);
      break;
    default:
      // Unsupported pipe for inter-core sync
      break;
  }
}

AICORE inline void wait_inter_flag(const inter_core_event_id event_id) {
  wait_flag_dev(static_cast<uint16_t>(event_id));
}

AICORE inline void set_inter_all_flag(const pipe_t producer_pipe,
                                      const inter_core_event_id event_id) {
  //   bits [3:0]   base    = 1
  //   bits [5:4]   mode     = 0: all, 1: AIV within group, 2: AIC and AIV
  //   within group bits [11:8]  flag_id = event_id
  constexpr uint16_t kBase = 1;
  constexpr uint16_t kMode = 0;
  constexpr uint16_t FFTS_MODE_BIT_START = 4;
  constexpr uint16_t FFTS_FLAG_ID_BIT_START = 8;
  const uint64_t kMsg =
      kBase | (kMode << FFTS_MODE_BIT_START) |
      ((static_cast<uint16_t>(event_id) & 0xf) << FFTS_FLAG_ID_BIT_START);
  switch (producer_pipe) {
    case PIPE_FIX:
      ffts_cross_core_sync(PIPE_FIX, kMsg);
      break;
    case PIPE_MTE3:
      ffts_cross_core_sync(PIPE_MTE3, kMsg);
      break;
    default:
      // Unsupported pipe for inter-core sync
      break;
  }
}

AICORE inline void wait_inter_all_flag(const inter_core_event_id event_id) {
  wait_flag_dev(static_cast<uint16_t>(event_id));
}

// ─── SyncAllImpl: full cross-core barrier ────────────────────────
constexpr uint16_t SYNC_AIV_FLAG = 12;
constexpr uint16_t SYNC_AIC_FLAG = 11;
constexpr uint16_t SYNC_AIC_AIV_FLAG = 13;
constexpr uint16_t SYNC_AIV_ONLY_ALL = 14;
constexpr uint16_t SYNC_MODE_SHIFT_VALUE = 4;
constexpr uint16_t SYNC_FLAG_SHIFT_VALUE = 8;

AICORE inline uint16_t GetffstMsg(uint16_t mode, uint16_t flagId) {
  return (0x1 + ((mode & 0x3) << SYNC_MODE_SHIFT_VALUE) +
          ((flagId & 0xf) << SYNC_FLAG_SHIFT_VALUE));
}

template <bool isAIVOnly = true>
AICORE inline void SyncAllImpl() {
  pipe_barrier(PIPE_ALL);
  if constexpr (isAIVOnly) {
    ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x0, SYNC_AIV_ONLY_ALL));
    wait_flag_dev(SYNC_AIV_ONLY_ALL);
    return;
  }
#if defined(__DAV_C220_CUBE__)
  wait_flag_dev(SYNC_AIV_FLAG);
  ffts_cross_core_sync(PIPE_FIX, GetffstMsg(0x0, SYNC_AIC_FLAG));
  wait_flag_dev(SYNC_AIC_FLAG);
  ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, SYNC_AIC_AIV_FLAG));
#elif defined(__DAV_C220_VEC__)
  ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, SYNC_AIV_FLAG));
  wait_flag_dev(SYNC_AIC_AIV_FLAG);
#endif
}