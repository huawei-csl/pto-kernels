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

AICORE inline void set_inter_flag(const pipe_t producer_pipe, const inter_core_event_id event_id) {
  //   bits [3:0]   base    = 1
  //   bits [5:4]   mode    = CV_CORE_SYNC = 2
  //   bits [11:8]  flag_id = event_id
  constexpr uint16_t kBase = 1;
  constexpr uint16_t kMode = 2;  // CV_CORE_SYNC
  constexpr uint16_t FFTS_MODE_BIT_START = 4;
  constexpr uint16_t FFTS_FLAG_ID_BIT_START = 8;
  const uint64_t kMsg = kBase | (kMode << FFTS_MODE_BIT_START) | ((static_cast<uint16_t>(event_id) & 0xf) << FFTS_FLAG_ID_BIT_START);
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


enum SyncDir { CubeToVec, VecToCube };

// CustomTSync<FlagID, Dir>
//
//   FlagID — FFTS flag slot index [0..11]; compile-time template param.
//   Dir    — CubeToVec (default): cube calls record(), vector calls wait().
//            VecToCube:          vector calls record(), cube calls wait().
template <uint8_t FlagID, SyncDir Dir = CubeToVec>
struct CustomTSync {
  // FFTS message word (encoding from TSyncCVID.hpp / _getFFTSMsg):
  //   bits [3:0]   base    = 1
  //   bits [5:4]   mode    = CV_CORE_SYNC = 2
  //   bits [11:8]  flag_id = FlagID
  // Every term is a template/compile-time constant → kMsg is a literal.
  static constexpr uint16_t kBase = 1;
  static constexpr uint16_t kMode = 2;  // CV_CORE_SYNC
  static constexpr uint64_t kMsg =
      kBase | (kMode << 4) | ((uint64_t)FlagID << 8);

  // Producer side: signal that data written to GM is visible.
  //
  // C2V: emits ffts_cross_core_sync(PIPE_FIX, kMsg).
  // V2C: emits ffts_cross_core_sync(PIPE_MTE3, kMsg).
  AICORE inline void record() const {
    if (Dir == CubeToVec) {
      ffts_cross_core_sync(PIPE_FIX, kMsg);
    } else {
      ffts_cross_core_sync(PIPE_MTE3, kMsg);
    }
  }

  // Consumer side: stall until the producer's record() signal arrives.
  AICORE inline void wait() const { wait_flag_dev(FlagID); }
};
