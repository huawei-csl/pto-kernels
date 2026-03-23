// MyTSync.hpp — bisheng-safe C2V/V2C cross-core sync wrapper
//
// Workaround for TSync_Custom bug (see issue_report_tsync/):
//   TSync_Custom stores flag_id as a runtime uint16_t member, so
//   _getFFTSMsg(CV_CORE_SYNC, flag_id) is not a compile-time literal.
//   Bisheng requires both args of ffts_cross_core_sync to be literals;
//   passing a non-literal compiles silently but generates wrong output.
//
// Fix: FlagID is a template non-type parameter → kMsg is a compile-time
// constant that bisheng folds to a literal at instantiation time.
//
// Requires: included after <pto/pto-inst.hpp>.

#pragma once

// MyTSync<FlagID, IsCubeToVec>
//
//   FlagID       — FFTS flag slot [0..11]; compile-time template param.
//   IsCubeToVec  — true (default): cube calls record(), vector calls wait().
//                  false:          vector calls record(), cube calls wait().
//                  C2V → emits on PIPE_FIX,  V2C → emits on PIPE_MTE3.
template <uint8_t FlagID, bool IsCubeToVec = true>
struct MyTSync {
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
  //      Caller must drain MTE3 first:  pipe_barrier(PIPE_MTE3).
  // V2C: emits ffts_cross_core_sync(PIPE_MTE3, kMsg).
  //      Caller must drain PIPE_V first: pipe_barrier(PIPE_V).
  AICORE inline void record() const {
    if constexpr (IsCubeToVec) {
      ffts_cross_core_sync(PIPE_FIX, kMsg);
    } else {
      ffts_cross_core_sync(PIPE_MTE3, kMsg);
    }
  }

  // Consumer side: stall until the producer's record() signal arrives.
  AICORE inline void wait() const { wait_flag_dev(FlagID); }
};
