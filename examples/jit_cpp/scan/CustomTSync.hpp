// CustomTSync.hpp — bisheng-safe C2V/V2C cross-core sync wrapper
//
// Workaround for two bugs in pto-isa TSync primitives (see issue_report.md):
//   • TSync_Custom::record() — flag_id is a runtime uint16_t member, passed as
//     the second arg of ffts_cross_core_sync.  Bisheng requires a compile-time
//     literal there; a runtime value silently produces wrong output.
//   • Event::Init()          — srcPipe is a computed constexpr pipe_t, not a
//     literal; bisheng rejects it as the first arg (compile error).
//
// Fix: FlagID is a template non-type parameter.  The message constant kMsg is
// computed entirely from template arguments at instantiation time, so bisheng
// sees it as a literal in both ffts_cross_core_sync arguments.
//
// Usage (C2V, flag slot 0):
//
//   CustomTSync<0> c2v_sync;
//
//   // Cube side — after TSTORE completes:
//   c2v_sync.record();
//
//   // Vector side — before TLOAD begins:
//   c2v_sync.wait();
//
// Requires: included after <pto/pto-inst.hpp> (provides AICORE, PIPE_MTE3,
//           ffts_cross_core_sync, wait_flag_dev).

#pragma once

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
