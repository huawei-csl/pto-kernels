# Issue: `TSync_Custom::record()` Produces Wrong Output (Bug 1)

**pto-isa commit under test:**
https://gitcode.com/cann/pto-isa/commit/64aedbb0a3214b4ffdf0415f65d8febf23273f40?ref=master

**Target hardware:** dav-2201 (`__DAV_C220_CUBE__` / `__DAV_C220_VEC__`)
**Compiler:** bisheng `-xcce --npu-arch=dav-2201 -std=c++17`

**Reproducer code:** [`issue_report_tsync/`](issue_report_tsync/)
**Other bugs and analysis:** [`issue_report_others/`](issue_report_others/)

---

## Root cause

`TSync_Custom::record()` (`pto/npu/a2a3/custom/TSync_Custom.hpp`, line 92)
calls:

```cpp
ffts_cross_core_sync(PIPE_FIX, _getFFTSMsg(CV_CORE_SYNC, flag_id));
```

`flag_id` is a **runtime `uint16_t` struct member**.  The bisheng compiler
requires the second argument of `ffts_cross_core_sync` to be a
**compile-time literal**.  When a non-literal is passed, bisheng does not
emit an error; instead it silently generates incorrect code — the sync
message is wrong, the vector core does not wait, and it reads stale GM data.

```cpp
// TSync_Custom.hpp line 88-97
AICORE inline void record() const
{
    if constexpr (is_c2v) {
        ffts_cross_core_sync(PIPE_FIX, _getFFTSMsg(CV_CORE_SYNC, flag_id));
        //                                                         ^^^^^^^^
        //   runtime uint16_t member — bisheng literal requirement violated;
        //   wrong code generated silently
    }
    ...
}
```

## Observed symptom

In the C2V sync kernel (cube writes `gm_input` → `gm_output`, vector adds
`subblockid`):

| | Expected | Observed |
|---|---|---|
| Vector sub-core 0 | `100.0 + 0 = 100.0` | `0.0 + 0 = 0.0` (stale) |
| Vector sub-core 1 | `100.0 + 1 = 101.0` | `0.0 + 1 =   1.0` (stale) |

The observed values match a vector that proceeds immediately without waiting —
reading the zero-initialised `gm_output` before the cube's `TSTORE` completes.

## Workaround

`MyTSync<FlagID>` (see `c2v_sync/MyTSync.hpp`) promotes `FlagID` to a
template non-type parameter so `kMsg` is a compile-time constant:

```cpp
template <uint8_t FlagID, bool IsCubeToVec = true>
struct MyTSync {
    static constexpr uint64_t kMsg =
        1u | (2u << 4u) | ((uint64_t)FlagID << 8u);  // literal at instantiation

    AICORE inline void record() const {
        ffts_cross_core_sync(PIPE_FIX, kMsg);  // both args: compile-time literals
    }
    AICORE inline void wait() const { wait_flag_dev(FlagID); }
};
```

Usage in this kernel: `MyTSync<0> sync; sync.record(); / sync.wait();`
