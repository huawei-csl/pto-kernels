# Issue: `TSync_Custom::record()` Produces Wrong Output; `Event::Init()` Fails to Compile

**pto-isa commit under test:**
https://gitcode.com/cann/pto-isa/commit/64aedbb0a3214b4ffdf0415f65d8febf23273f40?ref=master

**Target hardware:** dav-2201 (`__DAV_C220_CUBE__` / `__DAV_C220_VEC__`)
**Compiler:** bisheng `-xcce --npu-arch=dav-2201 -std=c++17`

---

## Background

The C2V sync kernel (cube stores to GM, vector loads and processes) uses
`ffts_cross_core_sync` on the cube side to signal the vector side and
`wait_flag_dev` on the vector side to wait.  The pto-isa library provides
two wrappers for this pattern:

- **`TSync_Custom`** (`pto/npu/a2a3/custom/TSync_Custom.hpp`)
- **`Event`** (`pto/npu/a2a3/TSync.hpp`)

Both wrappers are broken with the bisheng compiler.  `TPipe::Producer` works
correctly.  Raw intrinsic calls with compile-time-constant arguments also work.

---

## Bug 1 — `TSync_Custom::record()`: wrong output, no compile error

### Root cause

`TSync_Custom::record()` (line 92 of `TSync_Custom.hpp`) calls:

```cpp
ffts_cross_core_sync(PIPE_FIX, _getFFTSMsg(CV_CORE_SYNC, flag_id));
```

`flag_id` is a **runtime `uint16_t` member** of the struct.  The bisheng
compiler requires the second argument of `ffts_cross_core_sync` to be a
**compile-time literal**.  When a non-literal is passed, bisheng does not
emit an error; instead it silently generates incorrect code — the sync fires
with a wrong message value, causing the vector side to read the wrong data.

### Minimum reproducer

```cpp
#include <pto/npu/a2a3/custom/TSync_Custom.hpp>
#include <pto/npu/a2a3/custom/TSyncCVID.hpp>
using namespace pto;

extern "C" __global__ AICORE void reproducer_tsync_custom(
    __gm__ float* out, __gm__ uint8_t* ffts_addr)
{
#ifdef __DAV_C220_CUBE__
    set_ffts_base_addr((uint64_t)ffts_addr);
    // flag_id = 0 via aggregate initialiser, but it is a runtime uint16_t member
    TSync_Custom<SyncOpType::TSTORE_C2GM, SyncOpType::TLOAD> sync{0};
    pipe_barrier(PIPE_MTE3);
    sync.record();   // ← bisheng silent wrong-code: flag_id is not a literal
#endif
#ifdef __DAV_C220_VEC__
    set_ffts_base_addr((uint64_t)ffts_addr);
    TSync_Custom<SyncOpType::TSTORE_C2GM, SyncOpType::TLOAD> sync{0};
    sync.wait();     // wait_flag_dev(flag_id) — this part is fine
    out[get_subblockid()] = 1.0f;
#endif
}
```

**Expected:** compiles cleanly, vector waits for cube signal, output is `1.0f`.

**Observed:** compiles cleanly (no warning, no error), but the cube sync fires
with a wrong FFTS message.  The vector core does **not** wait; it reads stale
GM data.

In the full c2v sync kernel (`sync_c2v_kernel.cpp` ported to pto-isa), where
the cube writes `block_idx` values and the vector adds `subblockid`:

| Expected element value | Observed element value |
|---|---|
| `2 * block_idx + subblockid` (e.g. 47) | `block_idx + subblockid` (e.g. 24) |

The observed value matches what would be read if the vector proceeded before
the cube's `TSTORE` completed — confirming the sync signal was not received.

### Relevant source (TSync_Custom.hpp line 88–97)

```cpp
AICORE inline void record() const
{
    if constexpr (is_c2v) {
        ffts_cross_core_sync(PIPE_FIX, _getFFTSMsg(CV_CORE_SYNC, flag_id));
        //                                                         ^^^^^^^^
        //   flag_id is uint16_t member — not a compile-time literal
        //   bisheng requirement violated; wrong code generated silently
    } else {
        ffts_cross_core_sync(PIPE_MTE3, _getFFTSMsg(CV_CORE_SYNC, flag_id));
    }
}
```

---

## Bug 2 — `Event<Op::TMOV_A2V, Op::TADDS, false, EVENT_ID0>::Init()`: compile error

### Root cause

`Event::Init()` (line 107 of `TSync.hpp`) calls:

```cpp
ffts_cross_core_sync(srcPipe, getFFTSMsg(FFTS_MODE_VAL, CrossCoreId));
```

where `srcPipe` is defined as:

```cpp
static constexpr pipe_t srcPipe = GetPipeByOp<srcOp>();
```

`GetPipeByOp<>` returns a value from `opPipeList[]`, which is a
`constexpr pipe_t` but **not a literal** in the sense bisheng requires.
Bisheng enforces that the first argument of `ffts_cross_core_sync` be a
literal `pipe_t` value (e.g. the token `PIPE_FIX`), not an expression —
even a `constexpr` one.

### Minimum reproducer

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

extern "C" __global__ AICORE void reproducer_event(
    __gm__ uint8_t* ffts_addr)
{
#ifdef __DAV_C220_CUBE__
    set_ffts_base_addr((uint64_t)ffts_addr);
    Event<Op::TMOV_A2V, Op::TADDS, false, EVENT_ID0> ev;
    ev.Record<0>();   // ← compile error
#endif
#ifdef __DAV_C220_VEC__
    set_ffts_base_addr((uint64_t)ffts_addr);
    Event<Op::TMOV_A2V, Op::TADDS, false, EVENT_ID0> ev;
    ev.Wait<0>();
#endif
}
```

**Compile command:**
```
bisheng -fPIC -shared -xcce -O2 -std=c++17 --npu-arch=dav-2201 -DMEMORY_BASE \
  -I${PTO_LIB_PATH}/include -I${ASCEND_TOOLKIT_HOME}/include \
  reproducer_event.cpp -o reproducer_event.so
```

**Compile error:**
```
pto/npu/a2a3/TSync.hpp:113:13: error: the 1st parameter must be a type 'pipe_t'
            ffts_cross_core_sync(srcPipe, getFFTSMsg(FFTS_MODE_VAL, CrossCoreId));
            ^
```

### Relevant source (TSync.hpp lines 25–31 and 107–123)

```cpp
// srcPipe is constexpr but not a literal — bisheng rejects it as first arg
static constexpr pipe_t srcPipe = GetPipeByOp<srcOp>();

template <uint8_t CrossCoreId = 0xff>
PTO_INTERNAL Event &Init()
{
    if constexpr (IsCrossCore) {
        ffts_cross_core_sync(srcPipe, getFFTSMsg(FFTS_MODE_VAL, CrossCoreId));
        //                   ^^^^^^^
        //   computed constexpr pipe_t — not a literal; bisheng compile error
    }
    ...
}
```

---

## Why `TPipe::Producer` Works

`TPipe::Producer::record()` hard-codes literal `PIPE_FIX` and computes the
message from a compile-time template parameter `FlagID=0`:

```cpp
ffts_cross_core_sync(PIPE_FIX, getFFTSMsgCfg(CV_CORES_SYNC, FlagID));
//                   ^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//   literal pipe_t             FlagID=0 is a template param → constant-folds to 33
```

Both arguments fold to compile-time constants, satisfying bisheng's requirement.

---

## Workaround

Call the raw intrinsics directly with a pre-computed `static constexpr` message:

```cpp
// Compute the FFTS message entirely at compile time
static constexpr uint64_t kC2VSyncMsg =
    1u | (/* CV_CORE_SYNC= */ 2u << 4u) | (/* flag_id= */ 0u << 8u);  // = 0x21 = 33

// Cube side
pipe_barrier(PIPE_MTE3);                    // drain GM writes before signal
ffts_cross_core_sync(PIPE_FIX, kC2VSyncMsg); // both args are literals

// Vector side
wait_flag_dev(0);                           // literal flag_id
```

`kC2VSyncMsg` is a `static constexpr uint64_t` constant expression that
bisheng folds to the literal `33` at compile time, satisfying the requirement
for both arguments.

---

## Summary

| Wrapper | First arg (`pipe_t`) | Second arg (msg) | Result |
|---|---|---|---|
| `TSync_Custom::record()` | `PIPE_FIX` — literal ✓ | `_getFFTSMsg(..., flag_id)` — runtime member ✗ | **Wrong output, no error** |
| `Event::Init()` | `GetPipeByOp<srcOp>()` — computed constexpr ✗ | compile-time ✓ | **Compile error** |
| `TPipe::Producer::record()` | `PIPE_FIX` — literal ✓ | `getFFTSMsgCfg(..., FlagID=0)` — template param ✓ | Works |
| Raw `ffts_cross_core_sync` + `static constexpr` msg | `PIPE_FIX` — literal ✓ | `kC2VSyncMsg` — `static constexpr` ✓ | Works |

**Suggested fix:** in `TSync_Custom`, promote `flag_id` from a runtime member to a
template non-type parameter so that `_getFFTSMsg(CV_CORE_SYNC, FlagID)` can be
evaluated at compile time — analogous to how `TPipe` uses `FlagID` as a template
parameter.  In `Event::Init()`, replace `srcPipe` (computed constexpr) with the
literal token `PIPE_FIX` for cross-core events, or document that cross-core
`Event` is not supported by bisheng.
