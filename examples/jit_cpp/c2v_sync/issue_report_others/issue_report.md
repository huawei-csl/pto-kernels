# pto-isa TSync Issues — Additional Bugs and Analysis

**pto-isa commit under test:**
https://gitcode.com/cann/pto-isa/commit/64aedbb0a3214b4ffdf0415f65d8febf23273f40?ref=master

For Bug 1 (`TSync_Custom::record()` wrong output) and its reproducer code, see
[`../issue_report_tsync/`](../issue_report_tsync/).

---

## Bug 2 — `Event<Op::TMOV_A2V, ...>::Init()`: compile error

### Root cause

`Event::Init()` (`pto/npu/a2a3/TSync.hpp`, line 113) calls:

```cpp
ffts_cross_core_sync(srcPipe, getFFTSMsg(FFTS_MODE_VAL, CrossCoreId));
```

where `srcPipe` is defined as:

```cpp
static constexpr pipe_t srcPipe = GetPipeByOp<srcOp>();
```

`GetPipeByOp<>()` returns a value from `opPipeList[]`.  The result is a
`constexpr pipe_t`, but not a **literal** in the sense bisheng requires —
it is a computed expression, not a bare token like `PIPE_FIX`.  Bisheng
enforces that the first argument of `ffts_cross_core_sync` be a literal
`pipe_t` value, and rejects `srcPipe` at compile time.

### Minimum reproducer

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

extern "C" __global__ AICORE void repro_event(
    __gm__ uint8_t* ffts_addr)
{
#ifdef __DAV_C220_CUBE__
    set_ffts_base_addr((uint64_t)ffts_addr);
    Event<Op::TMOV_A2V, Op::TADDS, false, EVENT_ID0> ev;
    ev.Record<0>();   // ← compile error here
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
  repro_event.cpp -o repro_event.so
```

**Compile error:**
```
pto/npu/a2a3/TSync.hpp:113:13: error: the 1st parameter must be a type 'pipe_t'
            ffts_cross_core_sync(srcPipe, getFFTSMsg(FFTS_MODE_VAL, CrossCoreId));
            ^
```

### Relevant source (`TSync.hpp` lines 25–31 and 107–123)

```cpp
// srcPipe is a computed constexpr — not a literal token
static constexpr pipe_t srcPipe = GetPipeByOp<srcOp>();

template <uint8_t CrossCoreId = 0xff>
PTO_INTERNAL Event &Init()
{
    if constexpr (IsCrossCore) {
        ffts_cross_core_sync(srcPipe, getFFTSMsg(FFTS_MODE_VAL, CrossCoreId));
        //                   ^^^^^^^  bisheng rejects: not a literal pipe_t
    }
    ...
}
```

---

## Why `TPipe::Producer` works

`TPipe::Producer::record()` hard-codes literal `PIPE_FIX` and computes the
message from a compile-time template parameter `FlagID`:

```cpp
ffts_cross_core_sync(PIPE_FIX, getFFTSMsgCfg(CV_CORES_SYNC, FlagID));
//                   ^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//   literal pipe_t token       FlagID=0 is a template param →
//                               expression folds to the literal 33 (0x21)
```

Both arguments are compile-time literals, satisfying bisheng's requirement.

---

## Bisheng strict-literal requirement summary

| Sync mechanism | 1st arg (`pipe_t`) | 2nd arg (msg) | Result |
|---|---|---|---|
| `TSync_Custom::record()` | `PIPE_FIX` — literal ✓ | `_getFFTSMsg(..., flag_id)` — runtime `uint16_t` ✗ | **Wrong output (silent)** |
| `Event::Init()` | `GetPipeByOp<srcOp>()` — computed constexpr ✗ | compile-time ✓ | **Compile error** |
| `TPipe::Producer::record()` | `PIPE_FIX` — literal ✓ | `getFFTSMsgCfg(..., FlagID=0)` — template param ✓ | Works |
| `MyTSync::record()` (workaround) | `PIPE_FIX` — literal ✓ | `kMsg` — `static constexpr` from template params ✓ | Works |

---

## Suggested fixes

**`TSync_Custom`:** promote `flag_id` from a runtime `uint16_t` member to a
template non-type parameter, so `_getFFTSMsg(CV_CORE_SYNC, FlagID)` is
evaluated at compile time — identical to how `TPipe` uses `FlagID`.

```cpp
// Before (broken)
struct TSync_Custom { uint16_t flag_id; ... };
TSync_Custom<TSTORE_C2GM, TLOAD> sync{0};

// After (fix)
template <SyncOpType ProducerOp, SyncOpType ConsumerOp, uint8_t FlagID>
struct TSync_Custom { ... };
TSync_Custom<TSTORE_C2GM, TLOAD, 0> sync;
```

**`Event::Init()`:** replace the computed `srcPipe` with the literal `PIPE_FIX`
for cross-core events (cross-core events always emit on `PIPE_FIX`), or
document that cross-core `Event` is not supported by bisheng.
