## C2V Sync — Cube-to-Vector synchronization via pto-isa wrappers

### What the kernel does

Each cube core (AIC) copies a slice of `gm_input` through its L1 buffer into
`gm_output`, then signals the two paired vector sub-cores (AIV).  Each sub-core
reads its half of that slice, adds its own sub-block index, and writes the result
back to `gm_output`.

The cross-core handoff uses one FFTS flag (`flag_id = 0`, mode 2 = inner-group
AIC/AIV sync).  In the original raw-CCE version this was spelled out as:

```cpp
// Cube side
ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (0 << 8));

// Vector side
wait_flag_dev(0);
```

The two versions here replace those two calls with pto-isa wrappers.

---

### Version 1 — TSYNC (`sync_c2v_tsync.cpp`)

Uses **`TSync_Custom<SyncOpType::TSTORE_C2GM, SyncOpType::TLOAD>`** from
`<pto/npu/a2a3/custom/TSync_Custom.hpp>`.

```cpp
constexpr TSync_Custom<SyncOpType::TSTORE_C2GM, SyncOpType::TLOAD> c2v_sync = {0};

// Cube: after writing gm_output, drain MTE3 then signal
pipe_barrier(PIPE_MTE3);
c2v_sync.record();   // → ffts_cross_core_sync(PIPE_FIX, _getFFTSMsg(CV_CORE_SYNC, 0))

// Vector: before reading gm_output
c2v_sync.wait();     // → wait_flag_dev(0)
```

`TSync_Custom::record()` emits `PIPE_FIX` rather than `PIPE_MTE3`.  The explicit
`pipe_barrier(PIPE_MTE3)` drains the GM-write pipeline first, preserving the
same ordering guarantee as the original.

---

### Version 2 — TPUSH / TPOP (`sync_c2v_tpushpop.cpp`)

Uses **`TPipe<>::Producer::record()`** and **`TPipe<>::Consumer::wait()`** from
`<pto/npu/a2a3/TPush.hpp>` / `<pto/npu/a2a3/TPop.hpp>`.

```cpp
using C2VPipe = TPipe<0, FIFOType::GM_FIFO, 1, 1, ProdTile, ConsTile>;

// Cube (TPUSH producer side)
pipe_barrier(PIPE_MTE3);
C2VPipe::Producer prod;
prod.record();   // → ffts_cross_core_sync(PIPE_FIX, getFFTSMsgCfg(CV_CORES_SYNC, 0))

// Vector (TPOP consumer side)
C2VPipe::Consumer cons;
cons.wait();     // → wait_flag_dev(0)
```

The tile types (`ProdTile`, `ConsTile`) are declared only to satisfy the TPipe
template; the actual data movement still uses raw copy intrinsics because the
cube's L1-buffer path (`copy_cbuf_to_gm`) has no direct pto tile equivalent.

---

### Usage

```bash
export ASCEND_TOOLKIT_HOME=<path-to-cann>
export PTO_LIB_PATH=<path-to-pto-isa>   # root of this repo, or ASCEND_TOOLKIT_HOME

# Option A — compile manually
bash compile.sh

# Option B — compile + run + verify in one step
cd examples/jit_cpp/c2v_sync
python run_sync_c2v.py
```

`run_sync_c2v.py` compiles each version on-the-fly, runs the kernel, compares
against the reference result (`indices // N`), prints `correct: True`, then
removes the temporary `.so`.
