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

The two versions here keep wrapper-style usage but use local compatibility wrappers
for the cross-core signal path where upstream wrappers are currently unreliable in
this environment.

---

### Version 1 — TSYNC-style (`sync_c2v_tsync.cpp`)

Uses local **`MyTSync<0>`** from `MyTSync.hpp` (TSYNC-like `record()` / `wait()` API).

```cpp
MyTSync<0> c2v_sync;

// Cube: after writing gm_output, drain MTE3 then signal
pipe_barrier(PIPE_MTE3);
c2v_sync.record();   // → ffts_cross_core_sync(PIPE_MTE3, kMsg)

// Vector: before reading gm_output
c2v_sync.wait();     // → wait_flag_dev(0)
```

Why local wrapper: in this snapshot/toolchain, `TSync_Custom` path and related
headers have integration/functionality gaps (documented in `issue_report/`), while
`MyTSync` keeps the call style and matches validated behavior.

---

### Version 2 — TPUSH/TPOP-style (`sync_c2v_tpushpop.cpp`)

Uses local **`MyC2VPipe<0>`** from `MyC2VPipe.hpp` (TPUSH/TPOP-like producer/consumer API).

```cpp
// Cube (TPUSH producer side)
pipe_barrier(PIPE_MTE3);
MyC2VPipe<0>::Producer prod;
prod.record();   // → ffts_cross_core_sync(PIPE_MTE3, packed_msg)

// Vector (TPOP consumer side)
MyC2VPipe<0>::Consumer cons;
cons.wait();     // → wait_flag_dev(0)
```

Why local wrapper: native `TPipe::Producer::record()` C2V path uses `PIPE_FIX`
in this snapshot; for this kernel shape it yields wrong numeric result (`...46`
tail). `MyC2VPipe` keeps wrapper style while using the validated `PIPE_MTE3` producer signal.

Data path in both versions is fully wrapper-based:
- Cube: `TLOAD(MatTile)` -> `TSTORE(Global, MatTile)`
- Vec: `TLOAD(VecTile)` -> `TADDS` -> `TSTORE(Global, VecTile)`

---

### Usage

```bash
export PTO_LIB_PATH=<path-to-pto-isa>   # tested against https://gitcode.com/cann/pto-isa/commit/64aedbb0a3214b4ffdf0415f65d8febf23273f40?ref=master

# Option A — compile manually
bash compile.sh

# Option B — compile + run + verify in one step
cd examples/jit_cpp/c2v_sync
python run_sync_c2v.py
```

`run_sync_c2v.py` compiles each version on-the-fly, runs the kernel, compares
against the reference result (`indices // N`), prints `correct: True`, then
removes the temporary `.so`.

For detailed repros, scripts, and logs of upstream wrapper issues, see
`examples/jit_cpp/c2v_sync/issue_report/`.
