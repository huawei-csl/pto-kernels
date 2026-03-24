# C2V sync issue report (TSYNC / TPipe)

**pto-isa commit under test:**  
https://gitcode.com/cann/pto-isa/commit/64aedbb0a3214b4ffdf0415f65d8febf23273f40?ref=master

## Background

We ported `npu_kernels/c2v_sync_cce/sync_c2v_kernel.cpp` to wrapper-based versions:

- `sync_c2v_tsync.cpp`
- `sync_c2v_tpushpop.cpp`

Reference kernel behavior is correct (`correct: True`).  
During wrapper migration, both versions initially produced wrong output (`...46` tail instead of `...47`).

## What failed

### 1) TSYNC path has integration gaps and functional issue on this platform/toolchain

- Header/API integration gaps in current snapshot:
  - `TSync.hpp` include path pulls `event.hpp`, which references `PIPE_FIX` not exposed in this build surface.
  - `TSync_Custom.hpp` references missing symbols (`CV_CORE_SYNC`, `_getFFTSMsg`) in this snapshot.
- Functional issue reproduced in full wrapper compute path:
  - `TSync_Custom<..., ...>::record()` C2V path produces wrong numeric result (`...46` tail vs expected `...47`) on this kernel shape.

### 2) TPipe C2V producer signal does not match the validated C2V behavior

- `TPipe<...>::Producer::record()` for C2V uses:
  - `ffts_cross_core_sync(PIPE_FIX, getFFTSMsgCfg(...))`
- The reference kernel uses:
  - `ffts_cross_core_sync(PIPE_MTE3, config)`
- In full wrapper compute repro, native `TPipe::Producer::record()` gives wrong numeric result (`...46` tail), while `PIPE_MTE3` producer signal matches reference and passes.

## Why custom wrappers are needed now

To keep wrapper-style callsites while preserving validated behavior:

- `MyTSync.hpp`
  - Provides `record()/wait()` API like TSYNC usage.
  - Uses compile-time `FlagID` and emits producer sync on `PIPE_MTE3`.
- `MyC2VPipe.hpp`
  - Provides `Producer::record()` / `Consumer::wait()` API like TPUSH/TPOP usage.
  - Keeps message packing from TPipe (`getFFTSMsgCfg`) but emits producer sync on `PIPE_MTE3`.

These wrappers are temporary compatibility/workaround layers until upstream TSYNC/TPipe behavior and include surface fully cover this C2V scenario.

## Reproducer files in this folder

- `repro_tsync_issue.cpp`
  - Uses wrapper data path (`TLOAD/TSTORE/TADDS`) and high-level
    `TSync_Custom<SyncOpType::TSTORE_C2GM, SyncOpType::TLOAD>` for sync.
  - Reproduces TSYNC wrapper issue with real numeric mismatch.
- `repro_tpipe_issue.cpp`
  - Uses wrapper data path (`TLOAD/TSTORE/TADDS`) and compares:
    - native `TPipe::Producer::record()` (PIPE_FIX path)
    - workaround `PIPE_MTE3` producer record path
- `compile.sh`
  - Builds all repro shared libraries.
- `run_repro.py`
  - Launches each repro kernel and reports runtime launch status.

## Compile and test scripts

From `pto-kernels/examples/jit_cpp/c2v_sync/issue_report`:

```bash
export ASCEND_TOOLKIT_HOME=<path-to-cann>
export PTO_LIB_PATH=<path-to-pto-isa-root>

chmod +x ./compile.sh
./compile.sh
python ./run_repro.py
```

Artifacts produced by `compile.sh`:

- `repro_tsync_issue_lib.so`
- `repro_tpipe_native_lib.so` (uses upstream `TPipe::Producer::record()`)
- `repro_tpipe_workaround_lib.so` (uses workaround `PIPE_MTE3` producer sync)

## Expected vs observed symptom

For `N=16384`, `SUB_BLOCK_DIM=2`, expected tail is from subblock `1` (`...47`).  
Observed in failing wrapper paths (`TSYNC_Custom` repro and native `TPipe` repro): tail remains `...46`, indicating missing `+1` effect on one sub-core path due to sync mismatch.

Observed in workaround path (`TPipe` with custom `PIPE_MTE3` producer record): tail is `...47` (correct).

## Current status

- `sync_c2v_tsync.cpp` using `MyTSync` -> correct.
- `sync_c2v_tpushpop.cpp` using `MyC2VPipe` -> correct.
- `issue_report/run_repro.py` shows:
  - `TSYNC repro`: WRONG
  - `TPipe repro (native)`: WRONG
  - `TPipe repro (workaround)`: CORRECT

Both preserve wrapper-style syntax while matching the working reference synchronization behavior.
