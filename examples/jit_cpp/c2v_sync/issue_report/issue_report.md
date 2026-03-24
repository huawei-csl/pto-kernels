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

### 1) TSYNC path has integration gaps on this platform/toolchain

- Including `pto/npu/a2a3/TSync.hpp` in this example triggers compile errors in current environment because `event.hpp` references `PIPE_FIX` that is not declared in this build surface.
- Even bypassing include issues, the generic TSYNC/Event/Custom sync path is not directly usable as a drop-in for this case without local adaptation.

### 2) TPipe C2V producer signal does not match the validated C2V behavior

- `TPipe<...>::Producer::record()` for C2V uses:
  - `ffts_cross_core_sync(PIPE_FIX, getFFTSMsgCfg(...))`
- The reference kernel uses:
  - `ffts_cross_core_sync(PIPE_MTE3, config)`
- For this C2V kernel shape on this platform, `PIPE_FIX` producer signaling led to stale-read-like behavior (last segment stays `46`), while `PIPE_MTE3` matches reference and passes.

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
  - Uses high-level `TSync_Custom<SyncOpType::TSTORE_C2GM, SyncOpType::TLOAD>` in a full C2V compute path.
  - Reproduces the current TSYNC wrapper issue.
- `repro_tpipe_issue.cpp`
  - Shows that C2V via `TPipe::Producer::record()` (PIPE_FIX path) differs from validated `PIPE_MTE3` producer sync.
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
Observed in failing wrapper path: tail remains `...46`, indicating missing `+1` effect on one sub-core path due to sync mismatch.

## Current status

- `sync_c2v_tsync.cpp` using `MyTSync` -> correct.
- `sync_c2v_tpushpop.cpp` using `MyC2VPipe` -> correct.

Both preserve wrapper-style syntax while matching the working reference synchronization behavior.
