# Dynamic BSND `chunk_h` — group-value heads (`H ≠ Hg`)

PTO kernel matching Triton FLA semantics for gated delta-rule hidden-state recurrence when **query/value heads `H`** exceed **shared key heads `Hg`** (e.g. GQA). Same runtime dynamics as ``dynamic_bsnd/chunk_h_kernel.cpp`` for batch and sequence layout (`cu_seqlens`), but ``K`` uses BSND stride ``Hg·D`` and maps ``head_g = head / (H/Hg)``.

## Build / load

Uses ``bisheng`` like other ``examples/jit_cpp`` samples (via ``pto_dynamic_common.compile_pto_kernel``). Macros:

- ``GDN_H`` — value head count ``H``
- ``GDN_HG`` — key head count ``Hg`` (default ``GDN_H`` if omitted)
- ``GDN_D``, ``GDN_C`` — hidden size and chunk size

Shared objects are cached under ``compiled_lib/chunk_h_bsnd_groupvalue_H{H}_Hg{Hg}_D{D}_C{C}.so``.

## Verification (NPU)

From ``chunk_gdn/dynamic_bsnd_groupvalue``:

```bash
export ASCEND_TOOLKIT_HOME=/path/to/Ascend/cann  # or ASCEND_HOME_PATH
export PTO_LIB_PATH=/path/to/pto-isa/include/..
python3 verify_dynamic_bsnd_groupvalue.py --device npu:7 --H-list 16,32,48,64
python3 verify_dynamic_bsnd_groupvalue.py --device npu:7 --quick   # one fixed-length smoke case per H
```

Expectations: **same case list** as ``dynamic_bsnd/verify_dynamic_bsnd.py`` lines 222–280 (fixed-length, varlen, tails, ladders). Gates follow chunk-local cumulative sums like the upstream verifier (``logsigmoid`` + chunk cumsum); keys are L2-normalized like ``verify_dynamic_bsnd``. Checks compare ``h_states`` and ``v_new`` against a CPU fp32 reference with the standard rtol/atol/statistical fallback used there.

## Benchmark

```bash
python3 bench_dynamic_bsnd_groupvalue.py
# Example:
GDN_BENCH_H=32 GDN_BENCH_HG=16 python3 bench_dynamic_bsnd_groupvalue.py
```

Reports PTO ``chunk_h`` latency and Triton FLA vendor timing when ``triton_baseline`` imports cleanly.

## Implementation notes

- Vec-stage GM loads for ``K`` use ``(token·Hg + head_g)·D`` row indexing with stride ``Hg·D`` (see ``chunk_h_kernel.cpp``).
- UB packing uses a fixed leading slack matching the legacy ``GDN_H=16`` kernel so large compile-time ``H`` does not exceed the vector UB budget (~192 KiB on 910B2).
