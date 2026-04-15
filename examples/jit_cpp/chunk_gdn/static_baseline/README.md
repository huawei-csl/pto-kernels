# Static PTO baseline (no TileLang JIT)

Self-contained PTO kernels copied from TileLang-generated sources under `../tilelang_codegen/kernels/`, compiled with `bisheng` and tested against PyTorch references on NPU. **No Python TileLang import** is required at runtime—only `torch` + `ctypes` + the compiled `.so` files.

## Shared pieces

| File | Role |
|------|------|
| `include/common.h` | Copy of `tilelang-ascend/src/tl_templates/pto/common.h` with **`namespace tl::ascend_pto` → `chunk_gdn_pto`**. |
| `pto_static_common.py` | Shared `bisheng` flags: local `include/`, then **`$PTO_LIB_PATH/include` before CANN** (same as other `jit_cpp` examples; defaults to CANN via `ASCEND_TOOLKIT_HOME`). Recompiles when a `*_kernel.cpp` **mtime** changes. |
| `static_kernel_libs.py` | Loads compiled shared libraries (ctypes); reloads when `*.cpp` sources change. |
| `sync_from_tilelang_kernels.py` | Copies `../tilelang_codegen/kernels/opt_gdn_*.cpp` into `*_kernel.cpp` here (include + namespace transforms). Run after regenerating dumps in `tilelang_codegen`. |
| `bench_static_gdn.py` | NPU benchmark for the static kernels (same shape and TFLOPS model as `../tilelang_codegen/bench_tilelang_gdn.py`). Uses a **single** `torch.npu.current_stream()._as_parameter_` for all launches so stream lookup is **not** inside the timed region. |
| `../gdn_bench_common.py` | Shared `do_bench` / op-count helpers used by both TileLang and static benchmarks. |

## Shapes

Kernels are specialized for the same configuration as `bench_tilelang_gdn.py` / tilelang-ascend GDN README:

**`B=16`, `H=16`, `L=16384`, `DK=128`, `DV=128`, `C=128`** (and `chunk_num=128` where applicable).

After editing TileLang drivers, run `../tilelang_codegen/scripts/dump_all_kernels.sh`, then **`python3 sync_from_tilelang_kernels.py`** from this directory.

## Kernels (`.cpp` → `compiled_lib/*.so` → Python test)

| Kernel source | Test driver | Reference tolerance |
|---------------|-------------|---------------------|
| `chunk_cumsum_kernel.cpp` | `run_chunk_cumsum_static.py` | rtol/atol `1e-5` |
| `chunk_h_kernel.cpp` | `run_chunk_h_static.py` | `1e-5` |
| `chunk_o_kernel.cpp` | `run_chunk_o_static.py` | `1e-5` |
| `scaled_dot_kkt_kernel.cpp` | `run_scaled_dot_kkt_static.py` | `1e-3` |
| `wy_fast_kernel.cpp` | `run_wy_fast_static.py` | `1e-5` |

Run per-kernel tests:

```bash
cd static_baseline
export ASCEND_HOME_PATH=/path/to/cann   # or ASCEND_TOOLKIT_HOME
# optional: export PTO_LIB_PATH=/path/to/cann
python3 run_all_static_kernels.py
```

`run_all_static_kernels.py` runs each `run_*_static.py` in a **subprocess** so NPU/RNG state matches isolated runs (in-process sequential imports were unreliable for later tests).

Or run a single test, e.g. `python3 run_chunk_o_static.py`.

### End-to-end GDN (chained static kernels + solve\_tril)

`gdn_chain_e2e_static.py` runs: `cumsum → KKT → solve_tril → wy_fast → chunk_h → chunk_o` with the same fixed shapes as the static kernels.

- **solve\_tril** (C=128): prefers `pto_tri_inv_rec_unroll` from the `pto_kernels` package; otherwise CPU `torch.linalg.inv(I + A)` with strict-lower `A`.

```bash
python3 gdn_chain_e2e_static.py
```

## Performance benchmark (static vs TileLang JIT)

From this directory (same device as TileLang benchmark):

```bash
python3 bench_static_gdn.py
```

Representative run on the same NPU session as `../tilelang_codegen/bench_tilelang_gdn.py`:

| Kernel | TileLang JIT latency (ms) | Static PTO latency (ms) |
| :-- | --: | --: |
| chunk_cumsum | 1.39 | 1.28 |
| chunk_scaled_dot_kkt | 9.70 | 9.73 |
| wy_fast | 9.76 | 9.77 |
| chunk_h | 9.01 | 9.12 |
| chunk_o | 11.71 | 11.63 |
| **total** | **41.58** | **41.53** |

Totals agree within measurement noise—the static `.so` is the same PTO ISA as the TileLang JIT path, only the launch wrapper differs.

## Environment

- `ASCEND_TOOLKIT_HOME` or `ASCEND_HOME_PATH` — CANN prefix (used as the default `PTO_LIB_PATH` when unset).
- `PTO_LIB_PATH` — prefix whose `include/` supplies PTO headers for `bisheng` (listed before CANN `-I`). Defaults to the same value as your CANN home when unset.

## Regenerating `*_kernel.cpp` from TileLang

1. In `../tilelang_codegen`, run `./scripts/dump_all_kernels.sh` (requires `TL_ROOT`, `ASCEND_HOME_PATH`, NPU).
2. In **this** directory: `python3 sync_from_tilelang_kernels.py`
3. Apply manual steps only if upstream codegen changes format:
   - `#include "tl_templates/pto/common.h"` → `#include "common.h"` (the sync script does this)
   - Drop duplicate `#include <pto/pto-inst.hpp>` if present
   - `tl::ascend_pto::` → `chunk_gdn_pto::` (the sync script does this)

Refresh `include/common.h` from upstream when needed and re-apply the namespace rename.

Optional: `PTO_STATIC_EXTRA_FLAGS` — extra flags appended to `bisheng` (space-separated).
