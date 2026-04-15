# Static PTO baseline (no TileLang JIT)

Self-contained PTO kernels extracted from TileLang-generated sources under `../tilelang_codegen/`, compiled with `bisheng` and tested against PyTorch references on NPU.

## Shared pieces

| File | Role |
|------|------|
| `include/common.h` | Copy of `tilelang-ascend/src/tl_templates/pto/common.h` with **`namespace tl::ascend_pto` → `chunk_gdn_pto`**. |
| `pto_static_common.py` | Shared `bisheng` flags: local `include/`, then **`$PTO_LIB_PATH/include` before CANN** (same as other `jit_cpp` examples; defaults to CANN via `ASCEND_TOOLKIT_HOME`). |

## Kernels (`.cpp` → `compiled_lib/*.so` → Python test)

All use the same fixed shape as the TileLang dumps: **`B=2`, `H=16`, `L=16384`, `DK=128`, `DV=128`, `C=128`** (and `chunk_num=128` where applicable).

| Kernel source | Test driver | Reference tolerance (matches TileLang tests) |
|---------------|---------------|-----------------------------------------------|
| `chunk_cumsum_kernel.cpp` | `run_chunk_cumsum_static.py` | rtol/atol `1e-5` |
| `chunk_h_kernel.cpp` | `run_chunk_h_static.py` | `1e-5` |
| `chunk_o_kernel.cpp` | `run_chunk_o_static.py` | `1e-5` |
| `scaled_dot_kkt_kernel.cpp` | `run_scaled_dot_kkt_static.py` | `1e-3` (same as `opt_gdn_chunk_scaled_dot_kkt.py`) |
| `wy_fast_kernel.cpp` | `run_wy_fast_static.py` | `1e-5` |

Run per-kernel tests:

```bash
cd static_baseline
export ASCEND_HOME_PATH=/path/to/cann   # or ASCEND_TOOLKIT_HOME
# optional: export PTO_LIB_PATH=/path/to/cann   # default; set if PTO headers live elsewhere
python3 run_all_static_kernels.py
```

Or run a single test, e.g. `python3 run_chunk_o_static.py`.

### End-to-end GDN (chained static kernels + solve\_tril)

`gdn_chain_e2e_static.py` runs the same pipeline as `tilelang-ascend/examples/linear_attention_and_rnn/opt_gdn_full.py`:

`cumsum → KKT → solve_tril → wy_fast → chunk_h → chunk_o`

- Shapes are fixed to the extracted kernels: `B=2`, `H=16`, `L=16384`, `DK=DV=C=128`.
- **solve\_tril** (C=128): prefers `pto_tri_inv_rec_unroll` from the `pto_kernels` package (same math as `kernel_tri_inv_rec_unroll.cpp` / `test_tri_inv_rec_unroll.py`: invert `I + U` with `U = A^T` strict upper, then transpose). If `pto_kernels` is not importable, falls back to CPU `torch.linalg.inv(I + A)` with `A` forced to strict lower via `torch.tril(..., -1)`.
- Asserts against **`ref_seq_gdn`** from `opt_gdn_full.py` at `rtol/atol = 1e-3`.

```bash
python3 gdn_chain_e2e_static.py
```

To use the PTO tri-inv kernel, install/build the `pto-kernels` Python extension so `from pto_kernels import pto_tri_inv_rec_unroll` works (this repo adds `../../../python` to `sys.path` automatically when present).

## Environment

- `ASCEND_TOOLKIT_HOME` or `ASCEND_HOME_PATH` — CANN prefix (used as the default `PTO_LIB_PATH` when unset).
- `PTO_LIB_PATH` — prefix whose `include/` supplies PTO headers for `bisheng` (listed before CANN `-I`). Defaults to the same value as your CANN home when unset.

## Regenerating `*_kernel.cpp` from TileLang

From `../tilelang_codegen/kernels/opt_gdn_*.cpp`:

1. Copy into the matching `*_kernel.cpp` name in this directory.
2. `#include "tl_templates/pto/common.h"` → `#include "common.h"`.
3. Remove a duplicate `#include <pto/pto-inst.hpp>` if present.
4. `tl::ascend_pto::` → `chunk_gdn_pto::` (must match `include/common.h`).

Refresh `include/common.h` from upstream when needed and re-apply the namespace rename.

Optional: `PTO_STATIC_EXTRA_FLAGS` — extra flags appended to `bisheng` (space-separated).
