# torch_emulation_pto

PyTorch CPU emulation of the five **PTO** kernels under `dynamic_bsnd/` (`chunk_cumsum`, `scaled_dot_kkt`, `wy_fast`, `chunk_h`, `chunk_o`). The code mirrors **data movement** (GM → UB/L1 → L0, `TLOAD` / `TSTORE` / `TEXTRACT`-style copies in `_memory.py`) as well as the math; see each module’s docstring.

## Emulation principles (buffering and PTO mapping)

- **Named SRAM roles** — Tensors tagged as UB, L1, L0A/L0B/L0C follow the same roles as in the C++ / PTO sources (`_memory.py` lists the op stand-ins).
- **Pre-allocate and reuse** — On-chip–style tiles are allocated **once at the start of each** ``*_fwd`` (before any sequence/head/chunk loop) and **reused** for every iteration; recurrent GM state (e.g. ``chunk_h``’s ``S``) is reset in place with ``zero_()`` where needed. That matches a fixed kernel tile budget instead of allocating inside the hot loop.
- **Explicit movement** — Loads, pads, and `TMOV`-style copies go through `_memory` helpers (`tload` / `tstore`, `tload_bsnd_rows`, `tfillpad_k_l1_tail_rows`, `tmov`, `tload_gm_fp32_dd_to_l1_half`, `tmov_l1_cc_gate_mask_from_l0c`, etc.) so the call graph lines up with the original PTO dataflow.
- **`gemm_v0`** — Cube matmul uses `textract_*` into **reused** L0A/L0B stripes plus a **reused** fp32 L0C buffer (`gemm_v0_accum_fp16(..., l0c_out=..., l0a_buf=..., l0b_buf=...)`), matching repeated `TEXTRACT` / accumulate behavior.

The goal is **readability and traceability to PTO**, not cycle-accurate async DMA (no `set_flag` / `wait_flag`).

## Import

From `examples/jit_cpp/chunk_gdn` (or with that directory on `PYTHONPATH`):

```python
from torch_emulation_pto import (
    chunk_cumsum_fwd,
    scaled_dot_kkt_fwd,
    wy_fast_fwd,
    chunk_h_fwd,
    chunk_o_fwd,
)
```

## Verify against CPU references

The verifier compares emulation to the same CPU **`ref_*`** math as `dynamic_bsnd/verify_dynamic_bsnd.py`, implemented in `torch_emulation_pto/cpu_refs.py` (pure PyTorch). **No NPU** — everything runs on the host. The verifier **does not** import `verify_dynamic_bsnd` or `dynamic_kernel_libs` (those trigger PTO kernel JIT and can block for a long time).

```bash
cd examples/jit_cpp/chunk_gdn
python torch_emulation_pto/verify_torch_emulation_pto.py
python torch_emulation_pto/verify_torch_emulation_pto.py --quick
python torch_emulation_pto/verify_torch_emulation_pto.py --smoke
python torch_emulation_pto/verify_torch_emulation_pto.py --quick --timeout 60
```

| Flag | Meaning |
|------|---------|
| `--seed N` | Base RNG seed (default `42`; each case adds an offset) |
| `--quick` | Three representative shapes only |
| `--smoke` | Tiny end-to-end finite-run check only (skips the full `ref_*` suite) |
| `--timeout SEC` | Max wall seconds **per test case** (Unix `SIGALRM`; default 120 with `--quick`, 600 otherwise; `0` disables) |

For each non-smoke run, every case reports **e2e** (full pipeline vs refs) and **iso** (each stage fed reference upstreams to localize mismatches).

Pass criteria match `verify_dynamic_bsnd`: strict allclose with `atol=1e-5`, `rtol=1e-2`, or a statistical fallback (RMSE vs mean \|ref\|, R²) when a few outliers break pointwise bounds.
