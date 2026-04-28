# Dynamic BSND — GQA group-value heads (`H ≠ Hg`)

PTO kernels when **value heads `H`** exceed shared **key heads `Hg`** (`head_g = head // (H // Hg)`, same as FLA/Triton). Layout: `k` / `q` are `[B,T,Hg,D]`; `v`, `w`, `u`, `o`, gates, and `A` use **H** along the head axis.

| Kernel | C++ | Role |
|--------|-----|------|
| `scaled_dot_kkt` | `scaled_dot_kkt_kernel.cpp` | Gated intra-chunk `KKᵀ` |
| `chunk_h` | `chunk_h_kernel.cpp` | Recurrent chunk state |
| `wy_fast` | `wy_fast_kernel.cpp` | WY recompute `W`, `U` |
| `chunk_o` | `chunk_o_kernel.cpp` | Chunk attention output |

Build: `bisheng` via `pto_dynamic_common.compile_pto_kernel` with `GDN_H`, `GDN_HG` (default `GDN_H`), `GDN_D`, `GDN_C`. Cached `*.so` names: `*_bsnd_groupvalue_H{H}_Hg{Hg}_D{D}_C{C}.so`.

---

## Verify (NPU)

```bash
cd /path/to/pto-kernels/examples/jit_cpp/chunk_gdn/dynamic_bsnd_groupvalue
export ASCEND_TOOLKIT_HOME=/path/to/Ascend/cann    # or ASCEND_HOME_PATH
export PTO_LIB_PATH=/path/to/pto-isa/include/..    # parent of pto headers
export GDN_NPU_DEVICE=npu:7

# Full case list (~30 shapes × stages × H); long-running
python3 verify_dynamic_bsnd_groupvalue.py --device npu:7 --H-list 16,32,48,64

# One case (T=128), all stages
python3 verify_dynamic_bsnd_groupvalue.py --device npu:7 --quick --H-list 32

# Only selected stages (see --help)
python3 verify_dynamic_bsnd_groupvalue.py --device npu:7 --stage kkt,chunk_h --quick
```

Use `--hg N` for key-head count (default **16**, or **`GDN_HG`**).

---

## Benchmark (PTO vs FLA Triton)

Default workload matches `dynamic_bsnd/bench_dynamic_bsnd.py`: `N_seq=16`, `L_seg=16384`, `T=262144`, `D=128`, **PTO `C=128`**.

```bash
cd /path/to/.../dynamic_bsnd_groupvalue
export ASCEND_TOOLKIT_HOME=...
export GDN_NPU_DEVICE=npu:7

# All stages, H ∈ {16,32,48,64}, Hg=16
python3 bench_dynamic_bsnd_groupvalue.py

# Single configuration
python3 bench_dynamic_bsnd_groupvalue.py --heads 32 --hg 16 --stage kkt,chunk_h,chunk_o,wy_fast
```

**Triton chunk tiles:** `chunk_scaled_dot_kkt_fwd` is benchmarked at **`BT=64`** by default (`GDN_TRITON_KKT_CHUNK`); optional **`BT=128`** is attempted if `GDN_TRITON_KKT_TRY128` is non-zero and compile succeeds. `chunk_fwd_o` uses `GDN_TRITON_CHUNK_O_CHUNK` (default **64**). Ratio columns are **`ms_triton / ms_pto`** (**``> 1`` ⇒ PTO faster**).

Read **`../dynamic_bsnd/README.md` → [PTO vs Triton chunk tile](../dynamic_bsnd/README.md#pto-vs-triton-chunk-tile)** before interpreting cross-tile comparisons.

---

## Measured latency (910B2, `npu:7`, `cube_core_num=24`)

Recorded **2026-04-28** on this tree. **`T=262144`**, **`Hg=16`**, PTO **`C=128`**.

### `scaled_dot_kkt`

Triton primary **`BT=64`**; optional **`BT=128`** omitted when MLIR compile fails.

| `H` | PTO `C=128` (ms) | Triton `BT=64` (ms) | `T64/PTO` | Triton `BT=128` (ms) | `T128/PTO` |
| --: | --: | --: | --: | --: | --: |
| 16 | 4.31 | 4.08 | 0.95 | — | — |
| 32 | 7.40 | 7.50 | 1.01 | — | — |
| 48 | 11.87 | 11.02 | 0.93 | — | — |
| 64 | 17.32 | 14.54 | 0.84 | — | — |

### `chunk_h` / `chunk_o` / `wy_fast`

| `H` | PTO chunk_h (ms) | Triton chunk_h (ms) | `T/PTO` | PTO chunk_o (ms) | Triton chunk_o `BT=64` (ms) | `T/PTO` | PTO wy_fast (ms) | Triton wy_fast (ms) | `T/PTO` |
| --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| 16 | 9.08 | 15.61 | 1.72 | 9.59 | 16.13 | 1.68 | 6.02 | 11.92 | 1.98 |
| 32 | 17.83 | 30.54 | 1.71 | 19.49 | 31.50 | 1.62 | 12.28 | 23.37 | 1.90 |
| 48 | 25.09 | 45.47 | 1.81 | 30.25 | 46.63 | 1.54 | 16.69 | 34.83 | 2.09 |
| 64 | 38.04 | 60.62 | 1.59 | 38.97 | — | — | 22.48 | 46.30 | 2.06 |

`chunk_o` Triton at **`H=64`** failed (**507015**) on the host used; PTO succeeded. Re-run **`bench_dynamic_bsnd_groupvalue.py`** after driver updates.

---

## Implementation notes

- Cube GM loads for **Q/K** use `(token·Hg + head_g)·D` and stride **`Hg·D`**; **V** and value-strided outputs use **`H·D`**.
- `chunk_h` Vec UB slack is fixed like legacy `GDN_H=16` so large **`H`** stays within UB budget on 910B2.
