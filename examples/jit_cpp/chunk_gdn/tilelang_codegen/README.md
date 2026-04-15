# TileLang → PTO C++ codegen (chunk GDN kernels)

This directory is **self-contained**: drivers, the codegen patch, benchmarking, and dump scripts live under this tree. Regenerating the PTO-ISA C++ sources does not require importing kernel code from other repositories.

## Layout

| Path | Role |
|------|------|
| `patch_libgen.py` | Monkey-patches TileLang’s `LibraryGenerator.compile_lib` to write generated C++ before `bisheng`. |
| `kernels/` | TileLang drivers (`opt_gdn_*.py`) and the generated `opt_gdn_*.cpp` artifacts (same folder as each driver). |
| `scripts/dump_all_kernels.sh` | Runs every kernel driver to refresh the dumped `.cpp` files. |
| `bench_tilelang_gdn.py` | NPU performance benchmark (latency, approximate ops, TFLOPS) for the kernels in `kernels/`. Omits the separate `solve_tril` stage, which is not implemented here. |

## What gets generated

Running each driver under `kernels/` drives TileLang’s PTO backend (`target="pto"`), JIT-compiles the kernel, and **writes the generated C++** next to that driver.

| TileLang driver | Generated PTO C++ | Notes |
|-----------------|-------------------|--------|
| `kernels/opt_gdn_chunk_cumsum.py` | `kernels/opt_gdn_chunk_cumsum.cpp` | Chunk-wise prefix sum along `L` |
| `kernels/opt_gdn_chunk_h.py` | `kernels/opt_gdn_chunk_h.cpp` | Chunk hidden state / `new_v` / final state |
| `kernels/opt_gdn_chunk_o.py` | `kernels/opt_gdn_chunk_o.cpp` | Chunk output given hidden state |
| `kernels/opt_gdn_chunk_scaled_dot_kkt.py` | `kernels/opt_gdn_chunk_scaled_dot_kkt.cpp` | Scaled dot KKT-style lower-triangular block |
| `kernels/opt_gdn_wy_fast.py` | `kernels/opt_gdn_wy_fast.cpp` | WY-style fast path for `U` and `W` |

## Prerequisites

- **Python environment** with `tilelang` installed (the same package you use for Ascend/PTO JIT).
- **Environment variables** (read by TileLang and by `patch_libgen.py`):
  - `TL_ROOT` — root of the TileLang source tree that provides `3rdparty/pto-isa/include` and templates.
  - `ASCEND_HOME_PATH` — CANN install prefix (headers and `lib64` for linking the JIT `.so`).
- **Ascend NPU + `torch.npu`** — the drivers call `torch` on NPU so the JIT path runs end-to-end. Codegen happens inside `LibraryGenerator.compile_lib` when the kernel is first compiled.

## PTO C++ codegen steps (how this works)

1. **`patch_libgen.py`**  
   Replaces `LibraryGenerator.compile_lib` with a wrapper that, before invoking `bisheng`, writes `self.lib_code` to the chosen `*.cpp` file under `kernels/`.

2. **Driver scripts (`kernels/opt_gdn_*.py`)**  
   Each script prepends the parent directory to `sys.path` so it can import `patch_libgen`, applies the patch, calls `tilelang.disable_cache()`, declares the kernel with `@tilelang.jit(..., target="pto")`, and runs the small built-in numerical test, which triggers JIT and thus the dump.

3. **Artifacts**  
   After a successful run you get the generated source under `kernels/`. TileLang’s own `compile_lib` invokes `bisheng` with PTO headers from `$TL_ROOT/3rdparty/pto-isa/include` ahead of CANN defaults, matching upstream TileLang practice for PTO.

## Regenerating the `.cpp` files

From **this directory** (`tilelang_codegen`):

```bash
export TL_ROOT=/path/to/tilelang-ascend      # example
export ASCEND_HOME_PATH=/path/to/cann        # example

./scripts/dump_all_kernels.sh
```

Or run individual drivers:

```bash
python3 kernels/opt_gdn_chunk_cumsum.py
python3 kernels/opt_gdn_chunk_h.py
python3 kernels/opt_gdn_chunk_o.py
python3 kernels/opt_gdn_chunk_scaled_dot_kkt.py
python3 kernels/opt_gdn_wy_fast.py
```

## Performance benchmark

From this directory, with NPU visible and `torch_npu` available:

```bash
export GDN_TRI_INVERSE_NPU_DEVICE=npu:0   # optional, default shown

python3 bench_tilelang_gdn.py
```

This mirrors the methodology of `gdn-tri-inverse/profiling/bench_tilelang_full_gdn.py` (event timing, approximate floating-point op counts, TFLOPS). The benchmark pipeline **does not** include a triangular solve: the scaled KKT output is passed straight into `wy_fast`, consistent with only shipping the TileLang kernels in `kernels/`. It prints markdown-style tables to stdout (shape `C=128` only, matching the tilelang-ascend GDN README).

### Measured results (representative run)

Shape: `(B,H,L,DK,DV,C) = (16,16,16384,128,128,128)` — same as `tilelang-ascend/examples/linear_attention_and_rnn/README.md` GDN table. Latencies vary by NPU and software stack; re-run `python3 bench_tilelang_gdn.py` on your machine.

| Kernel | Latency (ms) | #ops (approx) | TFLOPS |
| :-- | --: | --: | --: |
| chunk_cumsum | 1.39 | 4.19e+06 | 0.0030 |
| chunk_scaled_dot_kkt | 9.70 | 6.87e+10 | 7.0824 |
| wy_fast | 9.76 | 1.37e+11 | 14.0816 |
| chunk_h | 9.01 | 2.75e+11 | 30.4938 |
| chunk_o | 11.71 | 3.44e+11 | 29.3311 |
| **total** | **41.58** | **8.25e+11** | **19.8306** |

## Recompiling a dumped `.cpp` manually

Build flags match what TileLang’s `LibraryGenerator` uses for `target="pto"` (see `tilelang/jit/adapter/libgen.py` in your `TL_ROOT` checkout): `bisheng` with `-xcce`, PTO-ISA includes under `$TL_ROOT/3rdparty/pto-isa/include`, CANN headers/libs, and the tilelang template path. Adjust `-I`/`-L` for your machine.

The dumped `.cpp` is the compiler input TileLang generated; it is not meant to be edited by hand unless you know the PTO ABI you are targeting.
