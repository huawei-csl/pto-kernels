# TileLang → PTO C++ codegen (chunk GDN kernels)

This directory is **self-contained**: every script and helper lives here. Regenerating the PTO-ISA C++ sources does not require importing kernel code from other repositories.

## What gets generated

Running the Python entry points below drives TileLang’s PTO backend (`target="pto"`), JIT-compiles the kernel, and **writes the generated C++** next to this README.

| TileLang driver | Generated PTO C++ | Notes |
|-----------------|-------------------|--------|
| `opt_gdn_chunk_cumsum.py` | `opt_gdn_chunk_cumsum.cpp` | Chunk-wise prefix sum along `L` |
| `opt_gdn_chunk_h.py` | `opt_gdn_chunk_h.cpp` | Chunk hidden state / `new_v` / final state |
| `opt_gdn_chunk_o.py` | `opt_gdn_chunk_o.cpp` | Chunk output given hidden state |
| `opt_gdn_chunk_scaled_dot_kkt.py` | `opt_gdn_chunk_scaled_dot_kkt.cpp` | Scaled dot KKT-style lower-triangular block |
| `opt_gdn_wy_fast.py` | `opt_gdn_wy_fast.cpp` | WY-style fast path for `U` and `W` |

## Prerequisites

- **Python environment** with `tilelang` installed (the same package you use for Ascend/PTO JIT).
- **Environment variables** (read by TileLang and by `patch_libgen.py`):
  - `TL_ROOT` — root of the TileLang source tree that provides `3rdparty/pto-isa/include` and templates.
  - `ASCEND_HOME_PATH` — CANN install prefix (headers and `lib64` for linking the JIT `.so`).
- **Ascend NPU + `torch.npu`** — the drivers here call `torch` on NPU so the JIT path runs end-to-end. Codegen happens inside `LibraryGenerator.compile_lib` when the kernel is first compiled.

## PTO C++ codegen steps (how this works)

1. **`patch_libgen.py`**  
   Replaces `LibraryGenerator.compile_lib` with a wrapper that, before invoking `bisheng`, writes `self.lib_code` to the chosen `*.cpp` file in this directory.

2. **Driver scripts (`opt_gdn_*.py`)**  
   Each script:
   - applies the patch and assigns `LibraryGenerator.compile_lib`;
   - calls `tilelang.disable_cache()` so compilation (and dumping) is not skipped by a stale cache;
   - declares the kernel with `@tilelang.jit(..., target="pto")` so the backend emits PTO-ISA C++ rather than AscendC/Hybrid;
   - runs the small built-in numerical test, which triggers JIT and thus the dump.

3. **Artifacts**  
   After a successful run you get the generated source. TileLang’s own `compile_lib` invokes `bisheng` with PTO headers from `$TL_ROOT/3rdparty/pto-isa/include` ahead of CANN defaults, matching upstream TileLang practice for PTO.

## Regenerating the `.cpp` files

From **this directory**:

```bash
export TL_ROOT=/path/to/tilelang-ascend      # example
export ASCEND_HOME_PATH=/path/to/cann        # example

./dump_all_kernels.sh
```

Or run individual drivers:

```bash
python3 opt_gdn_chunk_cumsum.py
python3 opt_gdn_chunk_h.py
python3 opt_gdn_chunk_o.py
python3 opt_gdn_chunk_scaled_dot_kkt.py
python3 opt_gdn_wy_fast.py
```

## Recompiling a dumped `.cpp` manually

Build flags match what TileLang’s `LibraryGenerator` uses for `target="pto"` (see `tilelang/jit/adapter/libgen.py` in your `TL_ROOT` checkout): `bisheng` with `-xcce`, PTO-ISA includes under `$TL_ROOT/3rdparty/pto-isa/include`, CANN headers/libs, and the tilelang template path. Adjust `-I`/`-L` for your machine.

The dumped `.cpp` is the compiler input TileLang generated; it is not meant to be edited by hand unless you know the PTO ABI you are targeting.
