# vLLM-Ascend Triton GDN baseline

Benchmarks the same logical pipeline as `chunk.py` (cumsum → scaled KKT → WY `recompute_w_u` → `chunk_gated_delta_rule_fwd_h` → `chunk_fwd_o`), **without** timing `solve_tril` (the KKT output is fed straight into `recompute_w_u_fwd`, like `tilelang_codegen/bench_tilelang_gdn.py`).

Triton kernel implementations are **vendored** under `fla_vendor/` (copies of upstream FLA sources; see `fla_vendor/SOURCES.md`). The full `chunk_gated_delta_rule_fwd` wrapper from upstream `chunk.py` is **not** used (it pulls in `get_forward_context()`).

## Layout (intentional difference vs TileLang)

| | TileLang drivers | vLLM FLA Triton |
|--|------------------|-----------------|
| Core layout | `[B, H, L, …]` (head before sequence) | `[B, T, H, …]` (sequence before head) |

Benchmarks use **native** layouts for each stack; **no extra transpose time** is included. The two codepaths are not bit-identical (layout, kernels, and internal chunk tile **C = 128** in TileLang vs **BT = 64** in vLLM Triton). Chunk size is an **algorithm parameter**; it is fine to compare runs where **batch, total sequence length, heads, and hidden dims** match even when **C** and **BT** differ (approximate op counts for KKT/WY then scale with the chosen tile).

## Imports

Add **`pto-kernels/examples/jit_cpp/chunk_gdn`** to `PYTHONPATH` so `triton_baseline` resolves (same pattern as other drivers here):

```bash
export PYTHONPATH=/path/to/pto-kernels/examples/jit_cpp/chunk_gdn
```

The vendored kernels still use **`from vllm.triton_utils import tl, triton`** (vLLM’s Triton bindings on Ascend); your environment must provide the **`vllm`** package with Triton support. You do **not** need the `vllm_ascend` tree on `PYTHONPATH` for these scripts.

Varlen is required for these kernels: use **`B = 1`** and a stepped **`cu_seqlens`** (e.g. `N` sequences of length `L` each: `[0, L, 2L, …, N·L]`). That mirrors the **total token count** of the TileLang shape `(B,H,L,…)` when `B·L` matches `T`.

## Triton timing caveat

Triton kernel launches may not synchronize with `torch.npu.synchronize()` alone. The benchmark uses `gdn_bench_common.do_bench_triton`, which records `torch.npu.Event`s and calls **`end.synchronize()`** after each timed iteration (see `pto-kernels/.skills/npu_kernel_general/skills.md`).

## Commands

From `pto-kernels/examples/jit_cpp/chunk_gdn` (with NPU + `torch_npu`):

```bash
export PYTHONPATH=/path/to/pto-kernels/examples/jit_cpp/chunk_gdn

# Default matches TileLang total tokens: N_seq=16, L_seg=16384 → T=262144, H=DK=DV=128.
python3 triton_baseline/bench_triton_gdn.py

# Optional overrides:
export GDN_TRITON_NPU_DEVICE=npu:0
export GDN_TRITON_N_SEQ=16
export GDN_TRITON_L_SEG=16384
export GDN_TRITON_H=16
export GDN_TRITON_DK=128
export GDN_TRITON_DV=128
```

Numerical checks (refs + end-to-end smoke with `solve_tril`):

```bash
python3 triton_baseline/verify_triton_gdn_kernels.py
```

## Approximate op counts and TFLOPS

**Chunk size (`C` vs `BT`) enters the approximate op formulas** (especially KKT and WY), so **total reported FLOPs are not directly comparable** across TileLang and Triton when the internal tiles differ, even if batch, sequence, heads, and hidden sizes match. For **apples-to-apples** comparison between the two stacks, **use measured latency (ms)** per kernel and end-to-end; treat TFLOPS here as a rough within-stack figure derived from those formulas.

Use the same spirit as `gdn_bench_common.approx_ops_gdn`; Triton uses **`approx_ops_gdn_triton`** in `gdn_bench_common.py`, with **`BT`** in the KKT and WY terms where TileLang uses **`C`**. Per-kernel totals for **one** representative configuration:

**TileLang** (`tilelang_codegen/README.md`, shape `(B,H,L,DK,DV,C)=(16,16,16384,128,128,128)`, **no solve_tril** in the benchmark):

| Kernel | Latency (ms) | #ops (approx) | TFLOPS |
| :-- | --: | --: | --: |
| chunk_cumsum | 1.39 | 4.19e+06 | 0.0030 |
| chunk_scaled_dot_kkt | 9.70 | 6.87e+10 | 7.0824 |
| wy_fast | 9.76 | 1.37e+11 | 14.0816 |
| chunk_h | 9.01 | 2.75e+11 | 30.4938 |
| chunk_o | 11.71 | 3.44e+11 | 29.3311 |
| **total** | **41.58** | **8.25e+11** | **19.8306** |

**Triton** (measured on one NPU run; **your** latencies will vary): same **total** sequence length `T = N_seq·L_seg = 16·16384 = 262144`, `H = DK = DV = 128`, `B = 1` packed, internal tile **`BT = 64`**. Command: `PYTHONPATH=.../chunk_gdn python3 triton_baseline/bench_triton_gdn.py` (defaults above).

| Kernel | Latency (ms) | #ops (approx) | TFLOPS |
| :-- | --: | --: | --: |
| chunk_cumsum | 1.02 | 4.19e+06 | 0.0041 |
| chunk_scaled_dot_kkt | 4.83 | 3.44e+10 | 7.1075 |
| wy_fast | 15.60 | 6.87e+10 | 4.4048 |
| chunk_h | 30.85 | 2.75e+11 | 8.9110 |
| chunk_o | 16.11 | 3.44e+11 | 21.3240 |
| **total (no solve_tril)** | **68.42** | **7.22e+11** | **10.5464** |

Approximate op formulas for this Triton path (same `B,H,T,DK,DV` as above; **`BT`** only appears in KKT/WY):

| Kernel | #ops formula |
| :-- | :-- |
| chunk_cumsum | `B·H·T` |
| chunk_scaled_dot_kkt | `B·H·T·BT·DK` |
| wy_fast | `B·H·T·BT·(DK+DV)` |
| chunk_h | `4·B·H·T·DK·DV` |
| chunk_o | `5·B·H·T·DK·DV` |

## Files

| File | Role |
|------|------|
| `fla_vendor/` | Vendored Triton FLA sources + `SOURCES.md` (upstream link) |
| `refs_bthd.py` | PyTorch references for cumsum + KKT in `[B,T,H,…]` layout |
| `bench_triton_gdn.py` | Latency / TFLOPS (no `solve_tril`) |
| `verify_triton_gdn_kernels.py` | Per-kernel checks + e2e smoke (with `solve_tril`) |
