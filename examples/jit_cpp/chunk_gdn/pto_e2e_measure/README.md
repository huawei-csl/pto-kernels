# PTO GDN end-to-end measure / verification

This directory contains scripts that chain the **dynamic BSND** PTO kernels
(`dynamic_bsnd/`, chunk size **128**) with **fast_inverse** for `solve_tril`, and
compare end-to-end outputs to the **vendored Triton baseline** in
`../triton_baseline/` (chunk size **64**).

## Prerequisites

- Ascend NPU with `torch_npu`, `bisheng`, and `PTO_LIB_PATH` pointing at PTO-ISA
  headers (defaults are picked up from `ASCEND_TOOLKIT_HOME` / `/sources/pto-isa`
  when present).
- Python imports: `triton`, `vllm.triton_utils` (used by `triton_baseline/fla_vendor`).

## Verify PTO vs Triton (numerical)

From the repository root or from this folder:

```bash
cd /workdir/pto-kernels/examples/jit_cpp/chunk_gdn/pto_e2e_measure
export PTO_LIB_PATH="${PTO_LIB_PATH:-/sources/pto-isa}"
timeout 420s python3 verify_pto_triton_e2e.py --device npu:7 --no-plots
```

Defaults: scatter PNGs under `output/fig/`, metrics CSV under `csv/` (`e2e_metrics_<UTC>.csv` and
`e2e_metrics_latest.csv`). Override with `--fig-dir` and `--csv-dir`.

Optional: `--seed N` to change the base CPU RNG (each shape case adds an offset so cases differ).

The script prints PTO-vs-ref, Triton-vs-ref, and direct PTO-vs-Triton metrics:
RMSE over mean absolute reference magnitude, **R²**, **Pearson r**, and the fraction
of elements inside the `rtol` / `atol` band. Scatter plots use **PTO** on the x-axis
and **Triton** on the y-axis with a red **1:1** line (subsampled to 80k points if needed).
Use `--no-plots` to skip figures.

The script compiles `../fast_inverse/fast_inverse.cpp` once (JIT `.so` next to the
CPP file), runs the full pipeline on NPU, and requires all three agreement gates to pass:
PTO-vs-CPU reference, Triton-vs-CPU reference, and direct PTO-vs-Triton agreement.

## Current coverage

The refreshed suite currently runs **15 cases** spanning:

- single-sequence lengths from `T=128` through `T=4096`
- chunk-aligned packed varlen cases such as `[256,256]` and `[128,128,128]`
- ragged-tail packs such as `[150,300]` and `[129,255]`
- dense boundary mixes such as `[1,17,128,129,255]`
- longer mixed / ladder packs up to total `T=4096`

To regenerate both the summary CSV and scatter plots:

```bash
cd /workdir/pto-kernels/examples/jit_cpp/chunk_gdn/pto_e2e_measure
export PTO_LIB_PATH="${PTO_LIB_PATH:-/sources/pto-isa}"
timeout 900s python3 verify_pto_triton_e2e.py --device npu:7
```

This rewrites:

- `csv/e2e_metrics_latest.csv`
- `output/fig/*.png`
