# fast_hadamard_vmi — parametric-N fast Walsh–Hadamard (ptodsl / VMI)

A memory-bound fast Walsh–Hadamard transform (WHT) written in **ptodsl (VMI)**
raw `.pto`. It computes, in place, the unnormalized `y = x @ H` in fp16, where
`H` is the natural-order (Sylvester) ±1 Hadamard matrix of order `N`.

One kernel, `full`, is **parametric on the transform width** `N ∈ {32, 64, 128,
256, 512, 1024, 2048}`. Each row of the `(batch, N)` fp16 input is transformed by
a `log2(N)`-stage deinterleave butterfly. The kernel is multi-core (the batch is
split into `grid=G` contiguous row bands) with a per-block 4-buffer / 32 KB UB
software pipeline (prefetch depth 2), and the per-chunk tile loop is unrolled at
generate-time so every UB offset is an immediate.

This is the first `examples/ptodsl` example (the others under `examples/jit_cpp`
are C++/bisheng). Its distinguishing feature is that **the same `--check` code
path runs both under `cannsim` and on a real A5 device** — correctness is proved
through pyACL (torch-free), so no device is required to validate it.

## Files

| File | Purpose |
| --- | --- |
| `run_hadamard_vmi.py`      | Runner. `--check` (pyACL correctness, cannsim + device), `--nsweep` (all N), `--bench` (device-only bandwidth). |
| `full_n_mlir.py`           | `make_full(N)` — the `@pto.jit` wrapper selecting the per-N `.pto`. |
| `gen_had_full.py`          | Parametric `.pto` generator: `gen_had_full.py [N] [BATCH_HINT] [OUT.pto]`. |
| `fast_hadamard_vmi_full_n{32,64,128,256,512,1024,2048}.pto` | Committed per-N kernels (`@fast_hadamard_vmi_full_n{N}`, uniform naming). |
| `golden.py`                | Numpy reference `ref_hadamard` (`x @ H`). |
| `test/check_cannsim.py`    | cannsim entry: sets argv and calls the runner's `--check`. |
| `scripts/`                 | Minimal cannsim harness: `run_sim.sh`, `run_sim_entry.sh`, `camodel_entry.sh`, `env.sh`, `cannsim_metrics.py`. |

The `.pto` sit beside `full_n_mlir.py` because `@pto.jit(source=...)` resolves the
relative path against the declaring module's directory. Regenerate any of them
with, e.g., `python3 gen_had_full.py 512` (writes `fast_hadamard_vmi_full_n512.pto`).

## Run correctness on cannsim

```bash
export PTOAS_ROOT=/mounted_home/ptoas_feature_vmi \
       LLVM_BUILD_DIR=/llvm-workspace-llvm21/llvm-project/build-shared \
       PTOAS_HOST_TARGET_CPU=tsv110 PTOAS_ENV_SKIP_SMOKE_TEST=1
source /mounted_home/ptoas_feature_vmi/scripts/ptoas_env.sh

cd examples/ptodsl/fast_hadamard_vmi
# N and grid are chosen via env; batch defaults to the smallest valid = grid*65536/N
HAD_N=512 HAD_GRID=1 bash scripts/run_sim.sh test/check_cannsim.py sim_outputs/n512
```

Expect `[check] PASS: N=512 batch=128 grid=1 rel_err≈1e-3`. Verified under
`cannsim -s Ascend950`: N=32 (rel_err 6.6e-4), N=256 (7.3e-4), N=512 (9.7e-4).

## Run on device

Same runner, no cannsim wrapper (needs CANN + pyACL; `--bench` also needs
`torch` + `torch_npu` and a real A5):

```bash
python run_hadamard_vmi.py --check --n 512          # pyACL correctness
python run_hadamard_vmi.py --nsweep                 # correctness over all N
python run_hadamard_vmi.py --bench --n 512 --grid 64  # GM bandwidth vs a torch D2D copy floor
```

`--check` is torch-free and identical to the path cannsim runs. `--bench` reports
effective GM bandwidth (`4*batch*N` bytes/launch) next to the device-to-device
copy ceiling.

## Supported N and performance

- **N = 32 … 2048** are correctness-verified (rel_err ~1e-3). N=32 uses ROT=3
  `pto.vdintlv` window rotations and works (there is no net-rotation cap). **N=16
  (ROT=4) is untested** and left out; N > 2048 needs ≥8 chunks/row and is rejected.
- Divisibility: `batch % (grid * 65536/N) == 0`. The default batch is the smallest
  valid one, `grid * 65536/N`.
- **N ≤ 256** reach compute parity (their 8 tile units are disjoint 256-windows,
  saturating the store subpipe). **N > 256** are correct but store-subpipe-bound
  (~1.71× on cannsim): the in-place butterfly aliases within a tile, so each tile
  serializes into a load phase then a store phase. On device the transform is
  **memory-bound**, running at the HBM bandwidth ceiling.
