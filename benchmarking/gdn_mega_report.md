# GDN forward/backward reports

Figures are saved as PNG only; no SVG files are generated.

From the repository root, inside the CANN container with the forward library
already built:

```bash
source /usr/local/Ascend/cann/set_env.sh
export OMP_NUM_THREADS=4 NPU_DEVICE=npu:0 PYTHONPATH=python:.:tests
python benchmarking/gdn_mega_report.py --output benchmark_data/gdn_mega_bwd/report_v10
```

This one command generates three reports using **BF16 inverse factors and saved
gates/beta**, FP16 model inputs and FP32 gates. Forward writes the caches;
backward consumes those exact buffers. `--cache-dtype float16` or `float32`
selects a different inverse format consistently across all reports.

| Output under `--output` | Contents |
| --- | --- |
| `timings/heatmap.png` | Each cell: backward / forward ms, then their ratio. Global value heads 64/48/32/16, QK heads 16, TP 1/2/4/8, lengths 256/1024/4096/16384/32768. |
| `fla/speedup_heatmap.png` | Forward and backward speedup = saved FLA time / PTO time, with both times in each cell. Matches exact local shapes. |
| `errors/error_heatmap.png` | Max absolute error of each of six backward gradients against independent recursive FP32 PyTorch + autograd. Seven small random/stress cases, including dependent keys and gate extremes described below. |

Raw samples, per-gradient errors/budgets, source hashes, commit/dirty status,
forward library hash and measurement options are saved beside the plots.
Each section has `metadata.json`; `status=complete` means that section finished.
A failing numerical audit writes its errors/plot and exits nonzero. Timing
results only check finite gradients, not agreement with a long-sequence oracle.

The saved FLA file defaults to
`/mounted_home/benchmarks/fla-gdn-9e82a80/sweep/gdn_ascend_benchmark.json`
inside Docker (or `/home/fskogh/benchmarks/...` on the host). Override with
`--fla PATH`. FLA uses equal QK/value heads 16/24/32/48/64; those are separate
measurements, not a substitution into the TP grid. FLA is BF16/C64; PTO is
FP16/C128. The plot explicitly notes differences in input distributions,
state outputs and timing/allocation contracts. These are comparisons with the
saved benchmark, not controlled same-dtype experiments.

Timing is median synchronized wall time, excluding compilation and warmup.
Forward uses preallocated buffers and writes the cache; backward includes its
output/workspace allocation. No TP communication is measured. Local value-head
counts 2/4/6 are unsupported by the current forward producer and shown as N/A,
so the full TP report measures 60 configurations and marks 20 unavailable.
Five repeats are the default. Extra device-only forward benchmarks are skipped.

Fast development and replotting:

```bash
# Eight TP shapes, two FLA shapes and two numerical cases; separate output.
python benchmarking/gdn_mega_report.py --quick --repeats 3 --output benchmark_data/gdn_mega_bwd/report_smoke
# Only one section, keeping the others intact.
python benchmarking/gdn_mega_report.py --only errors --output benchmark_data/gdn_mega_bwd/report_v10
# No compilation, Torch import or NPU execution; preserves measurement metadata.
python benchmarking/gdn_mega_report.py --plot-only --output benchmark_data/gdn_mega_bwd/report_v10
```

Use `--only timings`, `--only fla` or `--only errors` for focused updates. Reuse
one output directory for the same settings; use a new directory for a new
cache dtype or implementation snapshot. Replotting a quick report automatically
retains its original selection. Full numerical checks live in
`tests/test_gdn_mega_bwd.py`; this small heatmap is an audit, not a substitute.

## Backward numerical coverage

All tests compare all six gradients using the same rounded inputs and upstream
gradients in three computations: the PTO kernel, a higher-precision independent
recursive Torch oracle with autograd, and a reordered input-precision Torch
recurrence with autograd. For each gradient tensor, one scalar error must obey
`max(abs(kernel - golden)) <= 2 * max(abs(Torch_low - golden)) + 1e-5`.
FP16/BF16 tests use an FP32 golden; FP32 tests use FP64. The heatmap displays
actual max absolute errors, not ratios or per-element tolerances. Its JSON also
stores the scalar budget. It uses FP16 inputs with FP32 gates and both forward
caches; it does not numerically validate every large timing shape.

The hardware suite covers FP16/BF16/FP32 recomputation, and FP16 forward inputs
with inverse-only, gates-only, both FP32, both FP16 and both BF16 cache modes.
Coverage includes grouped heads, empty/packed sequences, chunk boundaries and
tails, nonzero/zero/missing initial state, output-only/state-only upstream
gradients, noncontiguous tensors, streams, queued calls, immutable inputs/caches,
and the allocation-only production guard.

| Stress mode | Inputs |
| --- | --- |
| `both` | Random normalized keys, random beta in [0,1), log-gates in [-0.1,0]. |
| `replace` | Exactly identical unit keys, beta=1, g=0, through 513 tokens. |
| `weak` | Random keys, g in [-1e-5,0], through 513 tokens. |
| `correlated_weak` | Nearly identical normalized keys with 1e-3 perturbations, beta=1, g in [-1e-5,0], 257 tokens. |
| `strong_decay` | g in [-0.75,-0.25], so retention exp(g) is about 0.47–0.78, 256 tokens. |
| `zero_beta` | beta=0, exercising division/cancellation handling. |

Here g is **log retention**: g near zero means weak decay/high retention;
more-negative g means stronger decay/lower retention. The tests do not establish
accuracy for arbitrarily negative gates, unnormalized huge keys or long-context
adversarial inputs. Most backward oracle checks stop at 513 tokens; timing
sweeps reach 32K. CPU reference tests additionally use finite differences.

## Optional FlashQLA bound

```bash
python benchmarking/gdn_mega_report.py --only errors --flashqla-bound \
  --output benchmark_data/gdn_mega_bwd/report_flashqla_bound
```

This additionally applies [FlashQLA's pinned test helper](https://github.com/QwenLM/FlashQLA/blob/a14dbf3474514e1674526431e960325f72081e9f/tests/gdr_common.py#L189-L200):
`norm(kernel - reference, 2) <= 0.02 * norm(reference, 2)`, per gradient tensor.
There is no additive absolute tolerance; an exactly zero reference requires zero
error. Norms use FP64, and an extra independent FP64 recursive autograd reference
uses the **same rounded inputs and upstream gradients** as the original FP32
audit. This retains the original max-error assertion and checks both bounds.
It does not run FlashQLA's CUDA kernel or reproduce its full BF16 test suite.

Outputs include `flashqla_error_heatmap.png`; `errors.json` records error
and reference L2 norms, their ratio, FP64 max absolute error, reference maximum
and RMS, and pass/fail. Replotting also reproduces this additional heatmap.

On the seven existing audit cases (FP16 inputs, BF16 inverse + saved gates),
all **42 gradient checks pass**. Worst L2 relative error is **0.08268%**, versus
the **2%** limit. Weak-decay `dg` has max absolute error **0.307835** against
FP64, but L2 relative error **0.07700%**; its reference RMS is **121.9003**.
The original FP32 errors and Torch-derived budgets reproduce exactly. L2
relative error measures aggregate tensor accuracy; it does not cap individual
outliers or establish accuracy on untested shapes.
