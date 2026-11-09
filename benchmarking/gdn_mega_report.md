# GDN backward development and evaluation

From the repository root, inside the CANN environment:

```bash
source /usr/local/Ascend/cann/set_env.sh
export OMP_NUM_THREADS=4 NPU_DEVICE=npu:0
# Fast: four correctness cases, then one cached forward/backward timing.
python benchmarking/gdn_mega_bwd.py
# Extensive: TP sweep, numerical stress cases, and saved-FLA comparison.
python benchmarking/gdn_mega_bwd.py --extensive --fla /mounted_home/benchmarks/fla-gdn-9e82a80/sweep/gdn_ascend_benchmark.json
```

No manual PYTHONPATH setup is needed. Builds are content-cached; no full
repository rebuild is required. The forward library must already be built.

Development covers cached tails/identical keys, uncached BF16 packed/empty
sequences, and FP32 zero beta. All six gradients use the same independent
recursive Torch/autograd assertions as pytest. The default timing is
L=2048, Hv/Hqk=16/2, three measured samples after two warmups.

Useful development options:

```bash
python benchmarking/gdn_mega_bwd.py --bench-only --shape 8192 64 16 --repeats 5
python benchmarking/gdn_mega_bwd.py --bench-only --profile benchmark_data/gdn_mega_bwd/trace
# Complete regression coverage, beyond the development smoke checks:
PYTHONPATH=python:.:tests python -m pytest -q tests/test_gdn_mega_bwd.py tests/test_gdn_mega_bwd_reference.py
```

`--shape TOKENS HV HK` is repeatable. `--cache-dtype float16|bfloat16|float32`
selects the saved inverse format for both forward and backward; **FP16 is the
default**. The BF16 cache has a known correlated-key accuracy failure. The
extensive audit includes this case, displays the failure and exits nonzero;
it does not relax tolerances or treat pytest's expected-failure marker as a pass.

The extensive run saves exactly three PNGs under
`benchmark_data/gdn_mega_bwd/eval/` (change with `--output`):

- `heatmap.png`: backward / forward ms and ratio. Global Hv=64/48/32/16,
  Hqk=16, TP=1/2/4/8, lengths 256/1024/4096/16384/32768. Local Hv=2/4/6
  is unsupported by forward and shown as N/A. Identical local shapes reuse timings.
- `precision.png`: max absolute error for each of six gradients across ten
  random/stress cases, including weak/strong decay, repeated/correlated keys,
  zero/tiny beta, tails and an 8K scan. Asterisks mark failed bounds.
- `speedup.png`: exact-shape FLA/PTO forward and backward latency ratios,
  with both times in each cell. The figure labels dtype/chunk-size differences.

`results.json` preserves raw timing samples, per-gradient errors and budgets,
source hashes, status, and the supplied FLA data. Per-head-group logs preserve
subprocess output. All results are local/ignored. `--flashqla-bound` additionally
checks relative L2 error <=2% against FP64 and includes those values in the same
precision PNG. It uses identical rounded inputs/adjoints, and adds oracle cost.

```bash
# Replot the saved report, with no Torch/NPU work or external FLA file:
python benchmarking/gdn_mega_bwd.py --plot-only
```

Timing is median synchronized wall time: forward saves inverse/gates into
preallocated buffers; backward consumes them and includes output/workspace
allocation and dispatch. Compilation, warmup, input generation and TP
communication are excluded. FLA uses different input distributions, dtype,
chunk size, state outputs and allocation/autograd overhead; these are reported
latency comparisons, not a controlled identical-workload speedup.

The original gradient bound remains `max|kernel-golden| <=
2*max|reordered_low_precision_Torch-golden| + 1e-5`, separately for each tensor.
Historical v11/v12 results and superseded scripts remain local; they do not
describe the restored `99b370b82` kernel.
