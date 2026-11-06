Goal is to optimize the chunked GDN prefill in csrc/kernel/gdn_mega/.


These are the checkboxes to complete:

0) Make sure you are able to run inside docker file, and compile fast (compilatio should only be a few seconds) by disabling all the other kernel compiles than the relevant mega GDN. Having a fast running env with fast feedback is essential for kernel optimization, this is the docker env "docker run -it --rm --privileged --network=host --ipc=host --shm-size=16g     --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3     --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7     --device=/dev/davinci_manager --device=/dev/hisi_hdc     --volume /usr/local/sbin:/usr/local/sbin --volume /usr/local/Ascend/driver:/usr/local/Ascend/driver     --volume /usr/local/Ascend/firmware:/usr/local/Ascend/firmware     --volume /etc/ascend_install.info:/etc/ascend_install.info     --volume /var/queue_schedule:/var/queue_schedule --volume ~/.cache/:/root/.cache/ -v /scratch/model_weights/:/model_weights/ -v $HOME:/mounted_home -w /mounted_home  quay.io/ascend/cann:9.1.0-910b-ubuntu22.04-py3.11 /bin/bash". 
1) Make sure you write clean and simple benchmarking code such that we know the basic measures.
2) And that it passes the correctness tests.
3) If any of these previous steps are not really clean make sense (one docker run, compilation, run eval, etc should be very fast, best case just some seconds), pause and let me know what the problem is
4) This is the main work, to optimize the kernel until we achieve a 3x speedup. You can keep a memory.md with notes of things that you find that are suspicious in terms of performance, may cause slowdowns, clear improvements, etc, that you find while you are understanding the code deeply. And when you actually manage to get a speedup while keeping correctness (passing tests) you can commit that work and record an entry in success.md with the commit id, and a short description of improvement. And keep doing this loop until you hit the 3x speedup. (NEVER STOP UNTIL YOU HIT THE TARGET)

Optimization starting point tips:
To begin there are some questionable things in this kernel
- Uses of the identity matrix and negative identity matrix, zero matrix. Those are passed around a bit too much, is there no cleaner way?
- negating of matrix by mulitplying with -I, looking at the instruction that comes after is this really neccesary?
- Does the overall flow make sense? Does it minimize transfers between cube cores and vector cores? Does it minimize data movement? Does it maximally overlap steps with pipelining and double buffering,...

Profiling tips:
Commands to run but fix kernelname and python program
```bash
LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/Ascend910B2/lib:$LD_LIBRARY_PATH
msprof op simulator --aic-metrics=PipeUtilization --kernel-name="kernel_name_here" --output="msprof_res"  python progamname_here.py
```

Then use `../traceviz-mcp` for better traces/profiling visaulizations make use of the library which allow you to see pipelining overlap, compare two profiling runs, see bubbles/gaps etc.

Common pitfalls:
- incorrect syncronization, for example every wait flag dont have a corresponding set flag, causing a deadlock
- Tiles overflow the UB/L1/L0 memories, must be careful when double buffering

## Available optimization workflows and commands (2026-10-30)

These workflows use the two existing tools, `benchmarking/gdn_mega.py` and
`benchmarking/profile_gdn_mega.py`, plus `tests/test_gdn_mega.py`. Do not create
extra benchmark scripts or committed experiment files for each idea.

### 1. Run commands in the prepared NPU environment

The running container is `gdn-mega-opt`; its repository root is
`/mounted_home/autokernel_gdn/pto-kernels`. It has the CANN compiler, PyTorch,
torch-npu, msprof and access to the Ascend 910B2. Reuse it for NPU work.

One-shot build from the host:

```bash
docker exec -w /mounted_home/autokernel_gdn/pto-kernels \
  -e PYTHONPATH=python -e OMP_NUM_THREADS=4 -e NPU_DEVICE=npu:0 \
  gdn-mega-opt bash -c \
  'source /usr/local/Ascend/cann/set_env.sh; python benchmarking/gdn_mega.py build'
```

Or open a shell once. Commands in sections 2–7 below run in this shell:

```bash
docker exec -it -w /mounted_home/autokernel_gdn/pto-kernels gdn-mega-opt bash
source /usr/local/Ascend/cann/set_env.sh
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export NPU_DEVICE=npu:0
```

Run only ONE NPU test, benchmark or profiling job at a time. Independent CPU
compilation and analysis can run concurrently. Hardware validation here covers
910B2; do not claim A5 correctness from these runs.

### 2. Compile isolated candidates quickly

```bash
# All seven head specializations: recently about 10–11 seconds.
python benchmarking/gdn_mega.py build --output build/gdn_candidate

# Faster partial build: H8 and H16 only, suitable for the quick screen.
python benchmarking/gdn_mega.py build --quick --output build/gdn_quick

# Compile a separate source snapshot without editing the installed library.
python benchmarking/gdn_mega.py build \
  --source build/gdn_experiment_src --output build/gdn_experiment

# Rebuild the host binding when its C++ interface changes.
# GDN_MEGA_ONLY is local-only; restore its source hooks before configuring.
git apply .git/gdn-mega-only.patch
cmake -S . -B build -DGDN_MEGA_ONLY=ON
cmake --build build -j4
# Remove the local hooks before staging or committing source changes.
git apply -R .git/gdn-mega-only.patch
```

`build` writes a library and source fingerprints to the selected directory;
it does not install it. Pass that directory through `--library` when testing,
timing or profiling. A quick build cannot run the full head suite. Use separate
output directories for partial and full builds. Source snapshots must contain
the complete seven-file kernel and its normal head dispatch.

### 3. Check numerical accuracy: quick screen and full acceptance

```bash
# Nine selected cases; previously a few seconds with warm reference caches.
python benchmarking/gdn_mega.py test --library build/gdn_candidate --quick

# All 264 tests, including 178 fixed numerical cases.
# Recent warm-cache full runs took about 164–167 seconds; cold runs take longer.
python benchmarking/gdn_mega.py test --library build/gdn_candidate

# Optional independent cache-free rerun against the installed library.
env -u GDN_REFERENCE_CACHE python -m pytest tests/test_gdn_mega.py -q
```

The golden is an independent CPU FP64 token recurrence on the actual rounded
inputs. Every raw-output and final-state element must satisfy
`abs(error) <= 1e-4 + 1e-3*abs(reference)`, with finiteness and relative RMSE
below 0.005 also required. Do not change tolerances to make an optimization pass.
The full suite covers correlated keys, beta endpoints/interior values, tiny,
zero and strong gates, supplied/absent state, tails, packed/empty sequences and
long contexts. Audit cases use five poisoned-scratch launches and check input
integrity, output guards and scratch independence.

The CLI verifies the loaded library hash and reuses validated CPU golden caches.
It never caches candidate outputs or pass/fail decisions. A quick pass is only
a screen; retain a candidate only after full acceptance.

### 4. Measure speed quickly and compare candidates fairly

```bash
# Small initial screen with fewer repetitions; noisy, not final evidence.
python benchmarking/gdn_mega.py bench --library build/gdn_candidate \
  --head-configs 8/2 16/4 --lengths 256 4096 --trials 5 --inner 20 \
  --output benchmark_data/gdn_mega/quick_bench.json

# Representative steady-state measurements using the default 15 x 20 calls.
python benchmarking/gdn_mega.py bench --library build/gdn_candidate \
  --head-configs 8/2 16/4 24/8 48/16 --lengths 256 4096 32768

# Matched candidate versus installed library; keep exploratory reports local.
python benchmarking/gdn_mega.py heatmap --library build/gdn_candidate \
  --baseline-lib build/lib --head-configs 8/2 16/4 48/16 \
  --lengths 256 4096 --output benchmark_data/gdn_mega/comparison.json \
  --report-dir benchmark_data/gdn_mega/comparison
```

Both implementations get identical inputs and caller allocations. Inputs,
outputs and all workspace are allocated once per shape. After five checked
warmups, capture 20 calls in an NPU graph, replay once untimed, then measure
15 replays with NPU events. Report the median graph duration divided by 20.
Every call reuses the same initial state and buffers; final state is not fed
back. Allocation, input copies, reference computation, validation and graph
capture are excluded. No cache flush or external scratch clearing occurs.
These are warmed device timings, not end-to-end request latency.

Use identical repetition counts for comparisons. Check both short and long
sequences and multiple head configurations; repeat a promising comparison to
separate an improvement from timing noise. `--no-initial-state` and
`--gate-scale` support additional sensitivity measurements, not easier acceptance.

### 5. Regenerate the full speed and absolute-error heatmaps

```bash
# Full candidate versus original 10748eb, with both implementations freshly timed.
python benchmarking/gdn_mega.py heatmap --library build/gdn_candidate \
  --baseline-lib benchmark_data/gdn_mega/baseline_lib

# Re-render from saved measurements without another NPU run.
python benchmarking/gdn_mega.py render benchmark_data/gdn_mega/heatmap.json
```

The default grid is 63 cells: H/Hg = 8/2, 12/3, 16/4, 24/6, 24/8, 32/8,
48/12, 48/16 and 64/16; lengths = 128, 256, 1024, 4096, 8192, 16384 and
32768. The full sweep takes tens of minutes, largely from CPU validation at
large shapes. The original library exists in this workspace but is not shipped
in Git; a fresh checkout needs a compatible baseline built separately.

This updates `docs/gdn_mega/heatmap.md`, `error_heatmap.md`, and their two PNGs.
Each speed cell includes current and baseline median total microseconds. Failed
baselines retain measured runtimes with an asterisk; passing entries are not
starred. A failing current cell makes the command fail. Runtime ratios against
a failing baseline are not accuracy-equivalent speedups. The error image shows
maximum absolute raw-output errors; final states are checked too. This ordinary
input grid does not replace the full adversarial suite.

### 6. Profile on hardware with msprof

```bash
python benchmarking/profile_gdn_mega.py hardware \
  --library build/gdn_candidate --head-config 48/16 --length 4096 \
  --output benchmark_data/gdn_mega/profile_hw
```

Hardware mode uses `msprof op` with Roofline metrics and one real kernel launch.
It verifies that launch's output/state against FP64. I can examine the resulting
metrics and compare hardware timings across shapes to identify underused cube
work, vector/scalar overhead, memory traffic, imbalance and synchronization.
Use unprofiled benchmark runs to establish speedups, since profiling can perturb
execution. A full kernel profile includes both global phases.

### 7. Inspect detailed pipeline overlap with the simulator and traceviz

```bash
python benchmarking/profile_gdn_mega.py simulator \
  --library build/gdn_candidate --head-config 16/4 --length 512 \
  --output benchmark_data/gdn_mega/profile_sim

# Locate the actual OPPROF directory produced by that run.
find benchmark_data/gdn_mega/profile_sim -type d -name 'OPPROF*'

# Substitute that directory for OPPROF_REPLACE_WITH_RUN below.
python benchmarking/profile_gdn_mega.py render \
  --input benchmark_data/gdn_mega/profile_sim/OPPROF_REPLACE_WITH_RUN \
  --traceviz ../traceviz-mcp --units 'core0.*' \
  --output benchmark_data/gdn_mega/profile_view
```

Simulator mode uses `msprof op simulator` with PipeUtilization and checks FP64
accuracy. The renderer uses the sibling traceviz Python CLI and produces
`timeline.png` and `summary.txt`; `--units` selects cores and
`--window START_NS END_NS` selects a time interval. No traceviz MCP tools are
currently connected; the sibling checkout's CLI is available. I can inspect
the images/traces for cube, vector, scalar/control and MTE overlap, idle gaps,
barriers and workload distribution. Simulator timing is diagnostic, not a
hardware latency or TFLOPS measurement. Keep traces/figures in ignored
`benchmark_data/`; do not recreate committed profiling reports.

### 8. Analyze and implement larger algorithm/schedule changes

I can trace the complete pipeline, count logical and executed matrix work,
estimate bytes moved and UB/L1/L0 usage, and compare those estimates with
profiles. This can expose redundant operations, repeated grouped-head work,
unnecessary materialization, poor tile utilization or insufficient parallelism.
TFLOPS comparisons must use a stated operation count: fewer operations can make
a kernel faster while lowering reported logical TFLOPS.

I can prototype chunk size C, inverse leaf size C_inv, inverse algorithm A,
joining algorithm J, tensor precision/scaling, cube/vector partitioning,
double buffering and head/token scheduling in isolated source/build directories.
These are source changes, not existing runtime tuning flags. I can add targeted
regressions to the one existing test file, replay available captured inputs,
and compare generated input bytes to ensure a test refactor has not made the
problem easier. Keep only verified improvements and remove rejected experiments.

### 9. Run formatting and static analysis locally, without another Docker image

Run these from the HOST repository root, using its installed Python and uv:

```bash
pre-commit run --all-files
UV_PYTHON_DOWNLOADS=never uvx --python /usr/bin/python3 --from ruff==0.15.22 ruff check .
UV_PYTHON_DOWNLOADS=never uvx --python /usr/bin/python3 \
  --from prospector==1.14.0 prospector --ignore-paths extra-info
```

The additional Prospector ignore is only for this workspace's unreadable,
untracked `extra-info/` directory; omit it in a clean checkout. CI uses Python
3.10, while the local successful run used Python 3.12. Do not download a Docker
image or another Python just to lint. Keep numerical acceptance checks intact
when addressing lint; do not globally disable rules to hide findings.

### 10. Keep changes reviewable and record verified improvements

I can review diffs, inspect compiler output, use isolated worktrees, resolve
straightforward merge conflicts, run appropriate integration checks and commit
validated changes. Record an actual improvement's configuration, before/after
timings, accuracy results and commit in local `success.md`/`memory.md`.

After a full candidate build passes accuracy and performance validation, install
it explicitly if needed with `cp build/gdn_candidate/libno_workspace_kernel.so build/lib/`.
Never replace the original baseline library. Preserve library/source hashes in
measurement records and distinguish raw timings from accepted speedups.

`instructions.md`, `success.md`, `memory.md`, `docs/gdn_mega/README.md`, and
`csrc/kernel/gdn_mega/README.md` stay local and must not be added to the PR.
Keep only the two production benchmark/profile tools, the one GDN test file,
relevant tracked Markdown and the two heatmap PNGs. Raw data, caches and profile
artifacts belong under ignored `benchmark_data/` or `build/`. Ask about an
unclear behavioral/API tradeoff; do not silently change the contract.

