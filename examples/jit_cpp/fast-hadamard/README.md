## Fast Hadamard

### Usage

```bash
export PTO_LIB_PATH=${ASCEND_TOOLKIT_HOME}
cd examples/fast-hadamard
python run_hadamard.py
```

### Output

Runs correctness tests against a CPU reference, then benchmarks across
batch sizes {1..1024} and row lengths N = {128..16384}, printing duration
and effective bandwidth to the terminal and saving results to `fht_pto.csv`.
