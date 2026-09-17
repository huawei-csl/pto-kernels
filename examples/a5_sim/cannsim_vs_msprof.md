# SiLU / SwiGLU — msprof vs cannsim (Ascend950 / dav-c310-vec)

Pure-vector A5 examples for **`pto-kernels/examples/a5_sim`**. Recommended first step for Ascend950 simulator validation before mix kernels in [`megagdn-pto/benchmarks/a5_sim`](../../megagdn-pto/benchmarks/a5_sim).

## Executive summary

| Aspect | msprof op simulator | cannsim record |
|--------|---------------------|----------------|
| SoC flag | `Ascend950PR_9599` | `Ascend950` |
| AICore arch | `dav-c310-vec` | `dav-c310-vec` |
| Correctness (smoke) | **PASS** (SiLU T=128, SwiGLU T=128) | **PASS** (same shapes) |
| Invocation | Wraps `python3 vec_sim.py` directly | Executable `run_cannsim_entry.sh` + `-u "..."` |
| Typical smoke wall (Kunpeng) | SiLU ~52 s, SwiGLU ~75 s | SiLU ~42 s, SwiGLU ~52 s |
| Typical smoke wall (AMD EPYC) | SiLU ~12 s, SwiGLU ~18 s | SiLU ~9 s, SwiGLU ~11 s |
| Exit code | 0 on success | May return non-zero after **teardown segfault** even when JSON is valid |

## Tool overview

**msprof** preloads the CA model via `LD_PRELOAD` and runs Python + ctypes kernel launch (same pattern as [`ptoisa-a5-test/tests/torch_sim`](../../ptoisa-a5-test/tests/torch_sim/msprof_mechanism.md)).

**cannsim** runs a standalone entry script under full SoC simulation. User args pass via `-u "--kernel silu --mode ..."`, not trailing argv.

## Correctness

| Kernel | Shape | msprof | cannsim | Reference |
|--------|-------|--------|---------|-----------|
| SiLU | T=128 | PASS | PASS | `x * sigmoid(x)` on CPU |
| SwiGLU | batch=1, input_n=256 (T=128 out) | PASS | PASS | split + SiLU gate × value on CPU |

Inputs are allocated on CPU then copied to NPU; reference checks run on CPU (simulator rejects many dynamic NPU ops).

## Speed comparison (scale ladder, timing-only sweep)

### Kunpeng-920 (aarch64, May 2026)

**SiLU msprof vs cannsim** (seconds, wall clock):

| label | T | msprof | cannsim | ratio msprof/cannsim |
|-------|---|--------|---------|----------------------|
| smoke | 128 | 52 | 42 | 1.2× |
| tiny | 512 | 24 | 15 | 1.6× |
| small | 1024 | 26 | 17 | 1.5× |
| medium | 4096 | 29 | 17 | 1.7× |

**SwiGLU msprof vs cannsim**:

| label | T | msprof | cannsim | ratio |
|-------|---|--------|---------|-------|
| smoke | 128 | 75 | 52 | 1.4× |
| tiny | 512 | 49 | 27 | 1.8× |
| small | 1024 | 61 | 29 | 2.1× |
| medium | 4096 | 52 | 22 | 2.4× |

### AMD EPYC 9654 (x86_64, May 2026)

**SiLU msprof vs cannsim** (seconds, wall clock):

| label | T | msprof | cannsim | ratio msprof/cannsim |
|-------|---|--------|---------|----------------------|
| smoke | 128 | 12 | 9 | 1.3× |
| tiny | 512 | 7 | 4 | 1.8× |
| small | 1024 | 7 | 4 | 1.8× |
| medium | 4096 | 12 | 5 | 2.4× |

**SwiGLU msprof vs cannsim**:

| label | T | msprof | cannsim | ratio |
|-------|---|--------|---------|-------|
| smoke | 128 | 18 | 11 | 1.6× |
| tiny | 512 | 13 | 6 | 2.2× |
| small | 1024 | 17 | 6 | 2.8× |
| medium | 4096 | 20 | 7 | 2.9× |

On both hosts, cannsim is generally **faster** on wall clock for these pure-vector kernels once T≥512; msprof carries heavier profiling/injection overhead. Tool ratios are similar; absolute wall time is ~3–5× lower on the AMD EPYC host.

## Failure modes

| Issue | Mitigation |
|-------|------------|
| `torch.randn` on NPU under sim | Create tensors on CPU, `.to("npu:0")` |
| Reference ops on NPU fail | Compare `y.cpu()` vs CPU PyTorch ref |
| cannsim segfault on exit | JSON is still written; `run_cannsim.sh` accepts valid `--output-json` |
| A5 `pipe_barrier(PIPE_V)` compile error | Use `PIPE_ALL` in SwiGLU compute path |
| `Stride` ambiguous on A5 | Qualify as `pto::Stride<...>` |

## Invocation examples

```bash
cd pto-kernels/examples/a5_sim
source $ASCEND_HOME_PATH/bin/setenv.bash
export PTO_LIB_PATH=/path/to/pto-kernels/third_party/pto-isa

MSPROF_TIMEOUT=30 ./run_msprof.sh --kernel silu --mode sweep --skip-correctness \
  --output-json outputs/silu_sweep_msprof.json

./run_cannsim.sh --kernel swiglu --mode correctness --batch 1 --input-n 256 \
  --output-json outputs/smoke_swiglu_cannsim.json
```

## Recommendations

1. **Start with SiLU** (simplest 1D pipeline) under msprof smoke correctness.
2. Use **cannsim** for faster scale sweeps once smoke passes.
3. Use **mix chunk_h_mini** only after pure-vector path is green ([`megagdn-pto/benchmarks/a5_sim`](../../megagdn-pto/benchmarks/a5_sim)).

## References

- Harness README: [`README.md`](README.md)
- 910B chunk_h comparison: [`megagdn-pto/benchmarks/simulator/cannsim_vs_msprof.md`](../../megagdn-pto/benchmarks/simulator/cannsim_vs_msprof.md)
