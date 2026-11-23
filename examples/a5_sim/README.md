# A5 Pure-Vector Simulator Examples (SiLU + SwiGLU)

Self-contained **Ascend950PR** pure-vector PTO kernels with **msprof op simulator** and **cannsim record** harnesses. Use these to validate A5 simulator plumbing before tackling mix kernels (see [`megagdn-pto/benchmarks/a5_sim`](../../megagdn-pto/benchmarks/a5_sim)).

Kernels compile with `--cce-aicore-arch=dav-c310-vec` and `-DREGISTER_BASE`. For the 910B `chunk_h` simulator benchmark (different arch), see [`megagdn-pto/benchmarks/simulator/README.md`](../../megagdn-pto/benchmarks/simulator/README.md).

## Prerequisites

```bash
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
export PTO_LIB_PATH=/path/to/pto-kernels/third_party/pto-isa
pip install torch torch-npu
```

Build kernels:

```bash
cd pto-kernels/examples/a5_sim
python3 -m common.build --all
```

## Quick start

```bash
# Correctness smoke (msprof)
./run_msprof.sh --kernel silu --mode correctness --num-elements 128 --label smoke
./run_msprof.sh --kernel swiglu --mode correctness --batch 1 --input-n 256 --label smoke

# Same under cannsim
./run_cannsim.sh --kernel silu --mode correctness --num-elements 128 --label smoke
./run_cannsim.sh --kernel swiglu --mode correctness --batch 1 --input-n 256 --label smoke

# Scale ladder timing
./run_msprof.sh --kernel silu --mode sweep --skip-correctness \
  --output-json outputs/silu_sweep_msprof.json
./run_thread_sweep.sh   # OMP sweep, T=512, both tools
```

## Host environments

| Host | CPU | Logical CPUs | Arch | CANN |
|------|-----|--------------|------|------|
| Kunpeng server | HUAWEI Kunpeng 920 5250 | 192 (4×48 cores, 1 thread/core) | aarch64 | 9.0.0 |
| x86 server | AMD EPYC 9654 96-Core | 192 (1×96 cores, 2 threads/core) | x86_64 | 9.0.0 |

Both measured May 2026. On the same ladder shapes, the x86 host is roughly **3–5× faster** in simulator wall time (startup overhead still dominates smoke on both).

## Simulator time cost summary

Wall time uses `time.perf_counter()` around one kernel launch (includes PEM/msprof or cannsim startup). **T** = output element count (same ladder labels as the 910B `chunk_h` benchmark). **Correctness PASS** at smoke shape on both tools (PyTorch CPU reference).

### Kunpeng-920 (aarch64)

#### SiLU — msprof (`Ascend950PR_9599`)

| Label | T | Sim wall | ms/element |
|-------|---|----------|------------|
| smoke | 128 | **52 s** | 406 ms |
| tiny | 512 | **24 s** | 48 ms |
| small | 1024 | **26 s** | 25 ms |
| varlen_2x512 | 1024 | **26 s** | 26 ms |
| medium | 4096 | **29 s** | 7.1 ms |

#### SiLU — cannsim (`Ascend950`)

| Label | T | Sim wall | ms/element |
|-------|---|----------|------------|
| smoke | 128 | **42 s** | 331 ms |
| tiny | 512 | **15 s** | 30 ms |
| small | 1024 | **17 s** | 17 ms |
| varlen_2x512 | 1024 | **16 s** | 16 ms |
| medium | 4096 | **17 s** | 4.1 ms |

#### SwiGLU — msprof

| Label | T | Sim wall | ms/element |
|-------|---|----------|------------|
| smoke | 128 | **75 s** | 588 ms |
| tiny | 512 | **49 s** | 95 ms |
| small | 1024 | **61 s** | 59 ms |
| varlen_2x512 | 1024 | **47 s** | 46 ms |
| medium | 4096 | **52 s** | 13 ms |

#### SwiGLU — cannsim

| Label | T | Sim wall | ms/element |
|-------|---|----------|------------|
| smoke | 128 | **52 s** | 403 ms |
| tiny | 512 | **27 s** | 52 ms |
| small | 1024 | **29 s** | 28 ms |
| varlen_2x512 | 1024 | **21 s** | 21 ms |
| medium | 4096 | **22 s** | 5.4 ms |

**Scaling law (Kunpeng, approximate):**

- Fixed overhead **~15–75 s** at T=128 dominates smoke; do not extrapolate from smoke alone.
- After startup, cost scales **roughly linearly with T** at ~**0.005–0.06 s/element** on cannsim and ~**0.007–0.06 s/element** on msprof for T≥512.
- **Varlen vs fixed length** at the same T: negligible (1024 tokens: SiLU msprof 26 s vs 26 s).
- Pure-vector kernels finish in **minutes** on the default ladder; contrast with mix `chunk_h_mini` v1 (scalar matmul, 35+ min timeouts).

### AMD EPYC 9654 (x86_64)

#### SiLU — msprof (`Ascend950PR_9599`)

| Label | T | Sim wall | ms/element |
|-------|---|----------|------------|
| smoke | 128 | **12 s** | 91 ms |
| tiny | 512 | **7 s** | 14 ms |
| small | 1024 | **7 s** | 7.1 ms |
| varlen_2x512 | 1024 | **7 s** | 6.9 ms |
| medium | 4096 | **12 s** | 3.0 ms |

#### SiLU — cannsim (`Ascend950`)

| Label | T | Sim wall | ms/element |
|-------|---|----------|------------|
| smoke | 128 | **9 s** | 72 ms |
| tiny | 512 | **4 s** | 7.2 ms |
| small | 1024 | **4 s** | 3.9 ms |
| varlen_2x512 | 1024 | **4 s** | 3.5 ms |
| medium | 4096 | **5 s** | 1.3 ms |

#### SwiGLU — msprof

| Label | T | Sim wall | ms/element |
|-------|---|----------|------------|
| smoke | 128 | **18 s** | 137 ms |
| tiny | 512 | **13 s** | 25 ms |
| small | 1024 | **17 s** | 16 ms |
| varlen_2x512 | 1024 | **15 s** | 14 ms |
| medium | 4096 | **20 s** | 4.9 ms |

#### SwiGLU — cannsim

| Label | T | Sim wall | ms/element |
|-------|---|----------|------------|
| smoke | 128 | **11 s** | 87 ms |
| tiny | 512 | **6 s** | 12 ms |
| small | 1024 | **6 s** | 6.3 ms |
| varlen_2x512 | 1024 | **6 s** | 5.8 ms |
| medium | 4096 | **7 s** | 1.7 ms |

**Scaling law (AMD EPYC, approximate):**

- Fixed overhead **~9–18 s** at T=128 dominates smoke; do not extrapolate from smoke alone.
- After startup, cost scales **roughly linearly with T** at ~**0.001–0.012 s/element** on cannsim and ~**0.003–0.014 s/element** on msprof for T≥512.
- **Varlen vs fixed length** at the same T: negligible (1024 tokens: SiLU msprof 7 s vs 7 s).
- Pure-vector kernels finish in **under ~2 min** for the full ladder on this host.

### vs CPU thread count (OMP)

Measured on **Kunpeng-920** only (not re-run on x86).

Fixed workload **T=512** (SiLU), swept `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` together:

| OMP threads | msprof mean (s) | speedup vs 1 | cannsim mean (s) | speedup vs 1 |
|-------------|-----------------|--------------|------------------|--------------|
| 1 | 39.5 | 1.00× | 31.6 | 1.00× |
| 2 | 44.0 | 0.90× | 35.3 | 0.90× |
| 4 | 41.7 | 0.95× | 34.4 | 0.92× |
| 8 | 41.4 | 0.95× | 35.1 | 0.90× |
| 16 | 44.6 | 0.89× | 31.4 | 1.01× |
| 32 | 42.2 | 0.93× | 32.1 | 0.99× |

**Conclusion:** host OMP thread env vars change simulator wall time by at most **~±11%** (msprof) and **~±12%** (cannsim). Tuning `OMP_NUM_THREADS` is not an effective lever; PEM uses internal worker pools.

## Layout

```
examples/a5_sim/
├── kernels/silu_a5.cpp, swiglu_a5.cpp
├── vec_sim.py                 # driver (--kernel silu|swiglu)
├── common/build.py            # dav-c310-vec build
├── run_msprof.sh / run_cannsim.sh / run_thread_sweep.sh
├── configs/scale_ladder.json
└── outputs/                   # gitignored results
```

## References

- A5 PTO ST tests: `megagdn-pto/third_party/pto-isa/tests/npu/a5/src/st/testcase`
- A2 originals: `examples/jit_cpp/silu_dynamic`, `csrc/kernel/kernel_swiglu.cpp`
- Tool comparison: [`cannsim_vs_msprof.md`](cannsim_vs_msprof.md)
