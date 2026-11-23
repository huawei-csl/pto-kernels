# Paged Attention HighPerf

JIT demo for a PTO-ISA paged-attention decode kernel using PyTorch/NPU tensors.

The example includes:

| File | Purpose |
|---|---|
| pa_kernel.cpp | host-callable JIT entry point |
| pa_entry.hpp | AIC/AIV dispatch wrapper |
| pa_kernel_impl.hpp | PTO kernel implementation |
| pa_tiling_struct.hpp | tiling type definitions |
| pa_tiling.py | Python tiling/workspace construction |
| pa_compile_and_run.py | correctness smoke test |
| pa_benchmark.py | benchmark driver |
| jit_util_pa.py | JIT compile and ctypes wrapper |

## Requirements

Set the PTO-ISA include root and CANN toolkit path if they are not already in the environment:

    export PTO_LIB_PATH=/path/to/pto-isa
    export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.0.0

## Run

    cd examples/jit_cpp/paged_attention_highperf
    python3 pa_compile_and_run.py
    python3 pa_benchmark.py --device npu:0 --shape b=8,s=4096 --check --warmup 1 --iters 1

The full benchmark sweep can be run with:

    python3 pa_benchmark.py --device npu:0

Use --check for correctness validation against the Python/PyTorch reference. For very large shapes, the reference can dominate runtime and memory use.
