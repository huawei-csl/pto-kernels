# Dynamic BSND GatedDeltaNet

This directory contains a stage-by-stage PTO-ISA port of GatedDeltaNet for native BSND inputs (`[batch, seq, head, hidden]`) and optional packed varlen inputs driven by `cu_seqlens`.

Compared with `../static_baseline`, this path removes fixed `B/H/L` assumptions from the runtime ABI:

- `batch` and `seq_len` are runtime parameters
- packed varlen BSND is supported through `cu_seqlens`
- inputs stay in native BSND layout without PyTorch-side transpose
- stage kernels are being ported one-by-one so correctness and performance can be checked independently

Implemented today:

- `chunk_cumsum_kernel.cpp`
- `scaled_dot_kkt_kernel.cpp`
- `wy_fast_kernel.cpp`
- `chunk_h_kernel.cpp`
- `chunk_o_kernel.cpp`

Current status:

- All stage checks in `run_gated_delta_dynamic_bsnd.py` currently pass for both fixed-length BSND inputs and packed-varlen BSND inputs where applicable.
- `chunk_cumsum` is native PTO vector code and passes its fixed and packed-varlen checks.
- `scaled_dot_kkt` runs through one fused PTO cube+vector kernel. The coefficient build, masking, and packed output store are all kernel-side, and the stage check passes on both fixed and packed-varlen inputs.
- `wy_fast` is still hybrid. The packed `A1 @ K` and `A2 @ V` matmuls are PTO cube kernels, but the dynamic BSND packing/scaling for `A1/A2` still falls back to exact NPU Torch helper code for correctness. The stage check passes, but this stage is not yet fully de-hybridized and is still far slower than the static reference.
- `chunk_h` is still hybrid. The dominant `W @ S` and `K^T @ new_v` matmuls use PTO cube kernels, but the chunk-by-chunk recurrence and final state propagation are still orchestrated on the host. The stage check passes for fixed and packed-varlen inputs.
- `chunk_o` runs as one fused PTO cube+vector kernel with cross-core synchronization. `qk`, `qs`, gated `qk`, `qkv`, and direct BSND output store are all kernel-side, and the stage check passes on both fixed and packed-varlen inputs with FP16-stage tolerances.

Latest stage-check outputs from `run_gated_delta_dynamic_bsnd.py`:

- `chunk_cumsum`: fixed `0.062 ms`, packed-varlen `0.058 ms`
- `scaled_dot_kkt`: fixed `0.067 ms, 0.50 TFLOP/s`, packed-varlen `0.065 ms, 0.39 TFLOP/s`
- `wy_fast`: fixed `2.400 ms, 0.03 TFLOP/s`, packed-varlen `1.945 ms, 0.03 TFLOP/s`
- `chunk_h`: fixed `5.204 ms`, packed-varlen `4.057 ms`
- `chunk_o`: fixed `0.184 ms, 0.36 TFLOP/s`, packed-varlen `0.184 ms, 0.27 TFLOP/s`

Important caveats:

- The current driver is a stage-validation suite, not a fully native end-to-end GDN kernel chain.
- `wy_fast` and `chunk_h` still rely on Torch-side fallback/orchestration for correctness.
- The dynamic kernels remain much slower than the original static kernels, so correctness is ahead of performance at the moment.

Run the implemented stage checks with:

```bash
export PTO_LIB_PATH=/sources/pto-isa
python run_chunk_cumsum_dynamic_bsnd.py
python run_scaled_dot_kkt_dynamic_bsnd.py
python run_wy_fast_dynamic_bsnd.py
python run_chunk_h_dynamic_bsnd.py
python run_chunk_o_dynamic_bsnd.py
python run_gated_delta_dynamic_bsnd.py
```
