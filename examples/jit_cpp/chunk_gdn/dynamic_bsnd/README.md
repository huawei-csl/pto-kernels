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
- `wy_fast` runs as one fused PTO cube+vector kernel. The `A1 = A * (exp(g) * beta)` and `A2 = A * beta` coefficient builds use `TROWEXPANDMUL` for row-wise scaling, and the packed `A1 @ K` / `A2 @ V` matmuls are all kernel-side. The stage check passes on both fixed and packed-varlen inputs.
- `chunk_h` runs as one fused PTO cube+vector kernel with cross-core synchronization. The chunk-by-chunk recurrence (`state = state * exp(g_last) + K^T @ new_v`) is fully kernel-side with sequential chunks processed per (seq, head) work item. The stage check passes for fixed and packed-varlen inputs.
- `chunk_o` runs as one fused PTO cube+vector kernel with cross-core synchronization. `qk`, `qs`, gated `qk`, `qkv`, and direct BSND output store are all kernel-side, and the stage check passes on both fixed and packed-varlen inputs with FP16-stage tolerances.

Latest stage-check outputs from `run_gated_delta_dynamic_bsnd.py`:

- `chunk_cumsum`: fixed `0.064 ms`, packed-varlen `0.063 ms`
- `scaled_dot_kkt`: fixed `0.066 ms, 0.51 TFLOP/s`, packed-varlen `0.065 ms, 0.39 TFLOP/s`
- `wy_fast`: fixed `0.167 ms, 0.40 TFLOP/s`, packed-varlen `0.167 ms, 0.30 TFLOP/s`
- `chunk_h`: fixed `0.144 ms`, packed-varlen `0.146 ms`
- `chunk_o`: fixed `0.197 ms, 0.34 TFLOP/s`, packed-varlen `0.199 ms, 0.25 TFLOP/s`

Important caveats:

- The current driver is a stage-validation suite, not a fully native end-to-end GDN kernel chain.
- All five stages (`chunk_cumsum`, `scaled_dot_kkt`, `wy_fast`, `chunk_h`, `chunk_o`) are now fully fused PTO kernels with no Torch fallback.

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
