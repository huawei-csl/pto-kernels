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

Current note:

- `scaled_dot_kkt` uses the PTO cube kernel for the `K @ K^T` workspace and an exact NPU Torch epilogue for the BSND/varlen coefficient application while the all-PTO vector epilogue is still being debugged. Correctness is covered; performance is not yet at the static-baseline target for this stage.
- `wy_fast` uses PTO cube kernels for the packed `A1 @ K` and `A2 @ V` matmuls, with exact NPU Torch packing/scaling used to build `A1/A2` from the dynamic BSND inputs. Correctness is covered; performance is not yet at the static-baseline target for this stage.
- `chunk_h` uses PTO cube kernels for the two dominant matmuls in the recurrence (`W @ S` and `K^T @ new_v`). The chunk-by-chunk recurrent sequencing is currently orchestrated on the host to keep the dynamic varlen path correct while the fully in-kernel recurrence is still being ported.
- `chunk_o` now runs as one fused cube+vector PTO kernel with cross-core synchronization (`qk`, `qs`, gated `qk`, `qkv`, and direct BSND output store are all kernel-side). The current standalone check passes both fixed-length and packed-varlen cases with FP16-stage tolerances.

Run the implemented stage checks with:

```bash
export PTO_LIB_PATH=/sources/pto-isa
python run_chunk_cumsum_dynamic_bsnd.py
python run_scaled_dot_kkt_dynamic_bsnd.py
python run_wy_fast_dynamic_bsnd.py
python run_chunk_h_dynamic_bsnd.py
python run_gated_delta_dynamic_bsnd.py
```
