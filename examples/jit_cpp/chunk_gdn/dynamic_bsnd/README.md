# Dynamic BSND GatedDeltaNet

This directory contains a stage-by-stage PTO-ISA port of GatedDeltaNet for native BSND inputs (`[batch, seq, head, hidden]`) and optional packed varlen inputs driven by `cu_seqlens`.

Compared with `../static_baseline`, this path removes fixed `B/H/L` assumptions from the runtime ABI:

- `batch` and `seq_len` are runtime parameters
- packed varlen BSND is supported through `cu_seqlens`
- inputs stay in native BSND layout without PyTorch-side transpose
- stage kernels are being ported one-by-one so correctness and performance can be checked independently

Implemented today:

- `chunk_cumsum_kernel.cpp`

Run the implemented stage checks with:

```bash
export PTO_LIB_PATH=/sources/pto-isa
python run_chunk_cumsum_dynamic_bsnd.py
python run_gated_delta_dynamic_bsnd.py
```
