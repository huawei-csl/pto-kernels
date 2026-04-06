Triton baseline to compare with PTO kernel performance.

This directory contains a self-contained Triton-Ascend forward baseline for the
naive chunkwise linear-attention `chunk_o` path:

- `chunk_o.py`: fused forward kernel for the `HEAD_FIRST` layout `(B, H, T, D)`.
- `test_chunk_o.py`: forward correctness tests against a PyTorch reference.
- `benchmark_chunk_o.py`: measured comparison against the existing PTO C++ kernel.
- `performance_summary.md`: benchmark results captured on the current machine.

Current scope:

- Supports runtime-dynamic `B` and `L`.
- Keeps `D`, `V`, and `C` as compile-time constants for Triton codegen.
- Supports only the no-gating, fixed-length, `HEAD_FIRST` path.
- Supports optional `initial_state` and `output_final_state`.

Not implemented yet:

- `SEQ_FIRST` layout `(B, T, H, D)`.
- Gated variants (`USE_G`).
- Varlen / offsets-based execution.

Quick commands:

```bash
python -m pytest test_chunk_o.py -q
python benchmark_chunk_o.py --markdown-out performance_summary.md
```
