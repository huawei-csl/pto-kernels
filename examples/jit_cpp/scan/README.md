# Single core prefix sum (scan)

An implementation of prefix sum (scan) algorithm, based on https://arxiv.org/abs/2505.15112v1. Only single core algorithm is implemented (ScanU from the paper).

Usage:

```bash

# Optional
export PTO_LIB_PATH=${ASCEND_TOOLKIT_HOME}  # reuse CANN 8.5.0 headers

# Run scan tests
python ./run_scan.py
