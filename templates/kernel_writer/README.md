# Kernel Writer Template

Use this template when adding a kernel to pto-kernels without changing the repo
layout.

## Files To Add

```text
python/pto_kernels/ops/<category>/<kernel>/kernel.py
python/pto_kernels/ops/<category>/<kernel>/meta.py
bench/specs/<category>/<kernel>.yaml
bench/adapters/ptodsl/<category>/<kernel>.py
bench/adapters/ops_transformer/<category>/<kernel>.py
```

Register the kernel in `bench/kernel_inventory.yaml` and update the README
kernel inventory row with name, category, description, and status.

## Status Checklist

- `planned`: inventory entry exists, implementation not ready.
- `local_ptoas`: PTO-DSL source emits PTOAS output locally.
- `validated`: correctness has been checked on a CANN/NPU host.
- `parity`: correctness and performance parity have been checked against the
  baseline adapter.

## Commands

```bash
make test
make check
make check-env
make test-npu
```
