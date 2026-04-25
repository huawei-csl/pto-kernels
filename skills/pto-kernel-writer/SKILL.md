---
name: pto-kernel-writer
description: Use when adding a new pto-kernels operator package or turning a kernel idea into repo-ready source, metadata, benchmark specs, adapters, and validation notes.
---

# PTO Kernel Writer

Use this skill when creating or reviewing a new kernel entry in pto-kernels.

## Workflow

1. Create `python/pto_kernels/ops/<category>/<kernel>/kernel.py`.
2. Add `meta.py` next to the kernel with category, name, status, configs,
   source notes, and current blockers.
3. Add `bench/specs/<category>/<kernel>.yaml` once there is a runnable benchmark
   contract.
4. Add both adapter stubs:
   - `bench/adapters/ptodsl/<category>/<kernel>.py`
   - `bench/adapters/ops_transformer/<category>/<kernel>.py`
5. Register the kernel in `bench/kernel_inventory.yaml`.
6. Update the README kernel inventory with name, category, description, and
   status.
7. Keep local tests NPU-free by default. Mark NPU/custom-op tests with
   `pytest.mark.npu`.
8. Verify with `make test`, then use `make check-env` and `make test-npu` only
   on a CANN/NPU host.

## Kernel Source Contract

- Keep PTO-DSL source sync-free unless an explicit barrier is semantically
  required.
- Prefer `enable_insert_sync=True` and let PTOAS insert synchronization.
- Put shape and tuning choices in metadata or environment-tuned helpers rather
  than hidden constants.
- Keep each package focused on one public kernel family entry. If one source
  file exposes forward/backward variants, keep them in one package and list both
  builders in `meta.py`.

## Template Files

Use `templates/kernel_writer/` as the starting point for new entries:

- `README.md`: checklist and file placement guide.
- `kernel.py.template`: minimal source structure.
- `meta.py.template`: metadata structure.
- `bench_spec.yaml.template`: benchmark spec structure.
- `adapter_ptodsl.py.template`: PTO adapter structure.
- `adapter_baseline.py.template`: baseline adapter structure.

## Validation Commands

```bash
make test
make check
make bootstrap
make check-env
make test-npu
```

Run the last two commands only on a host with a configured Ascend CANN
environment and available NPU.
