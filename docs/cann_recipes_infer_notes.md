# CANN Recipes Infer Notes

This repo pins `cann-recipes-infer` as an external reference under
`external/src/cann-recipes-infer`.

- URL: `https://gitcode.com/cann/cann-recipes-infer.git`
- Commit: `377f20f62d86b3da882b5084b46e02c735e619a3`

## What It Is

`cann-recipes-infer` is a CANN platform recipe repository for LLM and
multimodal inference on Ascend Atlas hardware. Its examples combine model
scripts, runtime configuration, custom operator directories, performance notes,
and agent skills for end-to-end inference optimization.

The cloned tree uses these broad areas:

- `models/`: model-specific inference examples, launch scripts, requirements,
  and runner code.
- `executor/`: shared model runner, model loading, graph, stream, HCCL, and
  profiling helpers.
- `module/`: reusable model-side modules such as quantization, MoE/GMM, linear,
  sparse, and sequence-parallel utilities.
- `ops/`: custom operator examples grouped by AscendC, PyPTO, Python PyPTO, and
  TileLang.
- `.agent/`: inference-optimization agent roles and skills with analyzer,
  implementer, reviewer, and stage-gated validation patterns.

## Useful Patterns For pto-kernels

- Keep examples reproducible with a short README, explicit requirements, an
  environment setup script, and one obvious run command.
- Keep custom operator work in a dedicated operator tree instead of mixing it
  into model code.
- Separate analysis, implementation, and verification guidance for agent-driven
  work.
- Treat performance work as validation-gated: establish a baseline, change one
  kernel or runtime path, then capture correctness and timing evidence.
- Document hardware/toolchain assumptions close to the command that needs them.
- Prefer small reusable helpers for runtime setup, stream/HCCL coordination, and
  profiling instead of copy-pasting them into each example.

## What Not To Import

- Do not copy model-specific inference runners or model implementations into
  pto-kernels.
- Do not adopt the `models/`, `executor/`, or `.agent/` directory layout; this
  repo keeps kernel source under `python/pto_kernels/ops/<category>/<kernel>/`.
- Do not add large datasets, model weights, generated binaries, or recipe output
  artifacts to git.
- Do not make the default local test path depend on CANN, torch-npu, or an
  attached NPU.

## Local Use

Clone or refresh the pinned external reference with:

```bash
make bootstrap
```

Then inspect:

```bash
external/src/cann-recipes-infer/README.md
external/src/cann-recipes-infer/CONTRIBUTION.md
external/src/cann-recipes-infer/.agent/README.md
external/src/cann-recipes-infer/ops/
```
