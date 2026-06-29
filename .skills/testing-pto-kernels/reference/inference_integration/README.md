# Inference Integration TODO

This task intentionally does not verify vLLM, SGLang, or model-serving integration because the current environment does not provide those stacks.

Future work should add a small verified path that:

1. Builds a PTO kernel as a pybind/CMake deliverable.
2. Loads it from the target inference runtime.
3. Replaces one model operator with the PTO kernel.
4. Runs a numeric end-to-end check and a small latency benchmark.

Until then, do not claim whole-model inference integration from this skill.
