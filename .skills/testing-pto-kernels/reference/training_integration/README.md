# Training Integration TODO

This task intentionally does not verify torch autograd, TorchTitan, or training-loop integration because the current environment does not provide those stacks.

Future work should add:

1. A pybind/CMake PTO forward kernel.
2. A verified backward kernel or a documented PyTorch fallback.
3. A `torch.autograd.Function` wrapper.
4. A tiny training step that checks gradients and loss movement.

Until then, do not claim training integration from this skill.
