"""Run vabs_fp16 via JIT-compiled kernel + torch_npu tensors (a5 example)."""

import numpy as np
import torch
import torch_npu  # noqa: F401

from jit_util_a5_abs import DEFAULT_BLOCK_DIM, jit_compile

# Matches examples/a5/main_abs.cpp
VABS_SHAPE = (8, 128)
VABS_NUM_ELEMENTS = VABS_SHAPE[0] * VABS_SHAPE[1]
DEVICE = "npu:0"


def make_input_x():
    """Same data as scripts/data_gen_abs.py (seed=42)."""
    rng = np.random.default_rng(seed=42)
    return rng.uniform(-100, 100, VABS_SHAPE).astype(np.float16)


def main():
    torch.npu.config.allow_internal_format = False
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(DEVICE)

    x_np = make_input_x()
    # Allocate on CPU then copy to NPU (same pattern as tests/test_abs.py).
    x = torch.from_numpy(x_np)
    x = x.npu()
    z = torch.empty_like(x)

    print(f"[vabs] shape={VABS_SHAPE}, blockDim={DEFAULT_BLOCK_DIM}")
    print("Compiling kernel_abs.cpp ...")
    vabs = jit_compile(verbose=True)

    vabs(x, z, VABS_NUM_ELEMENTS, block_dim=DEFAULT_BLOCK_DIM)
    torch.npu.synchronize()

    # msprof CA simulator models pipeline timing only, not numeric results.
    print("Input X (first 16):", x.flatten()[:16].cpu())
    print("Output Z (first 16):", z.flatten()[:16].cpu())
    print("vabs_fp16 kernel launch completed.")


if __name__ == "__main__":
    main()
