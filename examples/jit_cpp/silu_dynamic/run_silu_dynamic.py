import random
import torch
import torch_npu  # noqa

from jit_util_silu import jit_compile


def silu_ref(x):
    return x * torch.sigmoid(x)


def random_2d_shape(
    min_m=1,
    max_m=2048,
    min_n=1,
    max_n=2048,
):
    m = random.randint(min_m, max_m)
    n = random.randint(min_n, max_n)
    return [m, n]


def test_silu_random_2d(num_tests=20):
    device = "npu"
    dtype = torch.float16
    torch.npu.set_device(device)

    silu_func = jit_compile("silu_dynamic.cpp")

    for i in range(num_tests):
        shape = random_2d_shape()

        x = torch.rand(shape, device=device, dtype=dtype)
        y = torch.empty_like(x)

        silu_func(y, x)
        torch.npu.synchronize()

        y_ref = silu_ref(x)

        torch.testing.assert_close(y, y_ref, rtol=0.1, atol=1e-5)

        print(f"Test {i+1}/{num_tests} passed — shape: {shape}")

    print("\nAll 2D random shape SILU tests passed!")


if __name__ == "__main__":
    test_silu_random_2d(num_tests=100)
