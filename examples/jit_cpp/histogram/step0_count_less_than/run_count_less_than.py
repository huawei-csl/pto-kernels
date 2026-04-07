import random
import torch
import torch_npu  # noqa

from jit_util_count_less_than import jit_compile


def hist_ref(x, bins):
    return torch.histc(x, bins=bins)


def random_2d_shape(
    min_m=1,
    max_m=2048,
    min_n=1,
    max_n=2048,
):
    m = random.randint(min_m, max_m)
    n = random.randint(min_n, max_n)
    return [m, n]


def test_count_less_than(size_mult=1, repeat_runs=20, use_atomic_impl=False):
    device = "npu:1"
    dtype = torch.float32
    torch.npu.set_device(device)

    # Tile size is fixed in the kernel
    tile_size = 512
    num_cores = torch.npu.get_device_properties(0).vector_core_num
    num_tiles = num_cores * tile_size
    total_len = num_tiles * size_mult

    # Create an input tensor bounded around our standard test pivots
    x = torch.rand(size=(total_len,), device="npu", dtype=dtype).contiguous()
    z = torch.zeros(1, device="npu", dtype=torch.int32)
    pivot = torch.rand(1).item()

    # Golden PyTorch implementation
    expected_count = (x < pivot).sum().to(torch.int32).cpu().item()

    if use_atomic_impl:
        count_func = jit_compile("kernel_count_less_than_atomic.cpp")
    else:
        count_func = jit_compile("kernel_count_less_than.cpp")

    # NPU kernel execution, test to see if any race conditions occur across multiple runs
    actual_count = []
    for _ in range(repeat_runs):
        count_func(x, z, pivot, block_dim=num_cores)
        actual_count.append(z.item())

    torch.npu.synchronize()

    # Check for consistency across runs and correctness against the expected count
    assert len(set(actual_count)) == 1, "Inconsistent results across runs"
    assert (
        expected_count == actual_count[0]
    ), f"Mismatch: expected {expected_count}, got {actual_count}"


if __name__ == "__main__":
    # Atomic implementation has problems
    # test_count_less_than(size_mult=64, repeat_runs=20, use_atomic_impl=True)
    test_count_less_than(size_mult=64, repeat_runs=20, use_atomic_impl=False)
