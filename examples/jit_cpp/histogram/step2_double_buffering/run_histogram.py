import random
import torch
import torch_npu  # noqa

from jit_util_histogram import jit_compile


def test_histogram(size_mult=1, repeat_runs=20):
    device = "npu:1"
    dtype = torch.float32
    torch.npu.set_device(device)

    # Tile size is fixed in the kernel
    tile_size = 512
    num_cores = 1  # torch.npu.get_device_properties().vector_core_num
    num_tiles = num_cores * tile_size
    total_len = num_tiles * size_mult

    bins = 256
    min_val = 0.0
    max_val = 256.0

    # Create an input tensor bounded around our standard test range
    x = torch.rand(size=(total_len,), device="npu", dtype=dtype).contiguous() * max_val
    z = torch.zeros(bins, device="npu", dtype=torch.int32)

    # Golden PyTorch implementation
    expected_hist = torch.histc(x.cpu(), bins, min=min_val, max=max_val).to(torch.int32)

    hist_func = jit_compile("kernel_histogram.cpp")

    # NPU kernel execution, test to see if any race conditions occur across multiple runs
    actual_hist = []
    for _ in range(repeat_runs):
        z.zero_()
        hist_func(x, z, bins, min_val, max_val, block_dim=num_cores)
        actual_hist.append(z.cpu().clone())

    torch.npu.synchronize()

    # Check for consistency across runs and correctness against the expected count
    for i, hist in enumerate(actual_hist):
        assert torch.equal(
            hist, actual_hist[0]
        ), f"Inconsistent results across runs at run {i}"

    assert torch.equal(
        expected_hist, actual_hist[0]
    ), "Mismatch between expected and actual histogram"


if __name__ == "__main__":
    test_histogram(size_mult=64, repeat_runs=20)
