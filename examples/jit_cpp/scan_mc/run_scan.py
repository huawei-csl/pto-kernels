import torch
import torch_npu  # noqa
from jit_util_scan import jit_compile, clean_up
import pytest
import os

# get npu device from NPU_DEVICE env variable, default to npu:1 if not set
device = os.getenv("NPU_DEVICE", "npu:1")


@pytest.mark.parametrize("tile_size", [16, 32, 64, 128])
@pytest.mark.parametrize("n_tiles", [10, 20])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=str)
def test_scan(tile_size: int, n_tiles: int, dtype: torch.dtype):
    total_len = tile_size * tile_size * n_tiles
    torch.npu.set_device(device)

    # Prepare Inputs
    x = torch.ones(size=(total_len,), device="npu", dtype=dtype).contiguous()
    s = torch.zeros(size=(total_len,), device="npu", dtype=torch.float32).contiguous()

    ones = torch.ones((tile_size, tile_size), device="npu", dtype=dtype).contiguous()
    utri = torch.triu(ones)
    ltri = torch.tril(ones, -1)

    # Expected PyTorch computation
    expected_scan = torch.cumsum(x.cpu().to(torch.float32), dim=0)

    # NPU JIT Kernel compilation
    file = "kernel_scan_mcssa.cpp"
    scan_func = jit_compile(file)

    print(
        f"Testing NPU scan kernel: tile_size={tile_size}x{tile_size}, total_len={total_len} ({n_tiles} tiles)"
    )

    scan_func(x, ones, utri, ltri, s, total_len, tile_size=tile_size)

    torch.npu.synchronize()

    # print("Comparing results...")
    # print("NPU scan result:\n", s.cpu())
    # print("Expected:\n", expected_scan)

    assert torch.allclose(
        s.cpu(), expected_scan, rtol=1e-3, atol=1e-2
    ), "Scan results do not match expected values!"

    # print("All results matched. Scan test passed successfully.\n")

    clean_up(file)


if __name__ == "__main__":
    test_scan(16, 14, torch.float32)
