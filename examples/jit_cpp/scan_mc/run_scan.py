import torch
import torch_npu  # noqa
from jit_util_scan import jit_compile, clean_up


def test_scan(tile_size=16, n_tiles=6):
    total_len = tile_size * tile_size * n_tiles
    device = "npu:5"
    dtype = torch.float32
    torch.npu.set_device(device)

    # Prepare Inputs
    x = torch.ones(size=(total_len,), device="npu", dtype=dtype).contiguous()
    s = torch.zeros_like(x)

    ones = torch.ones((tile_size, tile_size), device="npu", dtype=dtype).contiguous()
    utri = torch.triu(ones).contiguous()
    ltri = torch.tril(ones, -1).contiguous()

    # Expected PyTorch computation
    expected_scan = torch.cumsum(x.cpu(), dim=0)

    # NPU JIT Kernel compilation
    file = "kernel_scan_mcssa.cpp"
    scan_func = jit_compile(file)

    print(
        f"Testing NPU scan kernel: tile_size={tile_size}x{tile_size}, total_len={total_len} ({n_tiles} tiles)"
    )

    scan_func(x, ones, utri, ltri, s, total_len, tile_size)

    torch.npu.synchronize()

    print("Comparing results...")
    print("NPU scan result:\n", s.cpu()[::128])
    print("Expected:\n", expected_scan[::128])

    assert torch.allclose(
        s.cpu(), expected_scan, rtol=1e-3, atol=1e-2
    ), "Scan results do not match expected values!"

    print("All results matched. Scan test passed successfully.\n")

    clean_up(file)


if __name__ == "__main__":
    test_scan()
