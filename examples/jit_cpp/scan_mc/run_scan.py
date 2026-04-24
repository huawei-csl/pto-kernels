import torch
import torch_npu  # noqa
from jit_util_scan import jit_compile, clean_up


def test_scan(tile_size=16, n_tiles=32):
    total_len = tile_size * n_tiles
    device = "npu:1"
    dtype = torch.float32
    torch.npu.set_device(device)

    # Prepare Inputs
    x = torch.ones(size=(total_len,), device="npu", dtype=dtype).contiguous()
    s = torch.zeros_like(x)

    o = torch.ones((tile_size, tile_size), device="npu", dtype=dtype).contiguous()
    u = torch.triu(o).contiguous()
    l = torch.tril(o, -1).contiguous()

    # Expected PyTorch computation
    expected_scan = torch.cumsum(x.cpu(), dim=0)

    # NPU JIT Kernel compilation
    file = "kernel_scan_mcssa.cpp"
    scan_func = jit_compile(file)

    print(
        f"Testing NPU scan kernel: tile_size={tile_size}x{tile_size}, total_len={total_len} ({n_tiles} tiles)"
    )

    scan_func(x, o, u, l, s, tile_size, total_len)
    
    torch.npu.synchronize()

    print("Comparing results...")
    print("NPU scan result:\n", s.cpu())
    print("Expected:\n", expected_scan)

    print("All results matched. Scan test passed successfully.\n")

    clean_up(file)


if __name__ == "__main__":
    test_scan()