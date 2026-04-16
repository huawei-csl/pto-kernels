import torch
import torch_npu  # noqa
from jit_util_scan import jit_compile, clean_up


def test_scan(tile_size=64, n_tiles=64):
    total_len = tile_size * tile_size * n_tiles
    device = "npu:1"
    dtype = torch.float32
    torch.npu.set_device(device)

    # Prepare Inputs
    x = torch.rand(size=(total_len,), device="npu", dtype=dtype).contiguous()
    y = torch.zeros_like(x)

    # Generate upper triangular matrix of 1s (s x s)
    u = torch.triu(
        torch.ones((tile_size, tile_size), device="npu", dtype=dtype)
    ).contiguous()

    # Expected PyTorch computation
    expected_scan = torch.cumsum(x.cpu(), dim=0)

    # NPU JIT Kernel compilation
    scan_func = jit_compile("kernel_scan_single_core.cpp", tile_size=tile_size)

    print(
        f"Testing NPU scan kernel: tile_size={tile_size}x{tile_size}, total_len={total_len} ({n_tiles} tiles)"
    )

    # NPU JIT Kernel execution
    repeat_runs = 10
    actual_scan = []
    for _ in range(repeat_runs):
        y.zero_()
        scan_func(x, y, u, total_len)
        actual_scan.append(y.cpu().clone())

    torch.npu.synchronize()

    # Check for consistency across runs and correctness against the expected count
    repeat_results = []
    for i, scan in enumerate(actual_scan):
        are_close = torch.allclose(scan, expected_scan, rtol=1e-3, atol=1e-3)
        if not are_close:
            unequal_count = torch.sum(scan != expected_scan)
        else:
            unequal_count = 0
        repeat_results.append([are_close, unequal_count])

    has_mismatch = any(not eq for eq, _ in repeat_results)
    if has_mismatch:
        print("Expected:\n", expected_scan[-10:])
        for i, result in enumerate(repeat_results):
            eq, count = result
            if not eq:
                print(
                    f"Inconsistent results run {i}, different elements: {count}/{total_len}. Sample:"
                )
                print(actual_scan[i][-10:])
        raise AssertionError(
            f"Scan mismatch for tile_size={tile_size}, total_len={total_len} ({n_tiles} tiles)"
        )

    print("All results matched. Scan test passed successfully.\n")

    clean_up("kernel_scan_single_core.cpp", tile_size)


if __name__ == "__main__":
    test_scan(tile_size=16, n_tiles=1)
    test_scan(tile_size=16, n_tiles=16)
    test_scan(tile_size=16, n_tiles=64)
    test_scan(tile_size=16, n_tiles=100)

    test_scan(tile_size=32, n_tiles=1)
    test_scan(tile_size=32, n_tiles=16)
    test_scan(tile_size=32, n_tiles=64)
    test_scan(tile_size=32, n_tiles=100)

    test_scan(tile_size=64, n_tiles=1)
    test_scan(tile_size=64, n_tiles=16)
    test_scan(tile_size=64, n_tiles=64)
    test_scan(tile_size=64, n_tiles=100)

    test_scan(tile_size=128, n_tiles=1)
    test_scan(tile_size=128, n_tiles=16)
    test_scan(tile_size=128, n_tiles=64)
    test_scan(tile_size=128, n_tiles=100)
