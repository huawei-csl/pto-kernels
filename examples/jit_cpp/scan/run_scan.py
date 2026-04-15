import torch
import torch_npu  # noqa
from jit_util_scan import jit_compile, clean_up


def reference_scan(
    x: torch.Tensor, u_s: torch.Tensor, tile_size: int = 64
) -> torch.Tensor:
    """
    Torch reference implementation of the NPU scan algorithm.
    It uses matrix multiplication for inta-block (row-wise) prefix sums,
    followed by a sequential running sum accumulation across rows.
    Avoids using torch.cumsum.
    """
    total_len = x.numel()
    num_rows = total_len // tile_size

    # 1. Reshape x into blocks. Each block is a row.
    x_matrix = x.view(num_rows, tile_size)

    # u_s was transposed for the NPU kernel (DN layout). We transpose it back for standard Matmul
    U = u_s.t()

    # 2. Cube Phase: Matrix multiplication (row-wise prefix sum)
    # [num_rows, tile_size] @ [tile_size, tile_size] -> [num_rows, tile_size]
    row_sums = torch.matmul(x_matrix, U)

    # 3. Vector Phase: Sequential accumulation across rows
    result = torch.empty_like(row_sums)
    running_sum = torch.zeros((1,), device=x.device, dtype=x.dtype)

    for i in range(num_rows):
        current_row = row_sums[i] + running_sum
        result[i] = current_row
        running_sum = current_row[-1]

    return result.view(-1)


def test_scan(tile_size=64, n_tiles=64):
    total_len = tile_size * tile_size * n_tiles
    device = "npu:1"
    dtype = torch.float32
    torch.npu.set_device(device)
    torch.set_printoptions(sci_mode=False)

    # Prepare Inputs
    # x = torch.rand(size=(total_len,), device="npu", dtype=dtype).contiguous()
    x = torch.arange(1, total_len + 1, device="npu", dtype=dtype).contiguous()
    y = torch.zeros_like(x)

    # Generate upper triangular matrix of 1s (s x s) -> Transpose to represent Column-Major Down-Normal memory layout
    u_s = (
        torch.triu(torch.ones((tile_size, tile_size), device="npu", dtype=dtype))
        .t()
        .contiguous()
    )

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
        scan_func(x, y, u_s, total_len)
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
