import os
import ctypes
import torch

# Select device "cpu" or "npu"
DEVICE = "npu:0"

# The host wrapper (csrc/host/torch_simple_matmul.h) always launches on one core.
BLOCK_DIM = 1


if __name__ == "__main__":

    try:
        lib_path = "build/lib/libkernel_simple_matmul.so"
        lib_path = os.path.abspath(lib_path)
        lib = ctypes.CDLL(lib_path)
        print(f"Loaded library from {lib_path}")

        lib.pto_launch_simple_matmul_fp16.restype = None
        lib.pto_launch_simple_matmul_fp16.argtypes = [
            ctypes.c_uint32,  # blockDim
            ctypes.c_void_p,  # stream
            ctypes.c_void_p,  # a
            ctypes.c_void_p,  # b
            ctypes.c_void_p,  # c
            ctypes.c_uint32,  # matrix_size
        ]
        stream_ptr = torch.npu.current_stream()._as_parameter_  # noqa

        def torch_to_ctypes(tensor):
            return ctypes.c_void_p(tensor.data_ptr())

        matrix_size = 128

        a = torch.rand((matrix_size, matrix_size), device="cpu", dtype=torch.float16)
        b = torch.rand((matrix_size, matrix_size), device="cpu", dtype=torch.float16)

        a_npu = a.to(DEVICE)
        b_npu = b.to(DEVICE)
        # The kernel accumulates in fp32, so C is fp32 even for fp16 inputs.
        c_npu = torch.zeros(
            (matrix_size, matrix_size), device=DEVICE, dtype=torch.float32
        )

        lib.pto_launch_simple_matmul_fp16(
            BLOCK_DIM,
            stream_ptr,
            torch_to_ctypes(a_npu),
            torch_to_ctypes(b_npu),
            torch_to_ctypes(c_npu),
            matrix_size,
        )
        torch.npu.synchronize()

        actual = c_npu.cpu()
        expected = torch.matmul(a.float(), b.float())

        is_close = torch.allclose(actual, expected)
        print(f"Is all close? {is_close}")
        print(f"Max abs diff: {(actual - expected).abs().max().item()}")
        print(actual[0, :10])
        print(expected[0, :10])
    finally:
        del lib  # triggers dlclose in CPython
