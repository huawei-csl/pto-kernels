import os
import ctypes
import torch

# Select device "cpu" or "npu"
DEVICE = "npu:0"

NUM_AI_CORES = int(getattr(torch.npu.get_device_properties("npu"), "cube_core_num", 20))


if __name__ == "__main__":

    try:
        lib_path = "build/lib/libkernel_abs.so"
        lib_path = os.path.abspath(lib_path)
        lib = ctypes.CDLL(lib_path)
        print(f"Loaded library from {lib_path}")

        lib.call_vabs_fp16.restype = None
        lib.call_vabs_fp16.argtypes = [
            ctypes.c_int,  # block_dim
            ctypes.c_void_p,  # stream
            ctypes.c_void_p,  # y
            ctypes.c_void_p,  # x
            ctypes.c_int,  # total_len
        ]
        stream_ptr = torch.npu.current_stream()._as_parameter_  # noqa

        def torch_to_ctypes(tensor):
            return ctypes.c_void_p(tensor.data_ptr())

        batch_dim = 8
        length = [batch_dim, 128]

        x = torch.randn(length, device="cpu", dtype=torch.float16).to(DEVICE)
        actual = torch.empty_like(x)
        expected = torch.abs(x)

        print(f"Input: {x[-20:]}")
        lib.call_vabs_fp16(
            batch_dim,
            stream_ptr,
            torch_to_ctypes(x),
            torch_to_ctypes(actual),
            x.numel(),
        )
        is_close = torch.allclose(actual, expected)
        print(f"Is all close? {is_close}")
        print(actual[0, :10])
        print(expected[0, :10])
    finally:
        del lib  # triggers dlclose in CPython
