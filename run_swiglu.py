import os
import ctypes
import torch
import torch_npu  # noqa: F401  (registers the "npu" device with torch)


def swiglu_ref(x):
    gate, up = torch.chunk(x.float(), 2, dim=-1)
    return (gate * torch.sigmoid(gate) * up).to(x.dtype)


if __name__ == "__main__":

    try:
        lib_path = "build/lib/libkernel_swiglu.so"
        lib_path = os.path.abspath(lib_path)
        lib = ctypes.CDLL(lib_path)
        print(f"Loaded library from {lib_path}")

        # extern "C" void pto_launch_swiglu_fp16(uint32_t blockDim, void* stream,
        #     void* x, void* y, uint32_t batch, uint32_t input_n);
        lib.pto_launch_swiglu_fp16.restype = None
        lib.pto_launch_swiglu_fp16.argtypes = [
            ctypes.c_uint32,  # blockDim
            ctypes.c_void_p,  # stream
            ctypes.c_void_p,  # x  [batch, input_n]
            ctypes.c_void_p,  # y  [batch, input_n / 2]
            ctypes.c_uint32,  # batch
            ctypes.c_uint32,  # input_n (= 2 * N)
        ]
        stream_ptr = torch.npu.current_stream()._as_parameter_  # noqa

        def torch_to_ctypes(tensor):
            return ctypes.c_void_p(tensor.data_ptr())

        block_dim = 4
        batch = 4
        n = 64  # output hidden dim; input is [batch, 2 * n]
        input_n = 2 * n

        # SwiGLU interprets x as [gate | up] along the last dim.
        x = torch.randn(batch, input_n, device="cpu", dtype=torch.float16).to("npu:0")
        y = torch.empty(batch, n, device=x.device, dtype=torch.float16)

        lib.pto_launch_swiglu_fp16(
            block_dim,
            stream_ptr,
            torch_to_ctypes(x),
            torch_to_ctypes(y),
            batch,
            input_n,
        )
        torch.npu.synchronize()

        expected = swiglu_ref(x.cpu())
        torch.testing.assert_close(y.cpu(), expected, rtol=1e-2, atol=1e-5)
        print("OK: swiglu kernel output matches reference.")
    finally:
        del lib  # triggers dlclose in CPython
