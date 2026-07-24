import os
import ctypes
import torch

# Select device "cpu" or "npu"
DEVICE = "npu:0"

# The cumsum kernel runs on the vector cores.
NUM_AI_CORES = int(
    getattr(torch.npu.get_device_properties("npu"), "vector_core_num", 40)
)

# Compile-time kernel constants (default build: GDN_H=16, GDN_C=128)
C = 128  # chunk size
H = 16  # number of heads


def ref_gdn_chunk_cumsum(g, batch_size, seq_len):
    """Chunk-local cumulative sum of gates (CPU fp32 reference). g: [T, H]."""
    out = torch.zeros_like(g, dtype=torch.float32)
    gf = g.float()
    for b in range(batch_size):
        bos = b * seq_len
        for j in range(0, seq_len, C):
            s = bos + j
            e = min(s + C, bos + seq_len)
            out[s:e] = gf[s:e].cumsum(dim=0)
    return out


if __name__ == "__main__":

    try:
        lib_path = "build/lib/libkernel_gdn_chunk_cumsum.so"
        lib_path = os.path.abspath(lib_path)
        lib = ctypes.CDLL(lib_path)
        print(f"Loaded library from {lib_path}")

        lib.pto_launch_gdn_chunk_cumsum_fp32.restype = None
        lib.pto_launch_gdn_chunk_cumsum_fp32.argtypes = [
            ctypes.c_uint32,  # blockDim
            ctypes.c_void_p,  # stream
            ctypes.c_void_p,  # G     (in)
            ctypes.c_void_p,  # G_sum (out)
            ctypes.c_void_p,  # cu_seqlens (nullptr => fixed-length path)
            ctypes.c_int64,  # batch_size
            ctypes.c_int64,  # seq_len
        ]
        stream_ptr = torch.npu.current_stream()._as_parameter_  # noqa

        def torch_to_ctypes(tensor):
            return ctypes.c_void_p(tensor.data_ptr())

        torch.manual_seed(42)
        batch_size = 1
        seq_len = 256
        T = batch_size * seq_len

        g_cpu = torch.randn(T, H, dtype=torch.float32)
        G = g_cpu.to(DEVICE)

        block_dim = NUM_AI_CORES
        G_sum = torch.zeros(T, H, device=DEVICE, dtype=torch.float32)

        lib.pto_launch_gdn_chunk_cumsum_fp32(
            block_dim,
            stream_ptr,
            torch_to_ctypes(G),
            torch_to_ctypes(G_sum),
            None,  # cu_seqlens
            batch_size,
            seq_len,
        )
        torch.npu.synchronize()

        expected = ref_gdn_chunk_cumsum(g_cpu, batch_size, seq_len)
        actual = G_sum.cpu()

        print(f"G_sum max abs diff: {(actual - expected).abs().max().item()}")
        print(f"Is all close? {torch.allclose(actual, expected, rtol=1e-4, atol=1e-5)}")
        print(actual[:8, 0])
        print(expected[:8, 0])
    finally:
        del lib  # triggers dlclose in CPython
