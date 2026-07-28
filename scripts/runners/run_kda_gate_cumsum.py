import os
import ctypes
import torch

# Select device "cpu" or "npu"
DEVICE = "npu:0"

# The gate-cumsum kernel runs on the vector cores.
NUM_AI_CORES = int(
    getattr(torch.npu.get_device_properties("npu"), "vector_core_num", 40)
)

# Compile-time kernel constants (default build: GDN_H=16, GDN_D=128, GDN_C=128)
CHUNK = 128
D = 128  # key/gate dimension per head
H = 16  # number of heads


def ref_kda_gate_cumsum(g, batch_size, seq_len):
    """CPU fp32 reference: per-chunk prefix sum of per-dim gates. g: [T, H, D]."""
    g_f32 = g.float()
    out = torch.zeros_like(g_f32)
    for b in range(batch_size):
        bos = b * seq_len
        for j in range(0, seq_len, CHUNK):
            s = bos + j
            e = min(s + CHUNK, bos + seq_len)
            out[s:e] = g_f32[s:e].cumsum(dim=0)
    return out


if __name__ == "__main__":

    try:
        lib_path = "build/lib/libkernel_kda_gate_cumsum.so"
        lib_path = os.path.abspath(lib_path)
        lib = ctypes.CDLL(lib_path)
        print(f"Loaded library from {lib_path}")

        lib.pto_launch_kda_gate_cumsum.restype = None
        lib.pto_launch_kda_gate_cumsum.argtypes = [
            ctypes.c_uint32,  # blockDim
            ctypes.c_void_p,  # stream
            ctypes.c_void_p,  # g
            ctypes.c_void_p,  # g_sum (out)
            ctypes.c_void_p,  # cu_seqlens (nullptr => fixed-length path)
            ctypes.c_int64,  # batch_size
            ctypes.c_int64,  # seq_len
        ]
        stream_ptr = torch.npu.current_stream()._as_parameter_  # noqa

        def torch_to_ctypes(tensor):
            return ctypes.c_void_p(tensor.data_ptr())

        batch_size = 1
        seq_len = 256
        T = batch_size * seq_len

        torch.manual_seed(42)
        # Production-like gate magnitude: per-dim log-gates in (-1, 0).
        g_cpu = -torch.rand(T, H, D, dtype=torch.float16)
        expected = ref_kda_gate_cumsum(g_cpu, batch_size, seq_len)

        G_npu = g_cpu.contiguous().to(DEVICE)  # [T, H, D] fp16
        # Output is fp32: the kernel accumulates in fp32 to avoid fp16 drift.
        G_sum = torch.zeros(T, H, D, device=DEVICE, dtype=torch.float32)

        block_dim = NUM_AI_CORES

        lib.pto_launch_kda_gate_cumsum(
            block_dim,
            stream_ptr,
            torch_to_ctypes(G_npu),
            torch_to_ctypes(G_sum),
            None,  # cu_seqlens
            batch_size,
            seq_len,
        )
        torch.npu.synchronize()

        actual = G_sum.cpu()

        print(f"g_sum max abs diff: {(actual - expected).abs().max().item()}")
        print(f"Is all close? {torch.allclose(actual, expected, rtol=1e-4, atol=1e-5)}")
        print(actual[:4, 0, 0])
        print(expected[:4, 0, 0])
    finally:
        del lib  # triggers dlclose in CPython
