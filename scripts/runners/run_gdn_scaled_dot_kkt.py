import os
import ctypes
import torch
import torch.nn.functional as F

# Select device "cpu" or "npu"
DEVICE = "npu:0"

NUM_AI_CORES = int(getattr(torch.npu.get_device_properties("npu"), "cube_core_num", 20))

# Compile-time kernel constants (default build: GDN_H=16, GDN_HG=16, GDN_D=128, GDN_C=128)
C = 128  # chunk size
D = 128  # head dimension
H = 16  # number of value heads
Hg = 16  # number of key heads (= H: no GQA in default build)


def gdn_chunk_cumsum(g):
    """Per-chunk cumulative sum of gates (resets at chunk boundaries). g: [T, H]."""
    out = torch.zeros_like(g, dtype=torch.float32)
    for j in range(0, g.shape[0], C):
        e = min(j + C, g.shape[0])
        out[j:e, :] = g[j:e, :].float().cumsum(dim=0)
    return out


def safe_exp(x):
    """exp(min(x, 0)) — returns 0 for positive inputs, exp(x) otherwise."""
    return torch.where(x <= 0, torch.exp(x), torch.zeros_like(x))


def ref_scaled_dot_kkt(k, beta, g_cumsum):
    """CPU fp32 reference: A[i,j] = (K[i]·K[j]) * exp(min(g[i]-g[j], 0)) * beta[i],
    strictly lower-triangular (j < i only). Returns [T, H, C]."""
    T = k.shape[0]
    grp = H // k.shape[1]
    kf, bf, gf = k.float(), beta.float(), g_cumsum.float()

    out = torch.zeros(T, H, C, dtype=torch.float32)
    for j in range(0, T, C):
        s, e = j, min(j + C, T)
        v = e - s
        for h in range(H):
            hg = h // grp
            kc = kf[s:e, hg, :]  # [v, D]
            gc = gf[s:e, h]  # [v]
            bc = bf[s:e, h]  # [v]
            blk = (kc @ kc.T) * safe_exp(gc[:, None] - gc[None, :]) * bc[:, None]
            mask = torch.arange(v)[:, None] > torch.arange(v)[None, :]
            out[s:e, h, :v] = blk * mask.float()
    return out


if __name__ == "__main__":

    try:
        lib_path = "build/lib/libkernel_gdn_scaled_dot_kkt.so"
        lib_path = os.path.abspath(lib_path)
        lib = ctypes.CDLL(lib_path)
        print(f"Loaded library from {lib_path}")

        lib.pto_launch_gdn_scaled_dot_kkt.restype = None
        lib.pto_launch_gdn_scaled_dot_kkt.argtypes = [
            ctypes.c_uint32,  # blockDim
            ctypes.c_void_p,  # stream
            ctypes.c_void_p,  # K
            ctypes.c_void_p,  # Beta
            ctypes.c_void_p,  # G
            ctypes.c_void_p,  # Msk
            ctypes.c_void_p,  # workspace
            ctypes.c_void_p,  # A (out)
            ctypes.c_void_p,  # cu_seqlens (nullptr => fixed-length path)
            ctypes.c_int64,  # batch_size
            ctypes.c_int64,  # seq_len
            ctypes.c_int64,  # total_tokens
        ]
        stream_ptr = torch.npu.current_stream()._as_parameter_  # noqa

        def torch_to_ctypes(tensor):
            return ctypes.c_void_p(tensor.data_ptr())

        torch.manual_seed(42)
        batch_size = 1
        seq_len = 256
        T = seq_len  # fixed-length path: total_tokens == batch_size * seq_len

        k_cpu = F.normalize(torch.randn(T, Hg, D, dtype=torch.float16), dim=-1, p=2)
        beta_cpu = torch.rand(T, H, dtype=torch.float16)
        g_cumsum = gdn_chunk_cumsum(
            F.logsigmoid(torch.randn(T, H, dtype=torch.float32))
        )

        K = k_cpu.to(DEVICE)
        Beta = beta_cpu.T.contiguous().to(DEVICE)  # [H, T] fp16
        G = g_cumsum.T.contiguous().to(DEVICE)  # [H, T] fp32
        # Strictly lower-triangular causal mask (diagonal=-1)
        Msk = torch.tril(torch.ones(C, C, dtype=torch.float32), diagonal=-1).to(DEVICE)

        # One work item per (sequence, head) pair.
        block_dim = min(NUM_AI_CORES, batch_size * H)
        # Per-core workspace: 2 slots of [C, C] fp16 for double-buffering KK^T.
        workspace = torch.zeros(block_dim * 2, C, C, device=DEVICE, dtype=torch.float16)
        A = torch.zeros(T, H, C, device=DEVICE, dtype=torch.float16)

        lib.pto_launch_gdn_scaled_dot_kkt(
            block_dim,
            stream_ptr,
            torch_to_ctypes(K),
            torch_to_ctypes(Beta),
            torch_to_ctypes(G),
            torch_to_ctypes(Msk),
            torch_to_ctypes(workspace),
            torch_to_ctypes(A),
            None,  # cu_seqlens
            batch_size,
            seq_len,
            T,
        )
        torch.npu.synchronize()

        a_ref = ref_scaled_dot_kkt(k_cpu, beta_cpu, g_cumsum)
        a_act = A.float().cpu()

        print(f"A max abs diff: {(a_act - a_ref).abs().max().item()}")
        print(f"A is all close? {torch.allclose(a_act, a_ref, rtol=1e-2, atol=1e-2)}")
        print(a_act[1, 0, :8])
        print(a_ref[1, 0, :8])
    finally:
        del lib  # triggers dlclose in CPython
