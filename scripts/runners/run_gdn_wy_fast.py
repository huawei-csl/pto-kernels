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


def ref_wy_fast(k, v, beta, A, g_cumsum):
    """CPU fp32 reference: U = A @ (V * beta), W = A @ (K * beta * exp(g))."""
    T = k.shape[0]
    grp = H // k.shape[1]
    kf, vf, bf, Af, gf = k.float(), v.float(), beta.float(), A.float(), g_cumsum.float()

    w_out = torch.zeros(T, H, D, dtype=torch.float32)
    u_out = torch.zeros(T, H, D, dtype=torch.float32)

    for j in range(0, T, C):
        s, e = j, min(j + C, T)
        valid = e - s
        for h in range(H):
            hg = h // grp
            Ab = Af[s:e, h, :valid]  # [valid, valid]
            gc = gf[s:e, h]  # [valid]
            vb = vf[s:e, h, :] * bf[s:e, h, None]  # [valid, D]
            kb = kf[s:e, hg, :] * bf[s:e, h, None] * torch.exp(gc)[:, None]
            u_out[s:e, h, :] = Ab @ vb
            w_out[s:e, h, :] = Ab @ kb
    return w_out, u_out


if __name__ == "__main__":

    try:
        lib_path = "build/lib/libkernel_gdn_wy_fast.so"
        lib_path = os.path.abspath(lib_path)
        lib = ctypes.CDLL(lib_path)
        print(f"Loaded library from {lib_path}")

        lib.pto_launch_gdn_wy_fast.restype = None
        lib.pto_launch_gdn_wy_fast.argtypes = [
            ctypes.c_uint32,  # blockDim
            ctypes.c_void_p,  # stream
            ctypes.c_void_p,  # K
            ctypes.c_void_p,  # V
            ctypes.c_void_p,  # Beta
            ctypes.c_void_p,  # G
            ctypes.c_void_p,  # A
            ctypes.c_void_p,  # workspace_a1
            ctypes.c_void_p,  # workspace_a2
            ctypes.c_void_p,  # W (out)
            ctypes.c_void_p,  # U (out)
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
        chunks_per_seq = (seq_len + C - 1) // C

        k_cpu = F.normalize(torch.randn(T, Hg, D, dtype=torch.float16), dim=-1, p=2)
        v_cpu = torch.randn(T, H, D, dtype=torch.float16)
        beta_cpu = torch.rand(T, H, dtype=torch.float16)
        a_cpu = torch.randn(T, H, C, dtype=torch.float16)
        g_cumsum = gdn_chunk_cumsum(
            F.logsigmoid(torch.randn(T, H, dtype=torch.float32))
        )

        K = k_cpu.to(DEVICE)
        V = v_cpu.to(DEVICE)
        Beta = beta_cpu.T.contiguous().to(DEVICE)  # [H, T] fp16
        G = g_cumsum.T.contiguous().to(DEVICE)  # [H, T] fp32
        A = a_cpu.to(DEVICE)

        # One work item per (sequence, chunk, head) triple.
        block_dim = min(NUM_AI_CORES, batch_size * chunks_per_seq * H)
        # Per-core workspaces (fp16): A1 = A*(exp(g)*beta), A2 = A*beta.
        workspace_a1 = torch.zeros(block_dim, C, C, device=DEVICE, dtype=torch.float16)
        workspace_a2 = torch.zeros(block_dim, C, C, device=DEVICE, dtype=torch.float16)
        W = torch.zeros(T, H, D, device=DEVICE, dtype=torch.float16)
        U = torch.zeros(T, H, D, device=DEVICE, dtype=torch.float16)

        lib.pto_launch_gdn_wy_fast(
            block_dim,
            stream_ptr,
            torch_to_ctypes(K),
            torch_to_ctypes(V),
            torch_to_ctypes(Beta),
            torch_to_ctypes(G),
            torch_to_ctypes(A),
            torch_to_ctypes(workspace_a1),
            torch_to_ctypes(workspace_a2),
            torch_to_ctypes(W),
            torch_to_ctypes(U),
            None,  # cu_seqlens
            batch_size,
            seq_len,
            T,
        )
        torch.npu.synchronize()

        w_ref, u_ref = ref_wy_fast(k_cpu, v_cpu, beta_cpu, a_cpu, g_cumsum)
        w_act = W.float().cpu()
        u_act = U.float().cpu()

        print(f"W max abs diff: {(w_act - w_ref).abs().max().item()}")
        print(f"U max abs diff: {(u_act - u_ref).abs().max().item()}")
        print(f"W is all close? {torch.allclose(w_act, w_ref, rtol=1e-2, atol=1e-1)}")
        print(f"U is all close? {torch.allclose(u_act, u_ref, rtol=1e-2, atol=1e-1)}")
        print(w_act[0, 0, :8])
        print(w_ref[0, 0, :8])
    finally:
        del lib  # triggers dlclose in CPython
