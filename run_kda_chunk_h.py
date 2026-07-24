import os
import ctypes
import torch
import torch.nn.functional as F

# Select device "cpu" or "npu"
DEVICE = "npu:0"

NUM_AI_CORES = int(getattr(torch.npu.get_device_properties("npu"), "cube_core_num", 20))

# Compile-time kernel constants (default build: GDN_H=16, GDN_D=128, GDN_C=128)
CHUNK = 128
D = 128  # head dimension (K and V share it in the default build)
HV = 16  # number of heads


def chunk_cumsum_kda(g):
    """Per-chunk cumulative sum of per-dim gates. g: [T, HV, D] -> [T, HV, D] fp32."""
    out = torch.zeros_like(g, dtype=torch.float32)
    for j in range(0, g.shape[0], CHUNK):
        e = min(j + CHUNK, g.shape[0])
        out[j:e] = g[j:e].float().cumsum(dim=0)
    return out


def ref_kda_chunk_h(K, W, U, G_cs):
    """CPU fp32 reference for the chunk-recurrent hidden-state update.

    Per chunk:
      v_corr = U - W @ S
      k_rest = K * exp(g_total - g_cs)
      S_new  = diag(exp(g_total)) @ S + k_rest^T @ v_corr

    All inputs are [T, HV, D] fp32. Returns:
      S_snap [total_chunks, HV, D, D] fp32 — state entering each chunk
      V_corr [T, HV, D] fp32              — residual-corrected values
    """
    T = K.shape[0]
    nc = (T + CHUNK - 1) // CHUNK

    S_snap = torch.zeros(nc, HV, D, D, dtype=torch.float32)
    V_corr = torch.zeros(T, HV, D, dtype=torch.float32)

    for h in range(HV):
        S = torch.zeros(D, D, dtype=torch.float32)
        for ci in range(nc):
            s = ci * CHUNK
            e = min(s + CHUNK, T)

            g_cs = G_cs[s:e, h, :]  # [valid, D]
            g_total = g_cs[e - s - 1, :]  # [D]

            S_snap[ci, h] = S.clone()

            v_c = U[s:e, h, :] - W[s:e, h, :] @ S
            V_corr[s:e, h, :] = v_c

            k_rest = K[s:e, h, :] * torch.exp(g_total[None, :] - g_cs)
            S = torch.exp(g_total)[:, None] * S + k_rest.T @ v_c

    return S_snap, V_corr


if __name__ == "__main__":

    try:
        lib_path = "build/lib/libkernel_kda_chunk_h.so"
        lib_path = os.path.abspath(lib_path)
        lib = ctypes.CDLL(lib_path)
        print(f"Loaded library from {lib_path}")

        lib.pto_launch_kda_chunk_h.restype = None
        lib.pto_launch_kda_chunk_h.argtypes = [
            ctypes.c_uint32,  # blockDim
            ctypes.c_void_p,  # stream
            ctypes.c_void_p,  # K
            ctypes.c_void_p,  # W
            ctypes.c_void_p,  # U
            ctypes.c_void_p,  # G
            ctypes.c_void_p,  # S (out)
            ctypes.c_void_p,  # V_corr (out)
            ctypes.c_void_p,  # workspace
            ctypes.c_void_p,  # cu_seqlens (nullptr => fixed-length path)
            ctypes.c_int64,  # batch_size
            ctypes.c_int64,  # seq_len
            ctypes.c_int64,  # total_tokens
        ]
        stream_ptr = torch.npu.current_stream()._as_parameter_  # noqa

        def torch_to_ctypes(tensor):
            return ctypes.c_void_p(tensor.data_ptr())

        batch_size = 1
        seq_len = 256
        T = batch_size * seq_len  # fixed-length path

        torch.manual_seed(42)
        # L2-normalised keys avoid poorly-conditioned S matrices.
        k_cpu = F.normalize(torch.randn(T, HV, D), dim=-1, p=2).float()
        w_cpu = torch.randn(T, HV, D, dtype=torch.float32)
        u_cpu = torch.randn(T, HV, D, dtype=torch.float32)
        # Small gate magnitudes keep exp(g_total - g_cs) bounded for fp16 inputs.
        g_raw = -torch.rand(T, HV, D, dtype=torch.float32) * 0.05
        g_cs = chunk_cumsum_kda(g_raw)  # [T, HV, D] fp32

        S_ref, V_corr_ref = ref_kda_chunk_h(k_cpu, w_cpu, u_cpu, g_cs)

        total_chunks = (T + CHUNK - 1) // CHUNK

        # K and G are head-major; W and U are BSND.
        K_npu = k_cpu.half().permute(1, 0, 2).contiguous().to(DEVICE)  # [HV, T, D]
        W_npu = w_cpu.half().contiguous().to(DEVICE)  # [T, HV, D]
        U_npu = u_cpu.half().contiguous().to(DEVICE)  # [T, HV, D]
        G_npu = g_cs.permute(1, 0, 2).contiguous().to(DEVICE)  # [HV, T, D] fp32

        block_dim = NUM_AI_CORES

        # Per-core workspace (fp16): WS_WS[C,D] WS_K[C,D] WS_V[C,D] WS_S[D,D] WS_KV[D,D]
        ws_per_core = 3 * CHUNK * D + 2 * D * D
        workspace = torch.zeros(
            block_dim * ws_per_core, device=DEVICE, dtype=torch.float16
        )

        # Zero-initialised outputs: the kernel writes only the valid tokens/chunks.
        S = torch.zeros(total_chunks, HV, D, D, device=DEVICE, dtype=torch.float16)
        V_corr = torch.zeros(T, HV, D, device=DEVICE, dtype=torch.float16)

        lib.pto_launch_kda_chunk_h(
            block_dim,
            stream_ptr,
            torch_to_ctypes(K_npu),
            torch_to_ctypes(W_npu),
            torch_to_ctypes(U_npu),
            torch_to_ctypes(G_npu),
            torch_to_ctypes(S),
            torch_to_ctypes(V_corr),
            torch_to_ctypes(workspace),
            None,  # cu_seqlens
            batch_size,
            seq_len,
            T,
        )
        torch.npu.synchronize()

        s_act = S.float().cpu()
        v_act = V_corr.float().cpu()

        print(f"S max abs diff: {(s_act - S_ref).abs().max().item()}")
        print(f"V_corr max abs diff: {(v_act - V_corr_ref).abs().max().item()}")
        print(f"S is all close? {torch.allclose(s_act, S_ref, rtol=1e-2, atol=1e-2)}")
        print(
            "V_corr is all close? "
            f"{torch.allclose(v_act, V_corr_ref, rtol=1e-2, atol=1e-2)}"
        )
        print(v_act[0, 0, :8])
        print(V_corr_ref[0, 0, :8])
    finally:
        del lib  # triggers dlclose in CPython
