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
    """CPU fp32 kda_chunk_h reference — produces the (S, V_corr) inputs of chunk_o."""
    T = K.shape[0]
    nc = (T + CHUNK - 1) // CHUNK

    S_snap = torch.zeros(nc, HV, D, D, dtype=torch.float32)
    V_corr = torch.zeros(T, HV, D, dtype=torch.float32)

    for h in range(HV):
        S = torch.zeros(D, D, dtype=torch.float32)
        for ci in range(nc):
            s = ci * CHUNK
            e = min(s + CHUNK, T)

            g_cs = G_cs[s:e, h, :]
            g_total = g_cs[e - s - 1, :]

            S_snap[ci, h] = S.clone()
            v_c = U[s:e, h, :] - W[s:e, h, :] @ S
            V_corr[s:e, h, :] = v_c
            k_rest = K[s:e, h, :] * torch.exp(g_total[None, :] - g_cs)
            S = torch.exp(g_total)[:, None] * S + k_rest.T @ v_c

    return S_snap, V_corr


def ref_kda_chunk_o(Q, K, V_corr, S_snap, G_cs):
    """CPU fp32 reference for the output computation.

    Per chunk:
      q_eff = Q * exp(g_cs)
      k_eff = K * exp(-g_cs)
      Aqk   = tril(q_eff @ k_eff^T)   (inclusive diagonal)
      O     = q_eff @ S + Aqk @ V_corr

    Returns O [T, HV, D] fp32.
    """
    T = Q.shape[0]
    nc = (T + CHUNK - 1) // CHUNK
    output = torch.zeros(T, HV, D, dtype=torch.float32)

    for h in range(HV):
        for ci in range(nc):
            s = ci * CHUNK
            e = min(s + CHUNK, T)

            g_cs = G_cs[s:e, h, :]
            q_eff = Q[s:e, h, :] * torch.exp(g_cs)
            k_eff = K[s:e, h, :] * torch.exp(-g_cs)

            inter = q_eff @ S_snap[ci, h]
            Aqk = torch.tril(q_eff @ k_eff.T)  # inclusive causal mask

            output[s:e, h, :] = inter + Aqk @ V_corr[s:e, h, :]

    return output


if __name__ == "__main__":

    try:
        lib_path = "build/lib/libkernel_kda_chunk_o.so"
        lib_path = os.path.abspath(lib_path)
        lib = ctypes.CDLL(lib_path)
        print(f"Loaded library from {lib_path}")

        lib.pto_launch_kda_chunk_o.restype = None
        lib.pto_launch_kda_chunk_o.argtypes = [
            ctypes.c_uint32,  # blockDim
            ctypes.c_void_p,  # stream
            ctypes.c_void_p,  # Q
            ctypes.c_void_p,  # K
            ctypes.c_void_p,  # V_corr
            ctypes.c_void_p,  # S
            ctypes.c_void_p,  # G
            ctypes.c_void_p,  # Mask
            ctypes.c_void_p,  # workspace
            ctypes.c_void_p,  # O (out)
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
        q_cpu = F.normalize(torch.randn(T, HV, D), dim=-1, p=2).float()
        k_cpu = F.normalize(torch.randn(T, HV, D), dim=-1, p=2).float()
        w_cpu = torch.randn(T, HV, D, dtype=torch.float32)
        u_cpu = torch.randn(T, HV, D, dtype=torch.float32)
        # Small gate magnitudes keep exp(+/-g_cs) bounded, avoiding fp16 overflow.
        g_raw = -torch.rand(T, HV, D, dtype=torch.float32) * 0.05
        g_cs = chunk_cumsum_kda(g_raw)  # [T, HV, D] fp32

        total_chunks = (T + CHUNK - 1) // CHUNK

        # Build S and V_corr with the CPU chunk_h reference, then round-trip to
        # fp16 so the reference runs at the precision the kernel actually sees.
        S_f32, V_corr_f32 = ref_kda_chunk_h(k_cpu, w_cpu, u_cpu, g_cs)
        S_fp16 = S_f32.half()  # [total_chunks, HV, D, D]
        V_corr_fp16 = V_corr_f32.half()  # [T, HV, D]

        O_ref = ref_kda_chunk_o(q_cpu, k_cpu, V_corr_fp16.float(), S_fp16.float(), g_cs)

        # Q, K and G are head-major; V_corr and O are BSND.
        Q_npu = q_cpu.half().permute(1, 0, 2).contiguous().to(DEVICE)  # [HV, T, D]
        K_npu = k_cpu.half().permute(1, 0, 2).contiguous().to(DEVICE)  # [HV, T, D]
        G_npu = g_cs.permute(1, 0, 2).contiguous().to(DEVICE)  # [HV, T, D] fp32
        V_npu = V_corr_fp16.contiguous().to(DEVICE)  # [T, HV, D]
        S_npu = S_fp16.contiguous().to(DEVICE)  # [total_chunks, HV, D, D]

        # Inclusive lower-triangular mask [C, C] fp32: 1 where row >= col.
        Mask = (
            (torch.arange(CHUNK)[:, None] >= torch.arange(CHUNK)[None, :])
            .float()
            .to(DEVICE)
        )

        block_dim = NUM_AI_CORES

        # Per-core workspace (fp32): WS_Q, WS_K, WS_V, WS_S, WS_QK, WS_QS, WS_QKV.
        ws_per_core = 5 * CHUNK * D + D * D + CHUNK * CHUNK
        workspace = torch.zeros(
            block_dim * ws_per_core, device=DEVICE, dtype=torch.float32
        )

        O = torch.zeros(T, HV, D, device=DEVICE, dtype=torch.float16)

        lib.pto_launch_kda_chunk_o(
            block_dim,
            stream_ptr,
            torch_to_ctypes(Q_npu),
            torch_to_ctypes(K_npu),
            torch_to_ctypes(V_npu),
            torch_to_ctypes(S_npu),
            torch_to_ctypes(G_npu),
            torch_to_ctypes(Mask),
            torch_to_ctypes(workspace),
            torch_to_ctypes(O),
            None,  # cu_seqlens
            batch_size,
            seq_len,
            T,
        )
        torch.npu.synchronize()

        o_act = O.float().cpu()

        print(f"O max abs diff: {(o_act - O_ref).abs().max().item()}")
        print(f"O is all close? {torch.allclose(o_act, O_ref, rtol=1e-2, atol=1e-2)}")
        print(o_act[0, 0, :8])
        print(O_ref[0, 0, :8])
    finally:
        del lib  # triggers dlclose in CPython
