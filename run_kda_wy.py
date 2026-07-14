import os
import ctypes
import torch

# Select device "cpu" or "npu"
DEVICE = "npu:0"

NUM_AI_CORES = int(getattr(torch.npu.get_device_properties("npu"), "cube_core_num", 20))

# Compile-time kernel constants (default build: GDN_H=16, GDN_D=128, GDN_C=128)
CHUNK = 128
D = 128  # head dimension (K and V share it in the default build)
H = 16  # number of heads


def gate_cumsum(g_log):
    """Per-dim, per-chunk cumulative sum of gates. g_log: [T, H, D] -> [T, H, D] fp32."""
    out = torch.zeros_like(g_log, dtype=torch.float32)
    for j in range(0, g_log.shape[0], CHUNK):
        e = min(j + CHUNK, g_log.shape[0])
        out[j:e] = g_log[j:e].float().cumsum(dim=0)
    return out


def make_inputs(T):
    """Well-conditioned inputs plus the CPU-computed INV = (I + L)^{-1} per chunk."""
    torch.manual_seed(42)
    k = torch.randn(T, H, D)
    # L2-normalise keys: unnormalised keys give ill-conditioned L (cond ~1e6),
    # which makes the fp32 inverse inaccurate.
    k = k / k.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    v = torch.randn(T, H, D)
    beta = torch.sigmoid(torch.randn(T, H))
    g_cs = gate_cumsum(-torch.rand(T, H, D))  # per-dim gates in (-1, 0)

    INV = torch.zeros(T, H, CHUNK, dtype=torch.float32)
    for j in range(0, T, CHUNK):
        e = min(j + CHUNK, T)
        valid = e - j
        for h in range(H):
            kf = k[j:e, h].float()
            gf = g_cs[j:e, h].float()
            bf = beta[j:e, h].float()
            # L[r, c] = beta[r] * (k[r]*exp(g[r])) @ (k[c]*exp(-g[c])), strictly lower
            L = torch.tril(
                bf[:, None] * ((kf * torch.exp(gf)) @ (kf * torch.exp(-gf)).T),
                diagonal=-1,
            )
            INV[j:e, h, :valid] = torch.linalg.inv(
                torch.eye(valid, dtype=torch.float64) + L.double()
            ).float()
    return k, v, g_cs, beta, INV


def ref_kda_wy(k, v, g_cs, beta, INV):
    """CPU fp32 reference: A2 = INV * beta (column-scale), U = A2 @ V, W = A2 @ K_eff."""
    T = k.shape[0]
    u = torch.zeros(T, H, D, dtype=torch.float32)
    w = torch.zeros(T, H, D, dtype=torch.float32)
    for j in range(0, T, CHUNK):
        e = min(j + CHUNK, T)
        valid = e - j
        for h in range(H):
            A2 = INV[j:e, h, :valid].float() * beta[j:e, h].float()[None, :]
            K_eff = k[j:e, h].float() * torch.exp(g_cs[j:e, h].float())
            u[j:e, h] = A2 @ v[j:e, h].float()
            w[j:e, h] = A2 @ K_eff
    return u, w


if __name__ == "__main__":

    try:
        lib_path = "build/lib/libkernel_kda_wy.so"
        lib_path = os.path.abspath(lib_path)
        lib = ctypes.CDLL(lib_path)
        print(f"Loaded library from {lib_path}")

        lib.pto_launch_kda_wy.restype = None
        lib.pto_launch_kda_wy.argtypes = [
            ctypes.c_uint32,  # blockDim
            ctypes.c_void_p,  # stream
            ctypes.c_void_p,  # K
            ctypes.c_void_p,  # V
            ctypes.c_void_p,  # Beta
            ctypes.c_void_p,  # G
            ctypes.c_void_p,  # A (== INV)
            ctypes.c_void_p,  # workspace_a2
            ctypes.c_void_p,  # workspace_keff
            ctypes.c_void_p,  # U (out)
            ctypes.c_void_p,  # W (out)
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
        T = seq_len  # fixed-length path: total_tokens == batch_size * seq_len

        k, v, g_cs, beta, INV = make_inputs(T)
        u_ref, w_ref = ref_kda_wy(k, v, g_cs, beta, INV)

        # K, G, Beta are head-major; V and INV are BSND.
        K_npu = k.permute(1, 0, 2).contiguous().half().to(DEVICE)  # [H, T, D]
        V_npu = v.contiguous().half().to(DEVICE)  # [T, H, D]
        G_npu = g_cs.permute(1, 0, 2).contiguous().float().to(DEVICE)  # [H, T, D]
        Beta_npu = beta.permute(1, 0).contiguous().half().to(DEVICE)  # [H, T]
        INV_npu = INV.contiguous().half().to(DEVICE)  # [T, H, CHUNK]

        # Cap block_dim to the actual number of work items (chunk x head).
        chunks_per_seq = (seq_len + CHUNK - 1) // CHUNK
        block_dim = min(NUM_AI_CORES, batch_size * chunks_per_seq * H)

        # Per-core workspaces (fp16): A2 [C, C] and K_eff [C, D].
        ws_a2 = torch.empty(block_dim, CHUNK, CHUNK, device=DEVICE, dtype=torch.float16)
        ws_keff = torch.empty(block_dim, CHUNK, D, device=DEVICE, dtype=torch.float16)

        U = torch.empty(T, H, D, device=DEVICE, dtype=torch.float16)
        W = torch.empty(T, H, D, device=DEVICE, dtype=torch.float16)

        lib.pto_launch_kda_wy(
            block_dim,
            stream_ptr,
            torch_to_ctypes(K_npu),
            torch_to_ctypes(V_npu),
            torch_to_ctypes(Beta_npu),
            torch_to_ctypes(G_npu),
            torch_to_ctypes(INV_npu),
            torch_to_ctypes(ws_a2),
            torch_to_ctypes(ws_keff),
            torch_to_ctypes(U),
            torch_to_ctypes(W),
            None,  # cu_seqlens
            batch_size,
            seq_len,
            T,
        )
        torch.npu.synchronize()

        u_act = U.float().cpu()
        w_act = W.float().cpu()

        print(f"U max abs diff: {(u_act - u_ref).abs().max().item()}")
        print(f"W max abs diff: {(w_act - w_ref).abs().max().item()}")
        print(f"U is all close? {torch.allclose(u_act, u_ref, rtol=1e-2, atol=1e-2)}")
        print(f"W is all close? {torch.allclose(w_act, w_ref, rtol=1e-2, atol=1e-2)}")
        print(u_act[0, 0, :8])
        print(u_ref[0, 0, :8])
    finally:
        del lib  # triggers dlclose in CPython
