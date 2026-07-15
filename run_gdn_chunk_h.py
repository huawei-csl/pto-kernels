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


def ref_chunk_h(k, w, u, g_cumsum):
    """CPU fp32 reference: S_next = exp(g_last) * S + K_tilde^T @ (U - W @ S)."""
    T = k.shape[0]
    grp = w.shape[1] // k.shape[1]
    kf, wf, uf, gf = k.float(), w.float(), u.float(), g_cumsum.float()

    num_chunks = (T + C - 1) // C
    h_out = torch.zeros(num_chunks, H, D, D, dtype=torch.float32)
    v_new = torch.zeros(T, H, D, dtype=torch.float32)

    for h in range(H):
        hg = h // grp
        S = torch.zeros(D, D, dtype=torch.float32)
        for ci in range(num_chunks):
            s, e = ci * C, min((ci + 1) * C, T)
            gc = gf[s:e, h]
            gl = gc[e - s - 1]  # last gate in chunk
            h_out[ci, h] = S.clone()
            vc = uf[s:e, h, :] - wf[s:e, h, :] @ S
            v_new[s:e, h, :] = vc
            kv = kf[s:e, hg, :].T @ (vc * torch.exp(gl - gc)[:, None])
            S = torch.exp(gl) * S + kv
    return h_out, v_new


if __name__ == "__main__":

    try:
        lib_path = "build/lib/libkernel_gdn_chunk_h.so"
        lib_path = os.path.abspath(lib_path)
        lib = ctypes.CDLL(lib_path)
        print(f"Loaded library from {lib_path}")

        lib.pto_launch_gdn_chunk_h.restype = None
        lib.pto_launch_gdn_chunk_h.argtypes = [
            ctypes.c_uint32,  # blockDim
            ctypes.c_void_p,  # stream
            ctypes.c_void_p,  # K
            ctypes.c_void_p,  # W
            ctypes.c_void_p,  # U
            ctypes.c_void_p,  # G
            ctypes.c_void_p,  # S  (out)
            ctypes.c_void_p,  # V  (out)
            ctypes.c_void_p,  # FS (out)
            ctypes.c_void_p,  # workspace
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
        total_chunks = (T + C - 1) // C

        k_cpu = F.normalize(torch.randn(T, Hg, D, dtype=torch.float16), dim=-1, p=2)
        w_cpu = torch.randn(T, H, D, dtype=torch.float16)
        u_cpu = torch.randn(T, H, D, dtype=torch.float16)
        g_cumsum = gdn_chunk_cumsum(
            F.logsigmoid(torch.randn(T, H, dtype=torch.float32))
        )

        K = k_cpu.to(DEVICE)
        W = w_cpu.to(DEVICE)
        U = u_cpu.to(DEVICE)
        G = g_cumsum.T.contiguous().to(DEVICE)  # [H, T] fp32

        block_dim = NUM_AI_CORES
        S = torch.zeros(total_chunks, H, D, D, device=DEVICE, dtype=torch.float16)
        V = torch.zeros(T, H, D, device=DEVICE, dtype=torch.float16)
        FS = torch.zeros(batch_size, H, D, D, device=DEVICE, dtype=torch.float16)
        # Per-core workspace: 4 * D*D half elements (WS_WS, WS_K, WS_S, WS_KV).
        workspace = torch.zeros(
            block_dim * D * D * 4, device=DEVICE, dtype=torch.float16
        )

        lib.pto_launch_gdn_chunk_h(
            block_dim,
            stream_ptr,
            torch_to_ctypes(K),
            torch_to_ctypes(W),
            torch_to_ctypes(U),
            torch_to_ctypes(G),
            torch_to_ctypes(S),
            torch_to_ctypes(V),
            torch_to_ctypes(FS),
            torch_to_ctypes(workspace),
            None,  # cu_seqlens
            batch_size,
            seq_len,
            T,
        )
        torch.npu.synchronize()

        s_ref, v_ref = ref_chunk_h(k_cpu, w_cpu, u_cpu, g_cumsum)
        s_act = S.float().cpu()
        v_act = V.float().cpu()

        print(f"S max abs diff: {(s_act - s_ref).abs().max().item()}")
        print(f"V max abs diff: {(v_act - v_ref).abs().max().item()}")
        print(f"S is all close? {torch.allclose(s_act, s_ref, rtol=1e-2, atol=1e-2)}")
        print(f"V is all close? {torch.allclose(v_act, v_ref, rtol=1e-2, atol=1e-2)}")
        print(s_act[0, 0, 0, :8])
        print(s_ref[0, 0, 0, :8])
        print(f"Norm: {torch.norm(s_act)}")
        print(f"Norm: {torch.norm(s_ref)}")
    finally:
        del lib  # triggers dlclose in CPython
