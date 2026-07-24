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
    """CPU fp32 reference for chunk_h — generates the states S and V that chunk_o
    consumes. Returns (h_out [num_chunks, H, D, D], v_new [T, H, D])."""
    T = k.shape[0]
    grp = H // k.shape[1]
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


def ref_chunk_o(q, k, v_new, h_states, g_cumsum):
    """CPU fp32 reference: O = exp(g) * (Q @ S) + (Q @ K^T * gate * causal) @ V."""
    T = q.shape[0]
    grp = H // q.shape[1]
    qf, kf, vf, gf = q.float(), k.float(), v_new.float(), g_cumsum.float()

    num_chunks = (T + C - 1) // C
    o = torch.zeros(T, H, D, dtype=torch.float32)

    for h in range(H):
        hg = h // grp
        for ci in range(num_chunks):
            s, e = ci * C, min((ci + 1) * C, T)
            vlen = e - s
            qc = qf[s:e, hg, :]  # [vlen, D]
            kc = kf[s:e, hg, :]  # [vlen, D]
            vc = vf[s:e, h, :]  # [vlen, D]
            gc = gf[s:e, h]  # [vlen]
            inter = (qc @ h_states[ci, h]) * torch.exp(gc)[:, None]
            qk = qc @ kc.T  # [vlen, vlen]
            causal = torch.arange(vlen)[:, None] >= torch.arange(vlen)[None, :]
            gate = torch.exp(
                torch.minimum(gc[:, None] - gc[None, :], torch.zeros(vlen, vlen))
            )
            o[s:e, h, :] = inter + (qk * gate * causal.float()) @ vc
    return o


if __name__ == "__main__":

    try:
        lib_path = "build/lib/libkernel_gdn_chunk_o.so"
        lib_path = os.path.abspath(lib_path)
        lib = ctypes.CDLL(lib_path)
        print(f"Loaded library from {lib_path}")

        lib.pto_launch_gdn_chunk_o.restype = None
        lib.pto_launch_gdn_chunk_o.argtypes = [
            ctypes.c_uint32,  # blockDim
            ctypes.c_void_p,  # stream
            ctypes.c_void_p,  # Q
            ctypes.c_void_p,  # K
            ctypes.c_void_p,  # V
            ctypes.c_void_p,  # S
            ctypes.c_void_p,  # G
            ctypes.c_void_p,  # Msk
            ctypes.c_void_p,  # workspace_qk
            ctypes.c_void_p,  # workspace_qs_qkv
            ctypes.c_void_p,  # workspace_qk_gated
            ctypes.c_void_p,  # O (out)
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
        chunks_per_seq = (seq_len + C - 1) // C

        q_cpu = F.normalize(torch.randn(T, Hg, D, dtype=torch.float16), dim=-1, p=2)
        k_cpu = F.normalize(torch.randn(T, Hg, D, dtype=torch.float16), dim=-1, p=2)
        w_cpu = torch.randn(T, H, D, dtype=torch.float16)
        u_cpu = torch.randn(T, H, D, dtype=torch.float16)
        g_cumsum = gdn_chunk_cumsum(
            F.logsigmoid(torch.randn(T, H, dtype=torch.float32))
        )

        # States and corrected values come from the chunk_h reference, rounded to
        # fp16 so the CPU reference sees the same inputs as the kernel.
        h_out_f32, v_new_f32 = ref_chunk_h(k_cpu, w_cpu, u_cpu, g_cumsum)
        h_out_fp16 = h_out_f32.half()  # [total_chunks, H, D, D]
        v_new_fp16 = v_new_f32.half()  # [T, H, D]

        Q = q_cpu.to(DEVICE)
        K = k_cpu.to(DEVICE)
        V = v_new_fp16.to(DEVICE)
        # chunk_o expects S flattened to [total_chunks * H, D, D]
        S = h_out_fp16.reshape(total_chunks * H, D, D).to(DEVICE)
        G = g_cumsum.T.contiguous().to(DEVICE)  # [H, T] fp32
        Msk = torch.tril(torch.ones(C, C, dtype=torch.float32)).to(DEVICE)

        # One work item per (sequence, chunk, head) triple.
        block_dim = min(NUM_AI_CORES, batch_size * chunks_per_seq * H)
        # Per-core workspaces (fp16): QK [C, C], QS/QKV [C, D], gated QK [C, C].
        workspace_qk = torch.zeros(block_dim, C, C, device=DEVICE, dtype=torch.float16)
        workspace_qs_qkv = torch.zeros(
            block_dim, C, D, device=DEVICE, dtype=torch.float16
        )
        workspace_qk_gated = torch.zeros(
            block_dim, C, C, device=DEVICE, dtype=torch.float16
        )
        O = torch.zeros(T, H, D, device=DEVICE, dtype=torch.float16)

        lib.pto_launch_gdn_chunk_o(
            block_dim,
            stream_ptr,
            torch_to_ctypes(Q),
            torch_to_ctypes(K),
            torch_to_ctypes(V),
            torch_to_ctypes(S),
            torch_to_ctypes(G),
            torch_to_ctypes(Msk),
            torch_to_ctypes(workspace_qk),
            torch_to_ctypes(workspace_qs_qkv),
            torch_to_ctypes(workspace_qk_gated),
            torch_to_ctypes(O),
            None,  # cu_seqlens
            batch_size,
            seq_len,
            T,
        )
        torch.npu.synchronize()

        o_ref = ref_chunk_o(q_cpu, k_cpu, v_new_fp16, h_out_fp16.float(), g_cumsum)
        o_act = O.float().cpu()

        print(f"O max abs diff: {(o_act - o_ref).abs().max().item()}")
        print(f"O is all close? {torch.allclose(o_act, o_ref, rtol=1e-2, atol=1e-1)}")
        print(o_act[0, 0, :8])
        print(o_ref[0, 0, :8])
    finally:
        del lib  # triggers dlclose in CPython
