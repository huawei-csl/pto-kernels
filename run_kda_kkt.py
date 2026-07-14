import os
import ctypes
import torch
import torch.nn.functional as F

# Select device "cpu" or "npu"
DEVICE = "npu:0"

NUM_AI_CORES = int(getattr(torch.npu.get_device_properties("npu"), "cube_core_num", 20))

# Compile-time kernel constants
# (default build: KDA_KKT_H=4, KDA_KKT_D=128, KDA_KKT_C=128)
CHUNK = 128
D = 128  # key dimension per head
H = 4  # number of heads


def chunk_cumsum_per_dim(g):
    """Within-chunk cumulative sum of per-dim log-gates. g: [H, T, D] -> [H, T, D]."""
    out = torch.zeros_like(g, dtype=torch.float32)
    T = g.shape[1]
    for j in range(0, T, CHUNK):
        e = min(j + CHUNK, T)
        out[:, j:e, :] = g[:, j:e, :].float().cumsum(dim=1)
    return out


def make_inputs(T):
    """Head-major inputs: k [H, T, D], g_cs [H, T, D], beta [H, T] — all fp32."""
    torch.manual_seed(42)
    # L2-normalised keys: unnormalised keys give an ill-conditioned L.
    k = F.normalize(torch.randn(H, T, D), dim=-1, p=2).float()
    g_log = -torch.rand(H, T, D)  # log-gates in (-1, 0)
    g_cs = chunk_cumsum_per_dim(g_log)
    beta = torch.sigmoid(torch.randn(H, T)).float()
    return k, g_cs, beta


def ref_kda_kkt(k, g_cs, beta):
    """CPU fp32 reference, per chunk / head, strictly lower-tri r > c:

    L[r, c] = beta[r] * sum_d k[r,d] * k[c,d] * exp(min(g_cs[r,d] - g_cs[c,d], 0))

    Returns L [T, H, CHUNK] fp32 (BSND).
    """
    T = k.shape[1]
    out = torch.zeros(T, H, CHUNK, dtype=torch.float32)
    for j in range(0, T, CHUNK):
        e = min(j + CHUNK, T)
        valid = e - j
        for h in range(H):
            kc = k[h, j:e, :].float()  # [valid, D]
            gc = g_cs[h, j:e, :].float()  # [valid, D]
            bc = beta[h, j:e].float()  # [valid]

            diff = torch.clamp(gc[:, None, :] - gc[None, :, :], max=0.0)
            prod = (kc[:, None, :] * kc[None, :, :] * diff.exp()).sum(-1)
            prod = prod * bc[:, None]

            mask = torch.arange(valid)[:, None] > torch.arange(valid)[None, :]
            out[j:e, h, :valid] = prod * mask.float()
    return out


if __name__ == "__main__":

    try:
        lib_path = "build/lib/libkernel_kda_kkt.so"
        lib_path = os.path.abspath(lib_path)
        lib = ctypes.CDLL(lib_path)
        print(f"Loaded library from {lib_path}")

        lib.pto_launch_kda_kkt.restype = None
        lib.pto_launch_kda_kkt.argtypes = [
            ctypes.c_uint32,  # blockDim
            ctypes.c_void_p,  # stream
            ctypes.c_void_p,  # K
            ctypes.c_void_p,  # G_cs
            ctypes.c_void_p,  # Beta
            ctypes.c_void_p,  # Mask
            ctypes.c_void_p,  # L (out)
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
        T = batch_size * seq_len

        k, g_cs, beta = make_inputs(T)
        L_ref = ref_kda_kkt(k, g_cs, beta)

        # K, G_cs and Beta are head-major; L is BSND.
        K_npu = k.half().contiguous().to(DEVICE)  # [H, T, D] fp16
        G_npu = g_cs.float().contiguous().to(DEVICE)  # [H, T, D] fp32
        Beta_npu = beta.half().contiguous().to(DEVICE)  # [H, T]    fp16

        # Strict-lower-tri mask [C, C] fp32 — zeros the upper-tri entries of L.
        Mask = torch.tril(
            torch.ones(CHUNK, CHUNK, dtype=torch.float32), diagonal=-1
        ).to(DEVICE)

        # The Vec-only kernel splits each (seq, head) chunk into two row halves,
        # so there are 2 work items per (seq, head).
        block_dim = min(NUM_AI_CORES, batch_size * H * 2)

        # Zero-initialised: the kernel only writes strict-lower-tri entries.
        L = torch.zeros(T, H, CHUNK, device=DEVICE, dtype=torch.float16)

        lib.pto_launch_kda_kkt(
            block_dim,
            stream_ptr,
            torch_to_ctypes(K_npu),
            torch_to_ctypes(G_npu),
            torch_to_ctypes(Beta_npu),
            torch_to_ctypes(Mask),
            torch_to_ctypes(L),
            None,  # cu_seqlens
            batch_size,
            seq_len,
            T,
        )
        torch.npu.synchronize()

        L_act = L.float().cpu()

        print(f"L max abs diff: {(L_act - L_ref).abs().max().item()}")
        print(f"L is all close? {torch.allclose(L_act, L_ref, rtol=1e-2, atol=1e-2)}")
        print(L_act[1, 0, :8])
        print(L_ref[1, 0, :8])
    finally:
        del lib  # triggers dlclose in CPython
