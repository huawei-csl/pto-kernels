"""GDN before/after: plug our fused conv1d+SiLU into the repo's chunked GDN.

A Gated DeltaNet block computes, per layer:

    hidden --in_proj--> mixed_qkv ──► short depthwise causal conv1d + SiLU ──►
        split into q,k,v [T,H,D] ──► chunked GDN core:
            scaled_dot_kkt ─► wy_fast ─► chunk_h ─► chunk_o ──► o

The SHORT CONV1D+SiLU is exactly our kernel. This script measures:
  * the conv stage:   BEFORE = aclnnConvolution (torch F.conv1d+SiLU),
                      AFTER  = our fused PTO kernel
  * the GDN core:     the repo's 4 PTO kernels (pto_gdn_*), identical in both
  * the whole block:  conv + core, before vs after

Run:
    source ~/conv_pto/pto_env.sh
    cd ~/conv_pto/pto-kernels/examples/jit_cpp/conv1d
    ASCEND_RT_VISIBLE_DEVICES=5 python run_gdn_conv_integration.py
"""
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from jit_util_conv1d_dw import jit_compile, K

torch.npu.set_device("npu")
torch.manual_seed(0)
DEV = "npu"

# GDN default-build constants (from tests/test_gdn_chunk_h.py)
C = 128   # chunk size
D = 128   # head dim
H = 16    # value heads
Hg = 16   # key heads (= H, no GQA)
CONV_DIM = 3 * H * D   # q,k,v projections share the conv -> 3*2048 = 6144

try:
    from pto_kernels import (pto_chunk_h, pto_gdn_chunk_o,
                             pto_gdn_scaled_dot_kkt, pto_gdn_wy_fast)
    HAVE_GDN = True
except Exception as e:  # noqa: BLE001
    HAVE_GDN = False
    _GDN_ERR = repr(e)[:160]


def total_chunks(T):
    return (T + C - 1) // C


def chunk_cumsum(g):  # g [T,H] -> per-chunk cumsum [T,H]
    T = g.shape[0]
    out = torch.zeros_like(g, dtype=torch.float32)
    for j in range(0, T, C):
        e = min(j + C, T)
        out[j:e] = g[j:e].float().cumsum(dim=0)
    return out


# ---------------------------------------------------------------------------
# conv stage variants (depthwise causal conv1d + bias + SiLU over CONV_DIM)
# ---------------------------------------------------------------------------
def conv_ours(fn, x, w, bias):
    return fn(x, w, bias)                       # [T, CONV_DIM] fp16


def conv_aclnn(x, weight, bf):
    T, W = x.shape
    xt = F.pad(x.t().reshape(1, W, T).float(), (K - 1, 0))
    z = F.conv1d(xt, weight, bias=bf, groups=W)
    silu = z * torch.sigmoid(z)
    return silu.reshape(W, T).t().contiguous().to(torch.float16)


# ---------------------------------------------------------------------------
# GDN core (the repo's 4 chunked kernels), q/k/v from the conv output
# ---------------------------------------------------------------------------
def split_qkv(y):
    q = y[:, 0:H * D].reshape(-1, H, D)
    k = y[:, H * D:2 * H * D].reshape(-1, Hg, D)
    v = y[:, 2 * H * D:3 * H * D].reshape(-1, H, D)
    return q.contiguous(), k.contiguous(), v.contiguous()


# constant masks (hoisted; they do not depend on the data)
_MSK_KKT = torch.tril(torch.ones(C, C, dtype=torch.float32), diagonal=-1).to(DEV)
_MSK_O = torch.tril(torch.ones(C, C, dtype=torch.float32)).to(DEV)


def gdn_core(q, k, v, Beta, G, T):
    """The repo's 4 chunked GDN kernels. k already L2-normalized;
    Beta/G already [H,T]; masks hoisted. q,k,v are [T,H,D] fp16."""
    A = pto_gdn_scaled_dot_kkt(k, Beta, G, _MSK_KKT, batch_size=1, seq_len=T)
    w, u = pto_gdn_wy_fast(k, v, Beta, G, A, batch_size=1, seq_len=T)
    tc = total_chunks(T)
    s_out, v_new, _ = pto_chunk_h(k, w, u, G, batch_size=1, seq_len=T, total_chunks=tc)
    S = s_out.reshape(tc * H, D, D).to(torch.float16)
    o = pto_gdn_chunk_o(q, k, v_new.to(torch.float16), S, G, _MSK_O,
                        batch_size=1, seq_len=T)
    return o


# ---------------------------------------------------------------------------
def bench(call, iters=50, warmup=10):
    for _ in range(warmup):
        call()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        call()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    fn = jit_compile(str(Path(__file__).resolve().parent / "conv1d_dw_pto.cpp"),
                     verbose=True)

    print(f"\nGDN block: conv_dim={CONV_DIM} (q,k,v = 3 x H*D), H={H} D={D} C={C}")
    if not HAVE_GDN:
        print(f"\n[!] pto_kernels (repo GDN) NOT importable: {_GDN_ERR}")
        print("[!] Build it first: `SOC_VERSION=Ascend910B2 make wheel && make install`")
        print("[!] Falling back to CONV-STAGE-ONLY before/after.\n")

    print("== conv stage (depthwise causal conv1d + bias + SiLU), W = conv_dim ==")
    print(f"{'seq':>5} {'aclnn_us':>9} {'ours_us':>9} {'speedup':>8}")
    convs = {}
    for T in (128, 256, 384, 512):
        x = (2 * torch.rand((T, CONV_DIM), device=DEV, dtype=torch.float16) - 1)
        wc = (torch.rand((K, CONV_DIM), device=DEV, dtype=torch.float32) - 0.5)
        bc = (torch.rand((CONV_DIM,), device=DEV, dtype=torch.float32) - 0.5)
        weight = wc.t().contiguous().reshape(CONV_DIM, 1, K)
        bf = bc.float()
        t_acl = bench(lambda: conv_aclnn(x, weight, bf))
        t_our = bench(lambda: conv_ours(fn, x, wc, bc))
        convs[T] = (x, wc, bc, weight, bf, t_acl, t_our)
        print(f"{T:>5} {t_acl*1e6:>9.1f} {t_our*1e6:>9.1f} {t_acl/t_our:>7.2f}x")

    if not HAVE_GDN:
        return

    # ---- numerical sanity: our-conv path vs aclnn-conv path through the core ----
    print("\n== integrated-path correctness (our conv vs aclnn conv, then GDN core) ==")
    for T in (128, 512):
        x, wc, bc, weight, bf, _, _ = convs[T]
        beta = torch.rand((T, H), device=DEV, dtype=torch.float16)
        g_cumsum = chunk_cumsum(
            F.logsigmoid(torch.randn((T, H), device=DEV, dtype=torch.float32)))
        Beta = beta.T.contiguous()
        G = g_cumsum.T.contiguous()

        def run(conv_y):
            q, k, v = split_qkv(conv_y)
            k = F.normalize(k.float(), dim=-1, p=2).to(torch.float16)
            return gdn_core(q, k, v, Beta, G, T)

        o_our = run(conv_ours(fn, x, wc, bc))
        o_acl = run(conv_aclnn(x, weight, bf))
        torch.npu.synchronize()
        denom = o_acl.float().abs().mean().clamp_min(1e-6)
        rel = ((o_our.float() - o_acl.float()).abs().mean() / denom).item()
        mx = (o_our.float() - o_acl.float()).abs().max().item()
        print(f"  seq={T:>4}  GDN-output mean-rel-diff={rel:.3e}  max-abs={mx:.3e}")

    # ---- timing: clean, fully warmed, core broken down by kernel ----
    print("\n== full GDN block (conv + chunked core), before vs after ==")
    print(f"{'seq':>5} {'kkt_us':>8} {'wy_us':>8} {'h_us':>8} {'o_us':>8} "
          f"{'core_us':>9} {'blk_before':>11} {'blk_after':>10} "
          f"{'blk_spd':>8} {'conv%':>7}")
    for T in (128, 256, 384, 512):
        x, wc, bc, weight, bf, t_acl, t_our = convs[T]
        beta = torch.rand((T, H), device=DEV, dtype=torch.float16)
        g_cumsum = chunk_cumsum(
            F.logsigmoid(torch.randn((T, H), device=DEV, dtype=torch.float32)))
        Beta = beta.T.contiguous()
        G = g_cumsum.T.contiguous()

        # fixed pre-split, pre-normalized qkv so core timing is conv-independent
        q, k, v = split_qkv(conv_ours(fn, x, wc, bc))
        k = F.normalize(k.float(), dim=-1, p=2).to(torch.float16)
        tc = total_chunks(T)

        # hard global warmup of EVERY kernel (kills cold-start/autotune contamination)
        for _ in range(30):
            conv_aclnn(x, weight, bf)
            conv_ours(fn, x, wc, bc)
            gdn_core(q, k, v, Beta, G, T)
        torch.npu.synchronize()

        # per-kernel breakdown of the core
        A = pto_gdn_scaled_dot_kkt(k, Beta, G, _MSK_KKT, batch_size=1, seq_len=T)
        w, u = pto_gdn_wy_fast(k, v, Beta, G, A, batch_size=1, seq_len=T)
        s_out, v_new, _ = pto_chunk_h(k, w, u, G, batch_size=1, seq_len=T,
                                      total_chunks=tc)
        S = s_out.reshape(tc * H, D, D).to(torch.float16)
        t_kkt = bench(lambda: pto_gdn_scaled_dot_kkt(k, Beta, G, _MSK_KKT,
                                                     batch_size=1, seq_len=T))
        t_wy = bench(lambda: pto_gdn_wy_fast(k, v, Beta, G, A, batch_size=1, seq_len=T))
        t_h = bench(lambda: pto_chunk_h(k, w, u, G, batch_size=1, seq_len=T,
                                        total_chunks=tc))
        t_o = bench(lambda: pto_gdn_chunk_o(q, k, v_new.to(torch.float16), S, G,
                                            _MSK_O, batch_size=1, seq_len=T))
        t_core = bench(lambda: gdn_core(q, k, v, Beta, G, T))

        t_before = t_acl + t_core      # core identical both sides; conv is the only change
        t_after = t_our + t_core
        conv_pct = 100.0 * t_acl / t_before
        print(f"{T:>5} {t_kkt*1e6:>8.1f} {t_wy*1e6:>8.1f} {t_h*1e6:>8.1f} {t_o*1e6:>8.1f} "
              f"{t_core*1e6:>9.1f} {t_before*1e6:>11.1f} {t_after*1e6:>10.1f} "
              f"{t_before/t_after:>7.2f}x {conv_pct:>6.1f}%")

    # ---- honest FULL-LAYER view: also include in_proj/out_proj GEMMs ----
    # A real GDN layer is: in_proj -> conv+SiLU -> DeltaNet core -> out_proj.
    # The GEMMs are conv-independent (identical before/after) so they dilute the
    # conv's share. d_model = value_dim = H*D = 2048 (standard for this config).
    print("\n== full GDN LAYER (in_proj + conv + core + out_proj), before vs after ==")
    print(f"  (in_proj [T,{H*D}]x[{H*D},{CONV_DIM}], out_proj [T,{H*D}]x[{H*D},{H*D}], fp16 = aclnnMatmul)")
    print(f"{'seq':>5} {'inproj_us':>10} {'outproj_us':>11} {'lyr_before':>11} "
          f"{'lyr_after':>10} {'lyr_spd':>8} {'conv%':>7}")
    d_model = H * D
    for T in (128, 256, 384, 512):
        x, wc, bc, weight, bf, t_acl, t_our = convs[T]
        beta = torch.rand((T, H), device=DEV, dtype=torch.float16)
        g_cumsum = chunk_cumsum(
            F.logsigmoid(torch.randn((T, H), device=DEV, dtype=torch.float32)))
        Beta = beta.T.contiguous()
        G = g_cumsum.T.contiguous()
        q, k, v = split_qkv(conv_ours(fn, x, wc, bc))
        k = F.normalize(k.float(), dim=-1, p=2).to(torch.float16)

        hid = torch.randn((T, d_model), device=DEV, dtype=torch.float16)
        Win = torch.randn((d_model, CONV_DIM), device=DEV, dtype=torch.float16)
        Wout = torch.randn((d_model, d_model), device=DEV, dtype=torch.float16)
        ov = torch.randn((T, d_model), device=DEV, dtype=torch.float16)
        for _ in range(30):
            torch.mm(hid, Win); torch.mm(ov, Wout); gdn_core(q, k, v, Beta, G, T)
        torch.npu.synchronize()
        t_in = bench(lambda: torch.mm(hid, Win))
        t_out = bench(lambda: torch.mm(ov, Wout))
        t_core = bench(lambda: gdn_core(q, k, v, Beta, G, T))
        lyr_before = t_in + t_acl + t_core + t_out
        lyr_after = t_in + t_our + t_core + t_out
        conv_pct = 100.0 * t_acl / lyr_before
        print(f"{T:>5} {t_in*1e6:>10.1f} {t_out*1e6:>11.1f} {lyr_before*1e6:>11.1f} "
              f"{lyr_after*1e6:>10.1f} {lyr_before/lyr_after:>7.2f}x {conv_pct:>6.1f}%")


if __name__ == "__main__":
    main()
