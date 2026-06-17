"""Validate + benchmark the DEPTHWISE causal conv1d + bias + SiLU.

    source ~/conv_pto/pto_env.sh
    cd ~/conv_pto/pto-kernels/examples/jit_cpp/conv1d
    ASCEND_RT_VISIBLE_DEVICES=5 python run_conv1d_dw.py

What this does
  1. CORRECTNESS: many shapes (edge / GDN / random), worst-case max error vs the
     torch depthwise reference (which dispatches to aclnnConvolution on NPU).
  2. PERF: sweep (L, W); compare our kernel vs
        - aclnnConvolution  (torch F.conv1d depthwise + bias + SiLU), and
        - torch_npu.npu_fused_causal_conv1d  (the native Ascend fused causal conv;
          K is fixed at 3 and it is a 910_95-only op -> reported if unavailable).
"""
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from jit_util_conv1d_dw import jit_compile, K

torch.npu.set_device("npu")
torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Reference (torch depthwise causal conv1d + bias + SiLU)
#   On NPU, F.conv1d(groups=W) dispatches to aclnnConvolution.
# ---------------------------------------------------------------------------
def ref(x, w, bias):
    # x [L,W] fp16, w [K,W] fp32, bias [W] fp32 -> [L,W] fp16
    L, W = x.shape
    xt = F.pad(x.t().reshape(1, W, L).float(), (K - 1, 0))      # [1, W, L+K-1]
    weight = w.t().contiguous().reshape(W, 1, K)                 # [W,1,K]
    z = F.conv1d(xt, weight, bias=bias.float(), groups=W)        # [1, W, L]
    silu = z * torch.sigmoid(z)
    return silu.reshape(W, L).t().contiguous().to(torch.float16)


def make_inputs(L, W, scale=1.0):
    x = scale * (2 * torch.rand((L, W), device="npu", dtype=torch.float16) - 1)
    w = (torch.rand((K, W), device="npu", dtype=torch.float32) - 0.5)
    bias = (torch.rand((W,), device="npu", dtype=torch.float32) - 0.5)
    return x, w, bias


# ---------------------------------------------------------------------------
# 1. CORRECTNESS
# ---------------------------------------------------------------------------
def correctness(fn):
    print("== correctness (vs torch depthwise conv1d + bias + SiLU = aclnnConvolution) ==")
    cases = [
        # edge / tiny
        (1, 1), (1, 128), (2, 1), (1, 2048), (2, 2), (3, 64), (4, 8),
        (5, 130), (7, 1), (8, 1), (1, 4096),
        # boundary around lane width (128) and tile cap (MAX_W=3072)
        (16, 127), (16, 128), (16, 129), (16, 256), (16, 3071), (16, 3072),
        (16, 3073), (16, 6144), (16, 6145),
        # GDN shapes (W = H*D = 2048)
        (128, 2048), (256, 2048), (384, 2048), (512, 2048),
        # longer sequences
        (1024, 2048), (2048, 512), (4096, 256),
        # odd / non-aligned
        (200, 777), (333, 999), (17, 4099), (63, 1536), (129, 2049),
        # wider channels
        (32, 8192), (64, 16384),
        # L-chunk boundary stress (narrow W -> chunking active; L straddles
        # chunk sizes / LC_MIN=32 / halo K-1=3). Exercises the causal halo replay.
        (31, 256), (32, 256), (33, 256), (35, 256), (63, 256), (64, 256),
        (65, 256), (96, 256), (97, 256), (127, 1024), (128, 1024), (129, 1024),
        (255, 2048), (256, 2048), (257, 2048), (511, 512), (513, 512),
        (1000, 256), (1023, 2048), (1025, 2048), (3000, 512), (4095, 2048),
        (4097, 2048), (100, 128), (100, 129),
    ]
    # add randomized shapes (bounded element count) + varying input magnitude
    g = torch.Generator().manual_seed(1234)
    for _ in range(25):
        L = int(torch.randint(1, 1200, (1,), generator=g).item())
        W = int(torch.randint(1, 9000, (1,), generator=g).item())
        if L * W > 6_000_000:           # keep memory bounded
            W = max(1, 6_000_000 // L)
        cases.append((L, W))

    worst = 0.0
    worst_shape = None
    ok = 0
    for (L, W) in cases:
        for scale in (1.0, 8.0):        # also stress larger magnitudes
            x, w, bias = make_inputs(L, W, scale=scale)
            y = fn(x, w, bias)
            torch.npu.synchronize()
            yr = ref(x, w, bias)
            err = (y.float() - yr.float()).abs().max().item()
            if err > worst:
                worst, worst_shape = err, (L, W, scale)
            assert err < 5e-2, f"FAIL L={L} W={W} scale={scale} err={err}"
            ok += 1
    print(f"  {ok} cases passed (over {len(cases)} shapes x 2 magnitudes)")
    print(f"  WORST max-abs-error = {worst:.3e}  at L={worst_shape[0]} "
          f"W={worst_shape[1]} scale={worst_shape[2]}\n")


# ---------------------------------------------------------------------------
# native Ascend fused causal conv1d (K fixed = 3, 910_95-only op)
# ---------------------------------------------------------------------------
def try_native_fused(L, W):
    """Returns a zero-arg callable running npu_fused_causal_conv1d, or None."""
    try:
        Kn, dim, dt = 3, W, torch.float16
        weight = (torch.rand(Kn, dim, dtype=dt).npu() - 0.5)
        x = (2 * torch.rand(L, dim, dtype=dt).npu() - 1)
        qsl = torch.tensor([0, L], dtype=torch.int32).npu()
        conv_states = torch.zeros(1, Kn - 1, dim, dtype=dt).npu()
        cidx = torch.tensor([0], dtype=torch.int32).npu()
        bias = (torch.rand(dim, dtype=dt).npu() - 0.5)

        def call():
            return torch_npu.npu_fused_causal_conv1d(
                x, weight, conv_states, query_start_loc=qsl,
                cache_indices=cidx, bias=bias, activation_mode="silu")
        call()
        torch.npu.synchronize()
        return call
    except Exception as e:
        try_native_fused.err = repr(e)[:140]
        return None


# ---------------------------------------------------------------------------
# 2. PERFORMANCE
# ---------------------------------------------------------------------------
def bench(call, iters=100, warmup=20):
    for _ in range(warmup):
        call()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        call()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters


def perf(fn):
    print("== performance sweep ==")
    # probe native op once
    native_ok = try_native_fused(128, 2048) is not None
    if not native_ok:
        print(f"  [native] npu_fused_causal_conv1d UNAVAILABLE on this box: "
              f"{getattr(try_native_fused, 'err', '?')}")
        print(f"  [native] (it is a 910_95-only op, K fixed=3; this box is Ascend910B2)\n")

    hdr = f"{'L':>5} {'W':>6} {'ours_us':>9} {'aclnn_us':>9} {'native_us':>10} " \
          f"{'ours GB/s':>10} {'vs aclnn':>9}"
    print(hdr)
    shapes = [(128, 2048), (256, 2048), (384, 2048), (512, 2048),
              (1024, 2048), (256, 4096), (256, 8192), (512, 4096),
              (2048, 2048), (4096, 2048)]
    for (L, W) in shapes:
        x, w, bias = make_inputs(L, W)
        xt = F.pad(x.t().reshape(1, W, L).float(), (K - 1, 0))
        weight = w.t().contiguous().reshape(W, 1, K)
        bf = bias.float()

        t_ours = bench(lambda: fn(x, w, bias))
        t_aclnn = bench(lambda: (lambda z: z * torch.sigmoid(z))(
            F.conv1d(xt, weight, bias=bf, groups=W)))

        native_call = try_native_fused(L, W)
        t_native = bench(native_call) if native_call is not None else None
        native_str = f"{t_native*1e6:>10.1f}" if t_native is not None else f"{'n/a':>10}"

        gbps = 4.0 * L * W / t_ours / 1e9      # 2B in + 2B out per elem
        print(f"{L:>5} {W:>6} {t_ours*1e6:>9.1f} {t_aclnn*1e6:>9.1f} {native_str} "
              f"{gbps:>10.1f} {t_aclnn/t_ours:>8.2f}x")
    print()


def main():
    fn = jit_compile(str(Path(__file__).resolve().parent / "conv1d_dw_pto.cpp"),
                     verbose=True)
    correctness(fn)
    perf(fn)


if __name__ == "__main__":
    main()
