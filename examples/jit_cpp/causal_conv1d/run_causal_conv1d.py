"""Validate + benchmark the DEPTHWISE causal conv1d + bias + SiLU.

    # source the CANN environment first (sets ASCEND_TOOLKIT_HOME), then:
    ASCEND_RT_VISIBLE_DEVICES=0 python run_causal_conv1d.py

What this does
  1. CORRECTNESS: many shapes (edge / GDN / random), worst-case max error vs the
     torch depthwise reference (which dispatches to aclnnConvolution on NPU).
  2. PERF: batched sweep (B, seq, dim) with device-event timing; compare PTO vs
        - aclnnConvolution timed END-TO-END (torch F.conv1d depthwise + bias +
          SiLU, including the layout conversion a real [B,seq,dim]-fp16 pipeline
          pays — PTO does it fused from that layout), and
        - torch_npu.npu_fused_causal_conv1d (native Ascend fused causal conv;
          K fixed at 3, a 910_95-only op -> reported if unavailable).

Run on an otherwise-idle NPU (set ASCEND_RT_VISIBLE_DEVICES) for stable numbers.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from jit_util_causal_conv1d import jit_compile, K

torch.npu.set_device("npu")
torch.manual_seed(0)


def silu(z):
    return z * torch.sigmoid(z)


# ---------------------------------------------------------------------------
# Reference (torch depthwise causal conv1d + bias + SiLU)
#   On NPU, F.conv1d(groups=W) dispatches to aclnnConvolution.
# ---------------------------------------------------------------------------
def ref(x, w, bias):
    # x [L,W] fp16, w [K,W] fp32, bias [W] fp32 -> [L,W] fp16
    L, W = x.shape
    xt = F.pad(x.t().reshape(1, W, L).float(), (K - 1, 0))  # [1, W, L+K-1]
    weight = w.t().contiguous().reshape(W, 1, K)  # [W,1,K]
    z = F.conv1d(xt, weight, bias=bias.float(), groups=W)  # [1, W, L]
    return silu(z).reshape(W, L).t().contiguous().to(torch.float16)


def make_inputs(L, W, scale=1.0):
    x = scale * (2 * torch.rand((L, W), device="npu", dtype=torch.float16) - 1)
    w = torch.rand((K, W), device="npu", dtype=torch.float32) - 0.5
    bias = torch.rand((W,), device="npu", dtype=torch.float32) - 0.5
    return x, w, bias


# ---------------------------------------------------------------------------
# 1. CORRECTNESS
# ---------------------------------------------------------------------------
def correctness(fn):
    print(
        "== correctness (vs torch depthwise conv1d + bias + SiLU = aclnnConvolution) =="
    )
    cases = [
        # edge / tiny
        (1, 1),
        (1, 128),
        (2, 1),
        (1, 2048),
        (2, 2),
        (3, 64),
        (4, 8),
        (5, 130),
        (7, 1),
        (8, 1),
        (1, 4096),
        # boundary around lane width (128) and tile cap (MAX_W=3072)
        (16, 127),
        (16, 128),
        (16, 129),
        (16, 256),
        (16, 3071),
        (16, 3072),
        (16, 3073),
        (16, 6144),
        (16, 6145),
        # GDN shapes (W = H*D = 2048)
        (128, 2048),
        (256, 2048),
        (384, 2048),
        (512, 2048),
        # longer sequences
        (1024, 2048),
        (2048, 512),
        (4096, 256),
        # odd / non-aligned
        (200, 777),
        (333, 999),
        (17, 4099),
        (63, 1536),
        (129, 2049),
        # wider channels
        (32, 8192),
        (64, 16384),
        # L-chunk boundary stress (narrow W -> chunking active; L straddles
        # chunk sizes / LC_MIN=32 / halo K-1=3). Exercises the causal halo replay.
        (31, 256),
        (32, 256),
        (33, 256),
        (35, 256),
        (63, 256),
        (64, 256),
        (65, 256),
        (96, 256),
        (97, 256),
        (127, 1024),
        (128, 1024),
        (129, 1024),
        (255, 2048),
        (256, 2048),
        (257, 2048),
        (511, 512),
        (513, 512),
        (1000, 256),
        (1023, 2048),
        (1025, 2048),
        (3000, 512),
        (4095, 2048),
        (4097, 2048),
        (100, 128),
        (100, 129),
    ]
    # add randomized shapes (bounded element count) + varying input magnitude
    g = torch.Generator().manual_seed(1234)
    for _ in range(25):
        L = int(torch.randint(1, 1200, (1,), generator=g).item())
        W = int(torch.randint(1, 9000, (1,), generator=g).item())
        if L * W > 6_000_000:  # keep memory bounded
            W = max(1, 6_000_000 // L)
        cases.append((L, W))

    worst = 0.0
    worst_shape = None
    ok = 0
    for L, W in cases:
        for scale in (1.0, 8.0):  # also stress larger magnitudes
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
    print(
        f"  WORST max-abs-error = {worst:.3e}  at L={worst_shape[0]} "
        f"W={worst_shape[1]} scale={worst_shape[2]}\n"
    )


# ---------------------------------------------------------------------------
# native Ascend fused causal conv1d (K fixed = 3, 910_95-only op)
# ---------------------------------------------------------------------------
def try_native_fused(L, W):
    """Returns a zero-arg callable running npu_fused_causal_conv1d, or None."""
    try:
        Kn, dim, dt = 3, W, torch.float16
        weight = torch.rand(Kn, dim, dtype=dt).npu() - 0.5
        x = 2 * torch.rand(L, dim, dtype=dt).npu() - 1
        qsl = torch.tensor([0, L], dtype=torch.int32).npu()
        conv_states = torch.zeros(1, Kn - 1, dim, dtype=dt).npu()
        cidx = torch.tensor([0], dtype=torch.int32).npu()
        bias = torch.rand(dim, dtype=dt).npu() - 0.5

        def call():
            return torch_npu.npu_fused_causal_conv1d(
                x,
                weight,
                conv_states,
                query_start_loc=qsl,
                cache_indices=cidx,
                bias=bias,
                activation_mode="silu",
            )

        call()
        torch.npu.synchronize()
        return call
    except Exception as e:
        try_native_fused.err = repr(e)[:140]
        return None


# ---------------------------------------------------------------------------
# 2. PERFORMANCE  (batched; device-event timing; aclnn timed end-to-end)
# ---------------------------------------------------------------------------
def pick_iters(elems, target_us=250000.0):
    """Iteration count that keeps each timed run ~target_us of work: 100 where
    cheap, down to 3 for multi-GB shapes (rough ~100 GB/s estimate)."""
    est_per_call_us = elems / 25000.0
    return int(max(3, min(100, target_us / max(est_per_call_us, 1.0))))


def event_us(call, iters, runs=2):
    """Mean us/call over `runs` runs of `iters` back-to-back launches, timed with
    NPU events. Device-event (not wall-clock) timing keeps the ~200us host /
    dispatch floor and per-call allocation from dominating; bracketing many
    launches in one event pair (rather than taking a per-iter min) avoids
    latching a spurious near-zero reading off the async stream. `iters` adapts to
    shape size (pick_iters); warmup scales with it.
    """
    warmup = max(2, min(8, iters))
    for _ in range(warmup):
        call()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    run_avgs = []
    for _ in range(runs):
        start.record()
        for _ in range(iters):
            call()
        end.record()
        end.synchronize()
        run_avgs.append(start.elapsed_time(end) * 1e3 / iters)  # ms -> us/call
    return sum(run_avgs) / len(run_avgs)


# (B, seq, dim) — GDN matrix (dim 2048 = H*D, 6144 = q+k+v), sorted by B*seq*dim.
PERF_SHAPES = [
    (16, 512, 2048),
    (32, 512, 2048),
    (16, 1024, 2048),
    (16, 512, 6144),
    (64, 512, 2048),
    (32, 1024, 2048),
    (16, 2048, 2048),
    (32, 512, 6144),
    (16, 1024, 6144),
    (128, 512, 2048),
    (64, 1024, 2048),
    (32, 2048, 2048),
    (64, 512, 6144),
    (32, 1024, 6144),
    (16, 2048, 6144),
    (256, 512, 2048),
    (128, 1024, 2048),
    (64, 2048, 2048),
    (128, 512, 6144),
    (64, 1024, 6144),
    (32, 2048, 6144),
    (512, 512, 2048),
    (256, 1024, 2048),
    (128, 2048, 2048),
    (256, 512, 6144),
    (128, 1024, 6144),
    (64, 2048, 6144),
    (1024, 512, 2048),
    (512, 1024, 2048),
    (256, 2048, 2048),
    (512, 512, 6144),
    (256, 1024, 6144),
    (128, 2048, 6144),
    (2048, 512, 2048),
    (1024, 1024, 2048),
    (512, 2048, 2048),
    (1024, 512, 6144),
    (512, 1024, 6144),
    (256, 2048, 6144),
    (4096, 512, 2048),
    (2048, 1024, 2048),
    (1024, 2048, 2048),
    (2048, 512, 6144),
    (1024, 1024, 6144),
    (512, 2048, 6144),
]


def perf(fn):
    print("== performance sweep (batched, device-event avg-of-2, adaptive iters) ==")
    print(
        "   aclnn = aclnnConvolution timed END-TO-END from the [B,seq,dim] fp16\n"
        "   layout (transpose + pad + fp32 conv + SiLU + transpose-back + cast) —\n"
        "   the layout cost a real pipeline pays; PTO does it fused. SiLU on both.\n"
        "   aclnn's fp32 intermediates OOM past ~2GB; PTO (fused) still runs.\n"
    )
    native_ok = try_native_fused(128, 2048) is not None
    if not native_ok:
        print(
            f"  [native] npu_fused_causal_conv1d UNAVAILABLE on this box: "
            f"{getattr(try_native_fused, 'err', '?')}"
        )
        print(
            "  [native] (it is a 910_95-only op, K fixed=3; this box is Ascend910B2)\n"
        )

    hdr = (
        f"{'B*s*d':>7} {'seq':>5} {'dim':>5} {'B':>5} | {'PTO_us':>10} "
        f"{'aclnn_us':>10} {'speedup':>8} | {'GB/s':>6} {'it':>4} | {'maxerr':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for B, seq, dim in PERF_SHAPES:
        torch.npu.empty_cache()
        elems = B * seq * dim
        iters = pick_iters(elems)
        em = f"{elems // 1_000_000}M"
        try:
            x = 2 * torch.rand(B, seq, dim, device="npu", dtype=torch.float16) - 1
            w = torch.rand(K, dim, device="npu", dtype=torch.float32) - 0.5
            bias = torch.rand(dim, device="npu", dtype=torch.float32) - 0.5

            def ours(x=x, w=w, bias=bias):
                return fn.batched(x, w, bias, activation=True)

            t_ours = event_us(ours, iters)
            gbps = 4.0 * elems / (t_ours * 1e-6) / 1e9  # 2B in + 2B out per elem
        except RuntimeError as ex:
            print(f"{em:>7} {seq:>5} {dim:>5} {B:>5} | PTO err: {str(ex)[:40]}")
            torch.npu.empty_cache()
            continue

        t_aclnn = None
        err = None
        try:
            weight = w.t().contiguous().reshape(dim, 1, K)
            bf = bias.float()

            def aclnn(x=x, weight=weight, bf=bf, dim=dim):
                xt = F.pad(x.transpose(1, 2).float(), (K - 1, 0))
                z = F.conv1d(xt, weight, bias=bf, groups=dim)
                return silu(z).transpose(1, 2).contiguous().to(torch.float16)

            t_aclnn = event_us(aclnn, iters)
            y_pto = fn.batched(x, w, bias, activation=True)
            torch.npu.synchronize()
            ref = aclnn()
            err = (y_pto.float() - ref.float()).abs().max().item()
        except RuntimeError:
            torch.npu.empty_cache()

        acl_s = f"{t_aclnn:>10.1f}" if t_aclnn is not None else f"{'OOM':>10}"
        spd_s = f"{t_aclnn / t_ours:>7.2f}x" if t_aclnn is not None else f"{'n/a':>8}"
        err_s = f"{err:>8.2e}" if err is not None else f"{'n/a':>8}"
        print(
            f"{em:>7} {seq:>5} {dim:>5} {B:>5} | {t_ours:>10.1f} {acl_s} {spd_s} | "
            f"{gbps:>6.0f} {iters:>4} | {err_s}"
        )
    print()


def main():
    fn = jit_compile(
        str(Path(__file__).resolve().parent / "causal_conv1d_pto.cpp"), verbose=True
    )
    correctness(fn)
    perf(fn)


if __name__ == "__main__":
    main()
