"""Complete perf characterization of the depthwise fused conv1d+bias+SiLU.

Baseline ("aclnn") = torch F.conv1d(groups=W) [depthwise, causal left-pad] + bias
+ SiLU, which dispatches to aclnnConvolution on NPU -- the SAME math our kernel
does. speedup = aclnn_time / ours_time at the same shape.

Also reports: ours GB/s and % of the ~1758 GB/s practical HBM ceiling, and the
(col_w x num_wt x lchunks) work grid the kernel selects (Python mirror of the
in-kernel policy), so the 2-D tiling behaviour is visible.

    source ~/conv_pto/pto_env.sh
    cd ~/conv_pto/pto-kernels/examples/jit_cpp/conv1d
    ASCEND_RT_VISIBLE_DEVICES=5 python sweep_conv1d_dw.py
"""
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from jit_util_conv1d_dw import jit_compile, K, BLOCK_DIM

torch.npu.set_device("npu")
torch.manual_seed(0)

NUM_CORES = BLOCK_DIM * 2     # launch = cube_core_num*2 = 48 AIV blocks
MAX_W = 3072                  # UB tile cap in the kernel
LC_MIN = 32
OVH = 3000
HBM_CEIL = 1758.0            # GB/s, measured practical ceiling (torch silu)


def ceil(a, b):
    return (a + b - 1) // b


def roundup128(x):
    return ((x + 127) // 128) * 128


def grid(L, W):
    """Mirror of the in-kernel unified 2-D grid policy (for display)."""
    max_chunks = max(1, ceil(L, LC_MIN))
    cw_cap = max(128, (W * L) // OVH)
    nwt_ub = ceil(W, MAX_W)
    nwt_ovh = ceil(W, cw_cap)
    num_wt = max(nwt_ub, nwt_ovh, 1)
    if num_wt * max_chunks < NUM_CORES:
        num_wt = ceil(NUM_CORES, max_chunks)
    num_wt = max(1, min(num_wt, ceil(W, 128)))
    col_w = min(min(max(128, roundup128(ceil(W, num_wt))), MAX_W), W)
    num_wt = ceil(W, col_w)
    lchunks = min(max(1, ceil(NUM_CORES, num_wt)), max_chunks)
    return col_w, num_wt, lchunks


def ref(xt, weight, bf, W, L):
    z = F.conv1d(xt, weight, bias=bf, groups=W)
    s = z * torch.sigmoid(z)
    return s.reshape(W, L).t().contiguous().to(torch.float16)


def dev_time(call, it=60, wu=20):
    for _ in range(wu):
        call()
    torch.npu.synchronize()
    s = torch.npu.Event(True); e = torch.npu.Event(True)
    s.record()
    for _ in range(it):
        call()
    e.record(); torch.npu.synchronize()
    return s.elapsed_time(e) / it * 1e3   # us


def row(fn, L, W):
    x = (2 * torch.rand((L, W), device="npu", dtype=torch.float16) - 1)
    w = (torch.rand((K, W), device="npu", dtype=torch.float32) - 0.5)
    b = (torch.rand((W,), device="npu", dtype=torch.float32) - 0.5)
    xt = F.pad(x.t().reshape(1, W, L).float(), (K - 1, 0))
    weight = w.t().contiguous().reshape(W, 1, K)
    bf = b.float()
    t_our = dev_time(lambda: fn(x, w, b))
    t_acl = dev_time(lambda: ref(xt, weight, bf, W, L))
    gbps = 4.0 * L * W / (t_our * 1e-6) / 1e9
    cw, nwt, lc = grid(L, W)
    units = nwt * lc
    g = f"{cw}x{nwt}x{lc}={units}u"
    print(f"{L:>5} {W:>6} {g:>16} {t_our:>8.1f} {t_acl:>9.1f} {t_acl/t_our:>7.2f}x "
          f"{gbps:>8.1f} {100*gbps/HBM_CEIL:>6.1f}%")


def main():
    fn = jit_compile(str(Path(__file__).resolve().parent / "conv1d_dw_pto.cpp"),
                     verbose=False)
    hdr = (f"{'L':>5} {'W':>6} {'grid(cw x nwt x lc)':>16} {'ours_us':>8} "
           f"{'aclnn_us':>9} {'speedup':>8} {'GB/s':>8} {'%peak':>6}")

    print("\n#### L-sweep at W=2048 (GDN channel width = H*D) ####")
    print(hdr)
    for L in (1, 4, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192):
        row(fn, L, 2048)

    print("\n#### W-sweep at L=512 (GDN seq len) ####")
    print(hdr)
    for W in (128, 256, 512, 1024, 2048, 4096, 6144, 8192, 16384):
        row(fn, 512, W)

    print("\n#### 2-D grid (L x W) ####")
    print(hdr)
    for L in (128, 512, 2048, 4096):
        for W in (512, 2048, 4096, 8192):
            row(fn, L, W)


if __name__ == "__main__":
    main()
