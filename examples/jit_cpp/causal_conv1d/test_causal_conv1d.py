"""Correctness tests for the depthwise causal conv1d + bias + SiLU PTO kernel.

Run:
    pytest test_causal_conv1d.py -q --npu npu:0
"""

import ctypes
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from jit_util_causal_conv1d import BLOCK_DIM, K, compile_cpp, jit_compile

KERNEL_CPP = Path(__file__).resolve().parent / "causal_conv1d_pto.cpp"

DTYPES = [torch.float16, torch.bfloat16]
TEST_SEEDS = [0, 1]
# (batch, seq, dim): small general shapes + the GDN prefill regime
# (dim 2048 = H*D, 6144 = q+k+v; seq 128..512).
TEST_CASES = [
    (1, 16, 256),
    (2, 31, 256),
    (1, 128, 2048),
    (8, 128, 2048),
    (1, 256, 2048),
    (8, 384, 2048),
    (1, 512, 2048),
    (8, 512, 2048),
    (1, 128, 6144),
    (8, 256, 6144),
    (1, 384, 6144),
    (8, 512, 6144),
]
# max abs error tolerance vs the fp32 reference (dtype rounding only).
TOL = {torch.float16: 6e-3, torch.bfloat16: 6e-2}


def causal_conv1d_ref(x, w, bias, activation):
    """Depthwise causal conv1d (width K) + per-channel bias + optional SiLU.

    fp32 accumulate; x[:, <0] padded with zeros (no conv_states).
    x: [B, L, W]   w: [K, W]   bias: [W]   -> [B, L, W] (fp32)
    """
    B, L, W = x.shape
    pad = torch.zeros((B, K - 1, W), device=x.device, dtype=x.dtype)
    xe = torch.cat([pad, x], dim=1).float()
    wf = w.float()
    acc = sum(xe[:, k : k + L] * wf[k] for k in range(K)) + bias.float()
    return F.silu(acc) if activation else acc


@pytest.fixture(scope="session")
def conv1d_kernel(npu_device):
    return jit_compile(str(KERNEL_CPP), verbose=False)


@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("activation", [True, False])
@pytest.mark.parametrize("batch,seq,dim", TEST_CASES)
def test_batched_matches_reference(
    conv1d_kernel, npu_device, seed, dtype, activation, batch, seq, dim
):
    torch.manual_seed(seed)
    x = 2 * torch.rand(batch, seq, dim, device=npu_device, dtype=dtype) - 1
    w = torch.rand(K, dim, device=npu_device, dtype=torch.float32) - 0.5
    bias = torch.rand(dim, device=npu_device, dtype=torch.float32) - 0.5

    y = conv1d_kernel.batched(x, w, bias, activation=activation)
    torch.npu.synchronize()

    ref = causal_conv1d_ref(x, w, bias, activation)
    err = (y.float() - ref).abs().max().item()
    assert err <= TOL[dtype], f"max abs err {err:.3e} > tol {TOL[dtype]:.1e}"


@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize("seq,dim", [(16, 256), (128, 2048), (512, 6144)])
def test_single_fp16_entry_matches_reference(conv1d_kernel, npu_device, seed, seq, dim):
    """The non-batched [L, W] fp16 entry (always applies SiLU)."""
    torch.manual_seed(seed)
    x = 2 * torch.rand(seq, dim, device=npu_device, dtype=torch.float16) - 1
    w = torch.rand(K, dim, device=npu_device, dtype=torch.float32) - 0.5
    bias = torch.rand(dim, device=npu_device, dtype=torch.float32) - 0.5

    y = conv1d_kernel(x, w, bias)
    torch.npu.synchronize()

    ref = causal_conv1d_ref(x.unsqueeze(0), w, bias, activation=True)[0]
    err = (y.float() - ref).abs().max().item()
    assert err <= TOL[torch.float16], f"max abs err {err:.3e}"


# --- generalized K / MAX_W: compile a parametric wrapper per (K, MAX_W) that
# #includes the kernel and instantiates runConvSiluBatched at that K, leaving the
# K=4 production entry points untouched. Covers non-power-of-two K (3, 5),
# power-of-two K (2, 8), MAX_W variations, and both fp16 + bf16 I/O.
KCONFIGS = [(2, 3072), (3, 3072), (4, 2048), (5, 2048), (8, 1536)]

# Wrapper source: include the kernel + add parametric fp16 and bf16
# entries/launchers (weights/bias stay fp32; only the I/O element type changes).
_WRAPPER_SRC = """#include "{kernel}"
extern "C" __global__ AICORE void causal_conv1d_test_kernel_fp16(
    __gm__ uint8_t* x, __gm__ uint8_t* y, __gm__ uint8_t* wgt,
    __gm__ uint8_t* bia, uint32_t batch, uint32_t seqLen, uint32_t channels,
    uint32_t activation) {{
#if defined(__DAV_VEC__)
  constexpr uint32_t K = {k}, MAX_W = {mw};
  csilu::runConvSiluBatched<half, float, K, MAX_W>(
      (__gm__ half*)x, (__gm__ half*)y, (__gm__ float*)wgt, (__gm__ float*)bia,
      batch, seqLen, channels, activation);
#else
  (void)x; (void)y; (void)wgt; (void)bia; (void)batch; (void)seqLen;
  (void)channels; (void)activation;
#endif
}}
extern "C" __global__ AICORE void causal_conv1d_test_kernel_bf16(
    __gm__ uint8_t* x, __gm__ uint8_t* y, __gm__ uint8_t* wgt,
    __gm__ uint8_t* bia, uint32_t batch, uint32_t seqLen, uint32_t channels,
    uint32_t activation) {{
#if defined(__DAV_VEC__)
  constexpr uint32_t K = {k}, MAX_W = {mw};
  csilu::runConvSiluBatched<bfloat16_t, float, K, MAX_W>(
      (__gm__ bfloat16_t*)x, (__gm__ bfloat16_t*)y, (__gm__ float*)wgt,
      (__gm__ float*)bia, batch, seqLen, channels, activation);
#else
  (void)x; (void)y; (void)wgt; (void)bia; (void)batch; (void)seqLen;
  (void)channels; (void)activation;
#endif
}}
extern "C" void call_test_kernel_fp16(uint32_t blockDim, void* stream,
    uint8_t* x, uint8_t* y, uint8_t* wgt, uint8_t* bia, uint32_t batch,
    uint32_t seqLen, uint32_t channels, uint32_t activation) {{
  causal_conv1d_test_kernel_fp16<<<blockDim * 2, nullptr, stream>>>(
      x, y, wgt, bia, batch, seqLen, channels, activation);
}}
extern "C" void call_test_kernel_bf16(uint32_t blockDim, void* stream,
    uint8_t* x, uint8_t* y, uint8_t* wgt, uint8_t* bia, uint32_t batch,
    uint32_t seqLen, uint32_t channels, uint32_t activation) {{
  causal_conv1d_test_kernel_bf16<<<blockDim * 2, nullptr, stream>>>(
      x, y, wgt, bia, batch, seqLen, channels, activation);
}}
"""
_BATCHED_ARGS = (
    [ctypes.c_uint32, ctypes.c_void_p] + [ctypes.c_void_p] * 4 + [ctypes.c_uint32] * 4
)


def _ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def causal_conv1d_ref_k(x, w, bias, k_width):
    """fp32 reference for an arbitrary filter width k_width (always SiLU)."""
    B, L, W = x.shape
    pad = torch.zeros((B, k_width - 1, W), device=x.device, dtype=x.dtype)
    xe = torch.cat([pad, x], dim=1).float()
    wf = w.float()
    acc = sum(xe[:, k : k + L] * wf[k] for k in range(k_width)) + bias.float()
    return F.silu(acc)


@pytest.fixture(scope="session")
def kconfig_libs(npu_device):
    """Compile a parametric wrapper per (K, MAX_W) and load it."""
    gen_dir = KERNEL_CPP.parent / "outputs" / "gen"
    gen_dir.mkdir(parents=True, exist_ok=True)
    libs = {}
    for k_width, max_w in KCONFIGS:
        src = gen_dir / f"wrap_k{k_width}_mw{max_w}.cpp"
        src.write_text(_WRAPPER_SRC.format(kernel=KERNEL_CPP, k=k_width, mw=max_w))
        so = compile_cpp(str(src), out_name=f"conv1d_k{k_width}_mw{max_w}.so")
        libs[(k_width, max_w)] = ctypes.CDLL(so)
    return libs


@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("k_width,max_w", KCONFIGS)
@pytest.mark.parametrize(
    "batch,seq,dim",
    [(1, 64, 256), (2, 128, 2048), (8, 256, 2048), (1, 384, 6144), (4, 200, 777)],
)
def test_general_k_max_w(
    kconfig_libs, npu_device, seed, dtype, k_width, max_w, batch, seq, dim
):
    """Kernel built with different K / MAX_W, fp16 + bf16. dim > MAX_W tiles W."""
    torch.manual_seed(seed)
    x = 2 * torch.rand(batch, seq, dim, device=npu_device, dtype=dtype) - 1
    w = torch.rand(k_width, dim, device=npu_device, dtype=torch.float32) - 0.5
    bias = torch.rand(dim, device=npu_device, dtype=torch.float32) - 0.5
    y = torch.empty_like(x)

    lib = kconfig_libs[(k_width, max_w)]
    fn = (
        lib.call_test_kernel_fp16
        if dtype == torch.float16
        else lib.call_test_kernel_bf16
    )
    fn.argtypes = _BATCHED_ARGS
    fn.restype = None
    stream = torch.npu.current_stream()._as_parameter_  # noqa: SLF001
    fn(BLOCK_DIM, stream, _ptr(x), _ptr(y), _ptr(w), _ptr(bias), batch, seq, dim, 1)
    torch.npu.synchronize()

    ref = causal_conv1d_ref_k(x, w, bias, k_width)
    err = (y.float() - ref).abs().max().item()
    assert err <= TOL[dtype], (
        f"K={k_width} MAX_W={max_w} dt={dtype} B={batch} L={seq} W={dim}: "
        f"max abs err {err:.3e}"
    )
