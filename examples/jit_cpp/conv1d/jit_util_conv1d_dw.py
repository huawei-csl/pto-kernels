import ctypes
import os
import subprocess

import torch

ASCEND_TOOLKIT_HOME = os.environ["ASCEND_TOOLKIT_HOME"]
PTO_LIB_PATH = os.environ.get("PTO_LIB_PATH", ASCEND_TOOLKIT_HOME)
BLOCK_DIM = int(getattr(torch.npu.get_device_properties("npu:0"), "cube_core_num", 20))
K = 4


def compile_cpp(
    kernel_cpp: str,
    verbose: bool = False,
    timeout: int = 120,
    extra_flags=None,
    out_name: str = "conv1d_dw_jit.so",
) -> str:
    so_dir = os.path.join(os.path.dirname(kernel_cpp), "outputs", "so")
    os.makedirs(so_dir, exist_ok=True)
    lib_path = os.path.join(so_dir, out_name)
    flags = [
        "-fPIC",
        "-shared",
        "-xcce",
        "-DMEMORY_BASE",
        "-O2",
        "-std=c++17",
        "-Wno-ignored-attributes",
        "--cce-aicore-arch=dav-c220-vec",
        "-isystem",
        f"{PTO_LIB_PATH}/include",
    ]
    if extra_flags:
        flags += list(extra_flags)
    command = ["bisheng", *flags, kernel_cpp, "-o", lib_path]
    if verbose:
        print("compile:", " ".join(command))
    try:
        subprocess.run(command, timeout=timeout, check=True)
    except Exception as e:
        raise RuntimeError(f"Compile failed: {e}") from e
    return lib_path


def _p(t):
    return ctypes.c_void_p(t.data_ptr())


def load_lib(lib_path):
    lib = ctypes.CDLL(os.path.abspath(lib_path))
    # call_kernel(blockDim, stream, x, y, wgt, bia, L_in, W)
    lib.call_kernel.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    lib.call_kernel.restype = None

    # batched entries: call_kernel_batched(blockDim, stream, x,y,wgt,bia, batch,seqLen,W,activation)
    for name in ("call_kernel_batched", "call_kernel_batched_bf16"):
        getattr(lib, name).argtypes = [
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        getattr(lib, name).restype = None

    def conv1d_dw(x, w, bias, *, block_dim=BLOCK_DIM, stream_ptr=None):
        """Depthwise causal conv1d + bias + SiLU.
        x: [L, W] fp16 (contig).  w: [K, W] fp32 (contig).  bias: [W] fp32.
        Returns y: [L, W] fp16.
        """
        assert x.dim() == 2 and x.dtype == torch.float16 and x.is_contiguous()
        L_in, W = int(x.shape[0]), int(x.shape[1])
        assert w.shape == (K, W) and w.dtype == torch.float32 and w.is_contiguous()
        assert (
            bias.shape == (W,) and bias.dtype == torch.float32 and bias.is_contiguous()
        )
        y = torch.empty((L_in, W), device=x.device, dtype=torch.float16)
        if stream_ptr is None:
            stream_ptr = torch.npu.current_stream()._as_parameter_  # noqa: SLF001
        lib.call_kernel(block_dim, stream_ptr, _p(x), _p(y), _p(w), _p(bias), L_in, W)
        return y

    def conv1d_dw_batched(
        x, w, bias, *, activation=True, block_dim=BLOCK_DIM, stream_ptr=None
    ):
        """Depthwise causal conv1d + bias + (optional) SiLU, batched.
        x: [batch, seqLen, W] fp16 OR bf16 (contig).  w: [K, W] fp32.  bias: [W] fp32.
        Returns y: same shape/dtype as x.  x[<0]=0 per sequence (no conv_states).
        """
        assert x.dim() == 3 and x.is_contiguous()
        batch, seqLen, W = (int(v) for v in x.shape)
        assert w.shape == (K, W) and w.dtype == torch.float32 and w.is_contiguous()
        assert (
            bias.shape == (W,) and bias.dtype == torch.float32 and bias.is_contiguous()
        )
        y = torch.empty_like(x)
        if stream_ptr is None:
            stream_ptr = torch.npu.current_stream()._as_parameter_  # noqa: SLF001
        act = 1 if activation else 0
        if x.dtype == torch.float16:
            lib.call_kernel_batched(
                block_dim,
                stream_ptr,
                _p(x),
                _p(y),
                _p(w),
                _p(bias),
                batch,
                seqLen,
                W,
                act,
            )
        elif x.dtype == torch.bfloat16:
            lib.call_kernel_batched_bf16(
                block_dim,
                stream_ptr,
                _p(x),
                _p(y),
                _p(w),
                _p(bias),
                batch,
                seqLen,
                W,
                act,
            )
        else:
            raise TypeError(f"unsupported dtype {x.dtype}")
        return y

    conv1d_dw.batched = conv1d_dw_batched
    return conv1d_dw


def jit_compile(src_path, verbose=True):
    return load_lib(compile_cpp(src_path, verbose=verbose))
