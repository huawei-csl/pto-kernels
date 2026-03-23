import ctypes
import os
import subprocess
import sys

import torch
import torch_npu  # noqa: F401


# ---------------------------------------------------------------------------
# Compilation helpers
# ---------------------------------------------------------------------------

def compile_kernel(src_cpp: str, out_so: str, verbose: bool = False) -> None:
    ascend = os.environ["ASCEND_TOOLKIT_HOME"]
    pto = os.environ["PTO_LIB_PATH"]

    flags = [
        "-fPIC", "-shared", "-xcce", "-O2", "-std=c++17",
        "--npu-arch=dav-2201",
        "-DMEMORY_BASE",
        f"-I{pto}/include",
        f"-I{ascend}/include",
        f"-I{ascend}/pkg_inc",
        f"-I{ascend}/pkg_inc/runtime",
    ]
    cmd = ["bisheng", *flags, src_cpp, "-o", out_so]
    if verbose:
        print("compile:", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Library loader — same call_kernel ABI for both versions
# ---------------------------------------------------------------------------

def load_lib(so_path: str):
    lib = ctypes.CDLL(os.path.abspath(so_path))
    lib.call_kernel.argtypes = [
        ctypes.c_uint32,   # blockDim
        ctypes.c_void_p,   # stream
        ctypes.c_void_p,   # gm_input
        ctypes.c_void_p,   # gm_output
        ctypes.c_int,      # N
    ]
    lib.call_kernel.restype = None

    def run_kernel(gm_input, gm_output, N, block_dim, stream_ptr=None):
        if stream_ptr is None:
            stream_ptr = torch.npu.current_stream()._as_parameter_
        lib.call_kernel(
            block_dim,
            stream_ptr,
            ctypes.c_void_p(gm_input.data_ptr()),
            ctypes.c_void_p(gm_output.data_ptr()),
            N,
        )

    return run_kernel


# ---------------------------------------------------------------------------
# Test logic (identical to the original run_sync_c2v.py)
# ---------------------------------------------------------------------------

def test_kernel(run_kernel, label: str, device: str = "npu") -> None:
    torch.npu.set_device(device)

    try:
        BLOCK_DIM = int(torch.npu.get_device_properties(device).cube_core_num)
    except Exception:
        BLOCK_DIM = 24

    N = 16384
    SUB_BLOCK_DIM = 2
    TOTAL_LEN = BLOCK_DIM * SUB_BLOCK_DIM * N

    indices = torch.arange(TOTAL_LEN, dtype=torch.float32, device=device)
    gm_input = (indices // (N * SUB_BLOCK_DIM)) * SUB_BLOCK_DIM
    gm_output = torch.empty(TOTAL_LEN, dtype=torch.float32, device=device)
    ref_result = indices // N

    run_kernel(gm_input, gm_output, N, BLOCK_DIM)
    torch.npu.synchronize()

    correct = torch.equal(gm_output, ref_result)
    print(f"[{label}] correct: {correct}")
    if not correct:
        print(f"  ref:    {ref_result[:16]}")
        print(f"  output: {gm_output[:16]}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))

VERSIONS = [
    ("TSYNC",       "sync_c2v_tsync.cpp",    "sync_c2v_tsync_lib.so"),
    ("TPUSH/TPOP",  "sync_c2v_tpushpop.cpp", "sync_c2v_tpushpop_lib.so"),
]

for label, src, so in VERSIONS:
    src_path = os.path.join(HERE, src)
    so_path  = os.path.join(HERE, so)

    print(f"\n=== {label}: compiling {src} ===")
    compile_kernel(src_path, so_path, verbose=True)

    run_kernel = load_lib(so_path)
    test_kernel(run_kernel, label)

    os.remove(so_path)

print("\nAll versions passed.")
