import ctypes
import os
import sys

import torch
import torch_npu  # noqa: F401


def load_lib(so_path: str):
    lib = ctypes.CDLL(os.path.abspath(so_path))
    lib.call_kernel.argtypes = [
        ctypes.c_uint32,  # blockDim
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # gm_input
        ctypes.c_void_p,  # gm_output
        ctypes.c_int,     # N
    ]
    lib.call_kernel.restype = None
    return lib


def run_one(label: str, so_path: str, device: str = "npu"):
    if not os.path.exists(so_path):
        raise FileNotFoundError(f"{so_path} not found. Run compile.sh first.")

    lib = load_lib(so_path)

    torch.npu.set_device(device)
    try:
        block_dim = int(torch.npu.get_device_properties(device).cube_core_num)
    except Exception:
        block_dim = 24

    n = 16384
    sub_block_dim = 2
    total_len = block_dim * sub_block_dim * n

    gm_input = torch.arange(total_len, dtype=torch.float32, device=device)
    gm_output = torch.zeros(total_len, dtype=torch.float32, device=device)
    stream_ptr = torch.npu.current_stream()._as_parameter_

    lib.call_kernel(
        block_dim,
        stream_ptr,
        ctypes.c_void_p(gm_input.data_ptr()),
        ctypes.c_void_p(gm_output.data_ptr()),
        n,
    )
    torch.npu.synchronize()
    print(f"[{label}] launch: OK")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repros = [
        ("TSYNC repro", os.path.join(here, "repro_tsync_issue_lib.so")),
        ("TPipe repro (native)", os.path.join(here, "repro_tpipe_native_lib.so")),
        ("TPipe repro (workaround)", os.path.join(here, "repro_tpipe_workaround_lib.so")),
    ]

    failed = False
    for label, so in repros:
        try:
            run_one(label, so)
        except Exception as e:
            failed = True
            print(f"[{label}] launch: FAIL ({e})")

    if failed:
        sys.exit(1)
    print("All repro launches completed.")


if __name__ == "__main__":
    main()
