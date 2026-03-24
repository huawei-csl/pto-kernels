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

    indices = torch.arange(total_len, dtype=torch.float32, device=device)
    gm_input = (indices // (n * sub_block_dim)) * sub_block_dim
    gm_output = torch.zeros(total_len, dtype=torch.float32, device=device)
    ref_result = indices // n
    stream_ptr = torch.npu.current_stream()._as_parameter_

    lib.call_kernel(
        block_dim,
        stream_ptr,
        ctypes.c_void_p(gm_input.data_ptr()),
        ctypes.c_void_p(gm_output.data_ptr()),
        n,
    )
    torch.npu.synchronize()

    correct = torch.equal(gm_output, ref_result)
    print(f"[{label}] launch: OK, numeric_correct: {correct}")
    if not correct:
        print(f"  ref tail:    {ref_result[-16:]}")
        print(f"  output tail: {gm_output[-16:]}")
    return correct


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repros = [
        (
            "TSYNC repro",
            os.path.join(here, "repro_tsync_issue_lib.so"),
            "expected behavior: WRONG path (TSync_Custom C2V record path)",
        ),
        (
            "TPipe repro (native)",
            os.path.join(here, "repro_tpipe_native_lib.so"),
            "expected behavior: WRONG path in full kernel (native Producer::record uses PIPE_FIX)",
        ),
        (
            "TPipe repro (workaround)",
            os.path.join(here, "repro_tpipe_workaround_lib.so"),
            "expected behavior: CORRECT path (custom Producer::record uses PIPE_MTE3)",
        ),
    ]

    print("=== Expected behavior matrix ===")
    for label, _, expected in repros:
        print(f"- [{label}] {expected}")

    print("\n=== Runtime launch + numeric check ===")
    launch_failed = False
    numeric_results = []
    for label, so, expected in repros:
        try:
            print(f"[{label}] {expected}")
            numeric_correct = run_one(label, so)
            numeric_results.append((label, numeric_correct))
        except Exception as e:
            launch_failed = True
            print(f"[{label}] launch: FAIL ({e})")

    if launch_failed:
        sys.exit(1)

    print("\n=== Numeric summary ===")
    for label, numeric_correct in numeric_results:
        state = "CORRECT" if numeric_correct else "WRONG"
        print(f"- [{label}] observed numeric behavior: {state}")
    print("All repro launches completed.")


if __name__ == "__main__":
    main()
