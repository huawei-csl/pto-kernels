"""
run_repro.py — Bug 1 reproducer: TSync_Custom::record() wrong output

Test setup
----------
  gm_input  : all 100.0f   (cube copies this to gm_output)
  gm_output : all   0.0f   (zero-initialised; stale value if sync is broken)

  For each cube block b and its two vector sub-cores (subblockid 0 and 1):
    Cube  : copies gm_input[b*2N .. (b+1)*2N] → gm_output[same range]
    Vector: waits for cube signal, then adds subblockid to every element

  Expected output : 100.0 + subblockid
    sub 0 → 100.0,  sub 1 → 101.0

  Observed output with TSync_Custom (buggy):
    sub 0 →   0.0,  sub 1 →   1.0   (stale read — sync never fired correctly)

Run after compile.sh:
  python run_repro.py
"""

import ctypes
import os
import sys

import torch
import torch_npu  # noqa: F401


DEVICE = "npu"
# N must be a multiple of 256 (one row-burst = 256 floats = 1 KB).
N = 256
SUB_BLOCK_DIM = 2


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


def make_tensors(block_dim: int):
    """Return (gm_input, gm_output, ref_output) for a given blockDim."""
    total = block_dim * SUB_BLOCK_DIM * N

    # Cube copies gm_input to gm_output; vector adds subblockid afterwards.
    # gm_input: 100.0 for every element (cube owns 2*N per block, both sub-cores).
    gm_input = torch.full((total,), 100.0, dtype=torch.float32, device=DEVICE)

    # gm_output: zero-initialised — stale value if vector reads before cube writes.
    gm_output = torch.zeros(total, dtype=torch.float32, device=DEVICE)

    # Expected: 100.0 + subblockid, tiled across blocks.
    # Layout per block: [sub0: N × 100.0] [sub1: N × 101.0]
    per_block = torch.cat([
        torch.full((N,), 100.0, dtype=torch.float32),
        torch.full((N,), 101.0, dtype=torch.float32),
    ])
    ref_output = per_block.repeat(block_dim).to(DEVICE)

    return gm_input, gm_output, ref_output


def run(lib, block_dim: int, gm_input, gm_output):
    stream = torch.npu.current_stream()._as_parameter_
    lib.call_kernel(
        block_dim,
        stream,
        ctypes.c_void_p(gm_input.data_ptr()),
        ctypes.c_void_p(gm_output.data_ptr()),
        N,
    )
    torch.npu.synchronize()


def report(label: str, output, ref):
    passed = torch.equal(output, ref)
    status = "PASS" if passed else "FAIL (sync bug)"
    print(f"  [{status}]  {label}")
    if not passed:
        # Show the first differing block to make the pattern obvious.
        n = N
        got_sub0  = output[:n]
        got_sub1  = output[n:2*n]
        exp_sub0  = ref[:n]
        exp_sub1  = ref[n:2*n]
        print(f"    sub 0 — expected {exp_sub0[0].item():.1f}, "
              f"got {got_sub0[0].item():.1f}")
        print(f"    sub 1 — expected {exp_sub1[0].item():.1f}, "
              f"got {got_sub1[0].item():.1f}")
        print(f"    (if got = 0.0/1.0 the vector read stale zero-filled GM)")


# ---------------------------------------------------------------------------

torch.npu.set_device(DEVICE)
try:
    block_dim = int(torch.npu.get_device_properties(DEVICE).cube_core_num)
except Exception:
    block_dim = 1

print(f"blockDim={block_dim}  N={N}  SUB_BLOCK_DIM={SUB_BLOCK_DIM}")
print()

HERE = os.path.dirname(os.path.abspath(__file__))

VARIANTS = [
    ("TSync_Custom (built-in, buggy)",   "repro_builtin_lib.so"),
    ("MyTSync     (workaround, correct)", "repro_mytync_lib.so"),
]

all_passed = True
for label, so_name in VARIANTS:
    so_path = os.path.join(HERE, so_name)
    if not os.path.exists(so_path):
        print(f"ERROR: {so_name} not found — run compile.sh first", file=sys.stderr)
        sys.exit(1)

    lib = load_lib(so_path)
    gm_input, gm_output, ref = make_tensors(block_dim)
    run(lib, block_dim, gm_input, gm_output)
    report(label, gm_output, ref)
    if not torch.equal(gm_output, ref):
        all_passed = False

print()
print("Summary:", "all passed" if all_passed else "FAILURES — see above")
