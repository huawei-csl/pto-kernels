"""Allocation-free GDN timing and numerical validation on a real NPU.

PYTHONPATH=python timeout 120 python benchmarking/gdn_mega.py --output baseline.json
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_gdn_mega import (  # noqa: E402
    _assert_close, _expand_kv_heads, _make_inputs, _prepare_gdn_mega, _reference,
)
from pto_kernels import pto_gdn_mega  # noqa: E402


def errors(actual, expected):
    actual, expected = actual.float().cpu(), expected.float().cpu()
    diff = actual - expected
    return {
        "max_abs": diff.abs().max().item(),
        "relative_rmse": (diff.square().mean() / expected.square().mean().clamp_min(1e-16)).sqrt().item(),
    }


def benchmark_case(args, length):
    """Check and time one shape; release its graph/tensors on return."""
    inputs = _make_inputs(length, args.heads, args.key_heads)
    inputs[3].mul_(args.gate_scale)
    q, k = _expand_kv_heads(inputs[0], inputs[1], args.heads)
    initial = torch.randn(1, args.heads, 128, 128, dtype=torch.float16) * 0.1 if args.initial_state else None
    expected, expected_state = _reference(q, k, *inputs[2:], initial_state=initial)
    launch = _prepare_gdn_mega(
        *(t.to(args.device) for t in inputs),
        torch.tensor([0, length], dtype=torch.int32, device=args.device),
        initial.to(args.device) if initial is not None else None,
    )
    for _ in range(1 if args.profile_once else 5):
        pto_gdn_mega(**launch)
        torch.npu.synchronize()
        actual = launch["out"].cpu().float() * 128**-0.5
        state = launch["final_state"].cpu().view_as(expected_state)
        _assert_close("out", actual, expected)
        _assert_close("state", state, expected_state)
    if args.profile_once:
        print(json.dumps(dict(length=length, output_error=errors(actual, expected),
                              state_error=errors(state, expected_state))), flush=True)
        return None
    # Capture launches so Python dispatch cannot leave gaps between kernels.
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(args.inner):
            pto_gdn_mega(**launch)
    graph.replay()
    torch.npu.synchronize()
    times = []
    for _ in range(args.trials):
        start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / args.inner)
    record = dict(length=length, heads=args.heads, key_heads=args.key_heads, initial_state=args.initial_state, gate_scale=args.gate_scale,
                  median_us=statistics.median(times), min_us=min(times), max_us=max(times),
                  output_error=errors(actual, expected), state_error=errors(state, expected_state))
    # Verify the timed graph also leaves correct outputs, outside its timing.
    _assert_close("out after replay", launch["out"].cpu().float() * 128**-0.5, expected)
    _assert_close("state after replay", launch["final_state"].cpu().view_as(expected_state), expected_state)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", nargs="+", type=int, default=[128, 256, 1024, 4096])
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--key-heads", type=int, default=4)
    parser.add_argument("--device", default=os.environ.get("NPU_DEVICE", "npu:0"))
    parser.add_argument("--initial-state", action="store_true", help="Benchmark with a nonzero incoming recurrent state")
    parser.add_argument("--gate-scale", type=float, default=1.0, help="Scale log gates to exercise different recurrent memory lengths")
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--inner", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile-once", action="store_true", help="One checked launch, without a timing graph")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = False
    # Verify which implementation the dynamic loader actually selected.
    libraries = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                 if line.split()[-1].endswith("/libno_workspace_kernel.so")}
    if len(libraries) != 1:
        raise RuntimeError(f"Expected one loaded kernel library, found {libraries}")
    library_hash = hashlib.sha256(Path(libraries.pop()).read_bytes()).hexdigest()
    properties = torch.npu.get_device_properties(args.device)
    environment = dict(device_name=properties.name, cube_cores=properties.cube_core_num,
                       vector_cores=properties.vector_core_num, torch_version=torch.__version__,
                       torch_npu_version=torch_npu.__version__)
    records = []
    for length in args.lengths:
        try:
            record = benchmark_case(args, length)
            if record is not None:
                record["status"] = "passed"
        except AssertionError as error:
            # Preserve numerical failures, never assign them a timing/speedup.
            record = dict(length=length, heads=args.heads, key_heads=args.key_heads,
                          initial_state=args.initial_state, gate_scale=args.gate_scale,
                          status="failed", error=str(error))
        if record is not None:
            record["kernel_library_sha256"] = library_hash
            record["environment"] = environment
            records.append(record)
            print(json.dumps(record), flush=True)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(records, indent=2) + "\n")

    if any(record["status"] == "failed" for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
